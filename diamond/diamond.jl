# Reversible simulation for neural network model of diamond
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT
# An appropriate conda environment should be activated
# Use CUDA_VISIBLE_DEVICES to set the GPU to run on
# Occasionally runs into PyGIL error, a known issue with PythonCall
# See also https://github.com/tummfm/difftre/blob/main/diamond.ipynb

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Molly
using Enzyme
using PythonCall
using TimerOutputs
using Optimisers
using CairoMakie
using BSON
import Zygote
import AtomsCalculators
import SimpleCrystals
import ChainRulesCore
import ChainRulesCore.NoTangent
using Dates
using LinearAlgebra
using Random
using Statistics
using Test

const out_dir = (length(ARGS) >= 1 ? ARGS[1] : "train_diamond")

const pymod_custom_energy = pyimport("DiffTRe.custom_energy")
const pymod_custom_space = pyimport("DiffTRe.custom_space")
const pymod_jax_md = pyimport("jax_md")
const pymod_jax = pyimport("jax")
const pymod_optax = pyimport("optax")
const pymod_pickle = pyimport("pickle")
const py_load_configuration = pyimport("DiffTRe.io").load_configuration

const T = Float64
const n_atoms = 1000
const n_atoms_x3 = 3 * n_atoms
const atom_mass = T(12.011)
const side_length = T(1.784) # nm
const side_offset = T(1e-12)
const boundary = TriclinicBoundary(
    SVector(side_length, zero(T)    , zero(T)    ),
    SVector(side_offset, side_length, zero(T)    ),
    SVector(side_offset, side_offset, side_length),
)
const Ω = box_volume(boundary) # nm^3
const temp = T(298.0) # K
const β = inv(ustrip(u"kJ * mol^-1", Unitful.R * temp * u"K")) # kJ^-1 * mol
const init_params_sw = T[7.049556277, 0.6022245584, 1.80, 21.0, 1.20, 0.14, 200.0]
const n_params_sw = length(init_params_sw)
const c_target = [649788.6, 74674.51, 348079.53] # kJ * mol^-1 * nm^-3, GPa in Thaler2021
const conv_to_GPa = ustrip(u"GPa", one(T) * u"kJ * mol^-1 * nm^-3" / Unitful.Na)
const γσ = T(5e-8) # Loss weight stress, kJ^-2 * mol^2 * nm^6
const γc = T(1e-10) # Loss weight stiffness, kJ^-2 * mol^2 * nm^6
const stiff_av_weight = T(0.0) # Weighting on stiffness stress tensor terms
const to = TimerOutput()

coords_copy(sys, neighbors=nothing; kwargs...) = copy(sys.coords)
velocities_copy(sys, neighbors=nothing; kwargs...) = copy(sys.velocities)

function CoordCopyLogger(n_steps_log)
    return GeneralObservableLogger(coords_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)
end

function VelocityCopyLogger(n_steps_log)
    return GeneralObservableLogger(velocities_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)
end

const py_box_tensor = pymod_jax.numpy.array(pylist([
    [side_length, zero(T)    , zero(T)    ],
    [zero(T)    , side_length, zero(T)    ],
    [zero(T)    , zero(T)    , side_length],
]))
const py_displacement = pymod_jax_md.space.periodic_general(py_box_tensor)[0]

const py_neighbor_fn = pymod_jax_md.partition.neighbor_list(
    py_displacement,
    pymod_jax.numpy.ones(3),
    0.14 * 1.80,
    dr_threshold=0.05,
    capacity_multiplier=1.5,
    disable_cell_list=pybuiltins.True,
)

const py_R_init_nofrac, _, py_box_init = py_load_configuration(gethostname() == "jgreener-lmb" ?
                                    "/home/jgreener/soft/difftre/data/confs/Diamond.pdb" :
                                    "/lmb/home/jgreener/soft/difftre/data/confs/Diamond.pdb")
const py_R_init = pymod_custom_space.scale_to_fractional_coordinates(py_R_init_nofrac, py_box_init)[0]
const py_neighbors_init = py_neighbor_fn(py_R_init)

const py_init_fn, py_GNN_energy = pymod_custom_energy.DimeNetPP_neighborlist(py_displacement,
                                                py_R_init, py_neighbors_init, 0.2)
const py_GNN_energy_jit = pymod_jax.jit(py_GNN_energy)

const py_random_key = pymod_jax.random.PRNGKey(rand(Int32))
const py_model_init_key = pymod_jax.random.split(py_random_key, 2)[0]

struct GNNParams{P}
    py_params::P # Py or Nothing
end

const pyscript_params = """
from haiku._src.data_structures import FlatMapping
global FlatMapping

def py_add_params(p1, p2):
    p_sum = {}
    assert p1.keys() == p2.keys()
    for k in p1:
        if type(p1[k]) is FlatMapping:
            p_sum[k] = py_add_params(p1[k], p2[k])
        else:
            p_sum[k] = p1[k] + p2[k]
    return FlatMapping(p_sum)

def py_multiply_params(p, y):
    p_multiply = {}
    for k in p:
        if type(p[k]) is FlatMapping:
            p_multiply[k] = py_multiply_params(p[k], y)
        else:
            p_multiply[k] = p[k] * y
    return FlatMapping(p_multiply)
"""

const py_params_nt = pyexec(
    @NamedTuple{py_add_params, py_multiply_params},
    pyscript_params,
    Main,
)

const py_add_params = py_params_nt.py_add_params
const py_multiply_params = py_params_nt.py_multiply_params

Base.:+(p1::GNNParams{Py}, p2::GNNParams{Py}) = GNNParams(py_add_params(p1.py_params, p2.py_params))
Base.:+(p1::GNNParams{Nothing}, p2::GNNParams{Nothing}) = GNNParams(nothing)
Base.:+(p1::GNNParams{Py}, p2::GNNParams{Nothing}) = p1
Base.:+(p1::GNNParams{Nothing}, p2::GNNParams{Py}) = p2

function Base.:*(p::GNNParams{Py}, y)
    if y == 1
        return p
    else
        return GNNParams(py_multiply_params(p.py_params, y))
    end
end

Base.:*(p::GNNParams{Nothing}, y) = GNNParams(nothing)

gnn_example_array(py_params) = py_params["Energy/~/Output_3/~/Dense_Series_1"]["w"]

function Base.show(io::IO, p::GNNParams{Py})
    # Show a representative set of parameters
    rep_param = pyconvert(T, gnn_example_array(p.py_params).sum().item())
    print(io, "GNNParams(", rep_param, ")")
end

Base.show(io::IO, ::GNNParams{Nothing}) = print(io, "GNNParams(nothing)")

struct DimeNetPP
    params::GNNParams
end

const py_gnn_init_params = py_init_fn(py_model_init_key, py_R_init, neighbor=py_neighbors_init)
const gnn_init_params = GNNParams(py_gnn_init_params)

const pyscript_loss = """
# See https://github.com/tummfm/difftre/blob/main/diamond.ipynb

import jax
global jax
from jax_md import space
global space
from DiffTRe import custom_quantity
global custom_quantity
from functools import partial
global partial

global GNN_energy_global
GNN_energy_global = GNN_energy

def energy_fn_template(energy_params):
    gnn_energy = partial(GNN_energy_global, energy_params)
    def energy(R, neighbor, **dynamic_kwargs):
        return gnn_energy(R, neighbor=neighbor, **dynamic_kwargs)
    return energy

global energy_fn_template_global
energy_fn_template_global = energy_fn_template

def virial_stress_tensor(frac_position, mass, velocity, neighbor, energy_params, box_tensor):
    energy_fn = energy_fn_template_global(energy_params)
    energy_fn_without_kwargs = lambda R, neighbor, box_tensor: energy_fn(R, neighbor=neighbor, box=box_tensor)
    R = frac_position # In unit box if fractional coordinates used
    negative_forces, box_gradient = jax.grad(energy_fn_without_kwargs, argnums=[0, 2])(R, neighbor, box_tensor)
    R = space.transform(box_tensor, R) # Transform back to real positions
    force_contribution = jax.numpy.dot(negative_forces.T, R)
    box_contribution = jax.numpy.dot(box_gradient.T, box_tensor)
    virial_tensor = force_contribution + box_contribution
    spatial_dim = frac_position.shape[-1]
    volume = custom_quantity.box_volume(box_tensor, spatial_dim)
    kinetic_tensor = custom_quantity.kinetic_energy_tensor(mass, velocity)
    return (kinetic_tensor + virial_tensor) / volume

global virial_stress_tensor_global
virial_stress_tensor_global = virial_stress_tensor

def loss(frac_position, mass, velocity, neighbor, energy_params, box_tensor):
    st = virial_stress_tensor_global(frac_position, mass, velocity, neighbor, energy_params, box_tensor)
    return (st * st).sum() * $γσ / 9

py_loss_jit = jax.jit(loss)

def forces_sum(params, frac_coords, neighbors, box_tensor, d_fs):
    fs = -jax.grad(jax.jit(GNN_energy_global), argnums=[1])(
                params, frac_coords, neighbors, box=box_tensor)[0]
    return (fs * d_fs).sum()

py_forces_sum_jit = jax.jit(forces_sum)
"""

const py_loss_nt = pyexec(
    @NamedTuple{py_loss_jit, py_forces_sum_jit},
    pyscript_loss,
    Main,
    (GNN_energy=py_GNN_energy,),
)

const py_loss_jit = py_loss_nt.py_loss_jit
const py_forces_sum_jit = py_loss_nt.py_forces_sum_jit

function svecs_to_jax_array(svec_arr)
    return pymod_jax.numpy.array(reinterpret(T, svec_arr)).reshape(n_atoms, 3)
end

function forces_gnn(params, frac_coords, py_neighbors)
    py_frac_coords = svecs_to_jax_array(frac_coords)
    py_fs = -pymod_jax.grad(py_GNN_energy_jit, argnums=[1])(
                        params.py_params, py_frac_coords, py_neighbors, box=py_box_tensor)[0]
    return SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_fs)))
end

function ChainRulesCore.rrule(::typeof(forces_gnn), params, frac_coords, py_neighbors)
    Y = forces_gnn(params, frac_coords, py_neighbors)

    function forces_gnn_pullback(d_fs)
        py_frac_coords = svecs_to_jax_array(frac_coords)
        py_d_fs = svecs_to_jax_array(d_fs)
        py_dl_dp, py_dl_dfc = pymod_jax.grad(py_forces_sum_jit, argnums=[0, 1])(
                    params.py_params, py_frac_coords, py_neighbors, py_box_tensor, py_d_fs)
        dl_dp = GNNParams(py_dl_dp)
        dl_dfc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dfc)))
        return NoTangent(), dl_dp, dl_dfc, NoTangent()
    end

    return Y, forces_gnn_pullback
end

function AtomsCalculators.forces(sys, inter::DimeNetPP; py_neighbors=nothing, kwargs...)
    frac_coords = sys.coords ./ side_length
    if isnothing(py_neighbors)
        py_neighbors_used = Zygote.ignore() do
            py_neighbor_fn(svecs_to_jax_array(frac_coords))
        end
    else
        py_neighbors_used = py_neighbors
    end
    return forces_gnn(inter.params, frac_coords, py_neighbors_used)
end

function AtomsCalculators.potential_energy(sys, inter::DimeNetPP; py_neighbors=nothing, kwargs...)
    frac_coords = sys.coords ./ side_length
    if isnothing(py_neighbors)
        py_neighbors_used = Zygote.ignore() do
            py_neighbor_fn(svecs_to_jax_array(frac_coords))
        end
    else
        py_neighbors_used = py_neighbors
    end
    E = potential_energy_gnn(inter.params, frac_coords, py_neighbors_used)
    return E
end

function potential_energy_gnn(params, frac_coords, py_neighbors)
    py_frac_coords = svecs_to_jax_array(frac_coords)
    E = py_GNN_energy_jit(params.py_params, py_frac_coords, py_neighbors, box=py_box_tensor)
    return pyconvert(T, E.item())
end

function ChainRulesCore.rrule(::typeof(potential_energy_gnn), params, frac_coords, py_neighbors)
    Y = potential_energy_gnn(params, frac_coords, py_neighbors)

    function potential_energy_gnn_pullback(dy)
        py_frac_coords = svecs_to_jax_array(frac_coords)
        py_dl_dp, py_dl_dfc = pymod_jax.grad(py_GNN_energy_jit, argnums=[0, 1])(
                    params.py_params, py_frac_coords, py_neighbors, box=py_box_tensor)
        dl_dp = GNNParams(py_dl_dp) * dy
        dl_dfc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dfc))) .* dy
        return NoTangent(), dl_dp, dl_dfc, NoTangent()
    end

    return Y, potential_energy_gnn_pullback
end

struct StillingerWeber{T}
    A::T
    B::T
    p::Int
    q::Int
    a::T
    λ::T
    γ::T
    σ::T
    ϵ::T
end

function Base.zero(::StillingerWeber{T}) where T
    return StillingerWeber(zero(T), zero(T), 0, 0, zero(T), zero(T), zero(T), zero(T), zero(T))
end

function Base.:+(sw1::StillingerWeber, sw2::StillingerWeber)
    StillingerWeber(
        sw1.A + sw2.A,
        sw1.B + sw2.B,
        sw1.p + sw2.p,
        sw1.q + sw2.q,
        sw1.a + sw2.a,
        sw1.λ + sw2.λ,
        sw1.γ + sw2.γ,
        sw1.σ + sw2.σ,
        sw1.ϵ + sw2.ϵ,
    )
end

function ChainRulesCore.rrule(T::Type{<:StillingerWeber}, xs...)
    Y = T(xs...)
    function StillingerWeber_pullback(Ȳ)
        return NoTangent(), Ȳ.A, Ȳ.B, NoTangent(), NoTangent(), Ȳ.a, Ȳ.λ, Ȳ.γ, Ȳ.σ, Ȳ.ϵ
    end
    return Y, StillingerWeber_pullback
end

# Also modify the function below
function potential_energy_sw(sw, coords, boundary, neighbors)
    A, B, p, q, a, λ, γ, σ, ϵ = sw.A, sw.B, sw.p, sw.q, sw.a, sw.λ, sw.γ, sw.σ, sw.ϵ
    E = zero(T)
    for ni in 1:length(neighbors)
        i, j, ks = neighbors[ni]
        rij = norm(vector(coords[i], coords[j], boundary))
        r = rij / σ
        if r < a
            pe = ϵ * A * (B * inv(r^p) - inv(r^q)) * exp(inv(r - a))
            E += pe
        end

        for k in ks
            rij_trip = r
            rik_trip = norm(vector(coords[i], coords[k], boundary)) / σ
            rjk_trip = norm(vector(coords[j], coords[k], boundary)) / σ

            if rij_trip < a && rik_trip < a
                θ_jik = bond_angle(coords[j], coords[i], coords[k], boundary)
                E += ϵ * λ * exp(γ * inv(rij_trip - a) + γ * inv(rik_trip - a)) * (cos(θ_jik) + T(1/3))^2
            end
            if rij_trip < a && rjk_trip < a
                θ_ijk = bond_angle(coords[i], coords[j], coords[k], boundary)
                E += ϵ * λ * exp(γ * inv(rij_trip - a) + γ * inv(rjk_trip - a)) * (cos(θ_ijk) + T(1/3))^2
            end
            if rik_trip < a && rjk_trip < a
                θ_ikj = bond_angle(coords[i], coords[k], coords[j], boundary)
                E += ϵ * λ * exp(γ * inv(rik_trip - a) + γ * inv(rjk_trip - a)) * (cos(θ_ikj) + T(1/3))^2
            end
        end
    end

    return E
end

# A copy of the above but inlined to avoid an Enzyme error
@inline function potential_energy_sw_inline(sw, coords, boundary, neighbors)
    A, B, p, q, a, λ, γ, σ, ϵ = sw.A, sw.B, sw.p, sw.q, sw.a, sw.λ, sw.γ, sw.σ, sw.ϵ
    E = zero(T)
    for ni in 1:length(neighbors)
        i, j, ks = neighbors[ni]
        rij = norm(vector(coords[i], coords[j], boundary))
        r = rij / σ
        if r < a
            pe = ϵ * A * (B * inv(r^p) - inv(r^q)) * exp(inv(r - a))
            E += pe
        end

        for k in ks
            rij_trip = r
            rik_trip = norm(vector(coords[i], coords[k], boundary)) / σ
            rjk_trip = norm(vector(coords[j], coords[k], boundary)) / σ

            if rij_trip < a && rik_trip < a
                θ_jik = bond_angle(coords[j], coords[i], coords[k], boundary)
                E += ϵ * λ * exp(γ * inv(rij_trip - a) + γ * inv(rik_trip - a)) * (cos(θ_jik) + T(1/3))^2
            end
            if rij_trip < a && rjk_trip < a
                θ_ijk = bond_angle(coords[i], coords[j], coords[k], boundary)
                E += ϵ * λ * exp(γ * inv(rij_trip - a) + γ * inv(rjk_trip - a)) * (cos(θ_ijk) + T(1/3))^2
            end
            if rik_trip < a && rjk_trip < a
                θ_ikj = bond_angle(coords[i], coords[k], coords[j], boundary)
                E += ϵ * λ * exp(γ * inv(rik_trip - a) + γ * inv(rjk_trip - a)) * (cos(θ_ikj) + T(1/3))^2
            end
        end
    end

    return E
end

function ChainRulesCore.rrule(::typeof(potential_energy_sw), sw, coords, boundary, neighbors)
    Y = potential_energy_sw(sw, coords, boundary, neighbors)

    function potential_energy_sw_pullback(dy)
        d_coords = zero(coords)

        grads = autodiff(
            Enzyme.Reverse,
            potential_energy_sw,
            Active,
            Active(sw),
            Duplicated(coords, d_coords),
            Const(boundary),
            Const(neighbors),
        )[1]

        d_sw = StillingerWeber(grads[1].A * dy, grads[1].B * dy, grads[1].p, grads[1].q,
                               grads[1].a * dy, grads[1].λ * dy, grads[1].γ * dy, grads[1].σ * dy,
                               grads[1].ϵ * dy)
        return NoTangent(), d_sw, d_coords .* dy, NoTangent(), NoTangent()
    end

    return Y, potential_energy_sw_pullback
end

function forces_sw!(fs_chunks, sw, coords, boundary, neighbors, n_threads=Threads.nthreads())
    A, B, p, q, a, λ, γ, σ, ϵ = sw.A, sw.B, sw.p, sw.q, sw.a, sw.λ, sw.γ, sw.σ, sw.ϵ
    n_chunks = n_threads
    PythonCall.GC.disable()
    Threads.@threads for chunk_i in 1:n_chunks
        @inbounds for ni in chunk_i:n_chunks:length(neighbors)
            i, j, ks = neighbors[ni]
            dr = vector(coords[i], coords[j], boundary)
            rij = norm(dr)
            r = rij / σ
            if r < a
                df_dr = -A * exp(inv(r - a)) * (B * p * inv(r^(p+1)) + (B * inv(r^p) - 1) * inv((r - a)^2))
                f = -ϵ * df_dr * inv(σ)
                fdr = f * dr / rij
                fs_chunks[chunk_i][i] -= fdr
                fs_chunks[chunk_i][j] += fdr
            end

            trip_cutoff = a * σ
            trip_cutoff_2 = 2 * trip_cutoff
            for k in ks
                dr_ik = vector(coords[i], coords[k], boundary)
                norm_dr_ik = norm(dr_ik)
                (norm_dr_ik > trip_cutoff_2) && continue
                dr_jk = vector(coords[j], coords[k], boundary)
                norm_dr_jk = norm(dr_jk)
                ((norm_dr_ik < trip_cutoff) || (norm_dr_jk < trip_cutoff)) || continue
                dr_ij, norm_dr_ij = dr, rij
                ndr_ij, ndr_ik, ndr_jk = dr_ij / norm_dr_ij, dr_ik / norm_dr_ik, dr_jk / norm_dr_jk
                dot_ij_ik, dot_ji_jk, dot_ki_kj = dot(dr_ij, dr_ik), dot(-dr_ij, dr_jk), dot(dr_ik, dr_jk)
                rij_trip = r
                rik_trip = norm_dr_ik / σ
                rjk_trip = norm_dr_jk / σ

                if rij_trip < a && rik_trip < a
                    cos_θ_jik = dot_ij_ik / (norm_dr_ij * norm_dr_ik)
                    exp_term = exp(γ * inv(rij_trip - a) + γ * inv(rik_trip - a))
                    cos_term = cos_θ_jik + T(1/3)
                    dh_term = λ * cos_term^2 * -γ * exp_term
                    dh_drij = inv((rij_trip - a)^2) * dh_term
                    dh_drik = inv((rik_trip - a)^2) * dh_term
                    dh_dcosθjik = 2 * λ * exp_term * cos_term
                    dcosθ_drj = (dr_ik * norm_dr_ij^2 - dr_ij * dot_ij_ik) / (norm_dr_ij^3 * norm_dr_ik)
                    dcosθ_drk = (dr_ij * norm_dr_ik^2 - dr_ik * dot_ij_ik) / (norm_dr_ik^3 * norm_dr_ij)
                    fj = -ϵ * (dh_drij * ndr_ij / σ + dh_dcosθjik * dcosθ_drj)
                    fk = -ϵ * (dh_drik * ndr_ik / σ + dh_dcosθjik * dcosθ_drk)
                    fs_chunks[chunk_i][i] -= fj + fk
                    fs_chunks[chunk_i][j] += fj
                    fs_chunks[chunk_i][k] += fk
                end
                if rij_trip < a && rjk_trip < a
                    cos_θ_ijk = dot_ji_jk / (norm_dr_ij * norm_dr_jk)
                    exp_term = exp(γ * inv(rij_trip - a) + γ * inv(rjk_trip - a))
                    cos_term = cos_θ_ijk + T(1/3)
                    dh_term = λ * cos_term^2 * -γ * exp_term
                    dh_drij = inv((rij_trip - a)^2) * dh_term
                    dh_drjk = inv((rjk_trip - a)^2) * dh_term
                    dh_dcosθijk = 2 * λ * exp_term * cos_term
                    dcosθ_dri = ( dr_jk * norm_dr_ij^2 + dr_ij * dot_ji_jk) / (norm_dr_ij^3 * norm_dr_jk)
                    dcosθ_drk = (-dr_ij * norm_dr_jk^2 - dr_jk * dot_ji_jk) / (norm_dr_jk^3 * norm_dr_ij)
                    fi = -ϵ * (dh_drij * -ndr_ij / σ + dh_dcosθijk * dcosθ_dri)
                    fk = -ϵ * (dh_drjk *  ndr_jk / σ + dh_dcosθijk * dcosθ_drk)
                    fs_chunks[chunk_i][j] -= fi + fk
                    fs_chunks[chunk_i][i] += fi
                    fs_chunks[chunk_i][k] += fk
                end
                if rik_trip < a && rjk_trip < a
                    cos_θ_ikj = dot_ki_kj / (norm_dr_ik * norm_dr_jk)
                    exp_term = exp(γ * inv(rik_trip - a) + γ * inv(rjk_trip - a))
                    cos_term = cos_θ_ikj + T(1/3)
                    dh_term = λ * cos_term^2 * -γ * exp_term
                    dh_drik = inv((rik_trip - a)^2) * dh_term
                    dh_drjk = inv((rjk_trip - a)^2) * dh_term
                    dh_dcosθikj = 2 * λ * exp_term * cos_term
                    dcosθ_dri = (-dr_jk * norm_dr_ik^2 + dr_ik * dot_ki_kj) / (norm_dr_ik^3 * norm_dr_jk)
                    dcosθ_drj = (-dr_ik * norm_dr_jk^2 + dr_jk * dot_ki_kj) / (norm_dr_jk^3 * norm_dr_ik)
                    fi = -ϵ * (dh_drik * -ndr_ik / σ + dh_dcosθikj * dcosθ_dri)
                    fj = -ϵ * (dh_drjk * -ndr_jk / σ + dh_dcosθikj * dcosθ_drj)
                    fs_chunks[chunk_i][k] -= fi + fj
                    fs_chunks[chunk_i][i] += fi
                    fs_chunks[chunk_i][j] += fj
                end
            end
        end
    end
    PythonCall.GC.enable()

    return nothing
end

function forces_sw(sw, coords, boundary, neighbors, n_threads=Threads.nthreads())
    fs_chunks = [zero(coords) for _ in 1:n_threads]
    forces_sw!(fs_chunks, sw, coords, boundary, neighbors, n_threads)
    return sum(fs_chunks)
end

function ChainRulesCore.rrule(::typeof(forces_sw), sw, coords, boundary, neighbors, n_threads)
    Y = forces_sw(sw, coords, boundary, neighbors, n_threads)

    function forces_sw_pullback(d_fs)
        fs_chunks = [zero(coords) for _ in 1:n_threads]
        d_fs_chunks = [copy(d_fs) for _ in 1:n_threads]
        d_coords = zero(coords)

        grads = autodiff(
            Enzyme.Reverse,
            forces_sw!,
            Const,
            Duplicated(fs_chunks, d_fs_chunks),
            Active(sw),
            Duplicated(coords, d_coords),
            Const(boundary),
            Const(neighbors),
            Const(n_threads),
        )[1]

        return NoTangent(), grads[2], d_coords, NoTangent(), NoTangent(), NoTangent()
    end

    return Y, forces_sw_pullback
end

function AtomsCalculators.forces(sys, inter::StillingerWeber; neighbors,
                                 n_threads=Threads.nthreads(), kwargs...)
    return forces_sw(inter, sys.coords, sys.boundary, neighbors, n_threads)
end

function AtomsCalculators.potential_energy(sys, inter::StillingerWeber; neighbors, kwargs...)
    E = potential_energy_sw(inter, sys.coords, sys.boundary, neighbors)
    return E
end

# Differs from standard neighbor list by returning triples (i, j, k)
struct TripletNeighborFinder{D}
    dist_cutoff_pair::D
    dist_cutoff_triplet::D
    n_steps::Int
end

function Molly.find_neighbors(sys,
                              nf::TripletNeighborFinder,
                              current_neighbors=nothing,
                              step_n::Integer=0,
                              force_recompute::Bool=false;
                              n_threads::Integer=Threads.nthreads())
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    sqdist_cutoff_pair = nf.dist_cutoff_pair^2
    sqdist_cutoff_triplet, sqdist_cutoff_triplet_2 = nf.dist_cutoff_triplet^2, (2 * nf.dist_cutoff_triplet)^2
    n_chunks = n_threads
    neighbors_list_chunks = [Tuple{Int32, Int32, Vector{Int32}}[] for _ in 1:n_chunks]
    PythonCall.GC.disable()
    Threads.@threads for chunk_i in 1:n_chunks
        @inbounds for pi in chunk_i:n_chunks:Molly.n_atoms_to_n_pairs(n_atoms)
            i, j = Molly.pair_index(n_atoms, pi)
            dr_ij = vector(sys.coords[i], sys.coords[j], sys.boundary)
            r2_ij = sum(abs2, dr_ij)
            if r2_ij <= sqdist_cutoff_pair
                third_atoms = Int32[]
                for k in (j + 1):n_atoms
                    dr_ik = vector(sys.coords[i], sys.coords[k], sys.boundary)
                    r2_ik = sum(abs2, dr_ik)
                    (r2_ik < sqdist_cutoff_triplet_2) || continue
                    dr_jk = vector(sys.coords[j], sys.coords[k], sys.boundary)
                    r2_jk = sum(abs2, dr_jk)
                    if r2_ik < sqdist_cutoff_triplet || r2_jk < sqdist_cutoff_triplet
                        push!(third_atoms, k)
                    end
                end
                push!(neighbors_list_chunks[chunk_i], (i, j, third_atoms))
            end
        end
    end
    PythonCall.GC.enable()

    neighbors_list = Tuple{Int32, Int32, Vector{Int32}}[]
    for chunk_i in 1:n_chunks
        append!(neighbors_list, neighbors_list_chunks[chunk_i])
    end

    return NeighborList(length(neighbors_list), neighbors_list)
end

const neighbor_finder_fwd = TripletNeighborFinder(0.6  , 0.35 , 10)
const neighbor_finder_rev = TripletNeighborFinder(0.504, 0.252, 1 )

function outer_products(vels)
    out = zeros(T, n_atoms, 3, 3)
    @inbounds for ai in 1:n_atoms
        sv = vels[ai]
        for i in 1:3, j in 1:3
            out[ai, i, j] = sv[i] * sv[j]
        end
    end
    return out
end

function mat_mul(Ft, R)
    FtR = zeros(T, 3, 3)
    @inbounds for i in 1:3, j in 1:3
        s = zero(T)
        for k in axes(Ft, 2)
            s += Ft[i, k] * R[k, j]
        end
        FtR[i, j] = s
    end
    return FtR
end

@inline function scale_frac_coord(fc, boundary)
    bv = boundary.basis_vectors
    return SVector(
        fc[1] * bv[1][1] + fc[2] * bv[2][1] + fc[3] * bv[3][1],
        fc[1] * bv[1][2] + fc[2] * bv[2][2] + fc[3] * bv[3][2],
        fc[1] * bv[1][3] + fc[2] * bv[2][3] + fc[3] * bv[3][3],
    )
end

# Also modify the function below
function potential_energy_sw_frac_coords(sw, frac_coords, boundary, neighbors)
    coords = zero(frac_coords)
    for i in 1:n_atoms
        coords[i] = scale_frac_coord(frac_coords[i], boundary)
    end
    return potential_energy_sw(sw, coords, boundary, neighbors)
end

# A copy of the above but inlined to avoid an Enzyme error
@inline function potential_energy_sw_frac_coords_inline(sw, frac_coords, boundary, neighbors)
    coords = zero(frac_coords)
    for i in 1:n_atoms
        coords[i] = scale_frac_coord(frac_coords[i], boundary)
    end
    return potential_energy_sw_inline(sw, coords, boundary, neighbors)
end

function dU_dh(sw, coords, boundary, neighbors)
    box_pe_grads = zeros(T, 9)
    zt, ot = zero(T), one(T)

    for i in 1:9
        d_boundary = TriclinicBoundary{T, true, T, T}(
            SVector(
                SVector(i == 1 ? ot : zt, i == 2 ? ot : zt, i == 3 ? ot : zt),
                SVector(i == 4 ? ot : zt, i == 5 ? ot : zt, i == 6 ? ot : zt),
                SVector(i == 7 ? ot : zt, i == 8 ? ot : zt, i == 9 ? ot : zt),
            ),
            zt, zt, zt, SVector(zt, zt, zt), zt, zt, zt, zt, zt,
        )
        frac_coords = coords ./ side_length
        box_pe_grads[i] = autodiff_deferred(
            Enzyme.Forward,
            potential_energy_sw_frac_coords,
            DuplicatedNoNeed,
            Const(sw),
            Const(frac_coords),
            Duplicated(boundary, d_boundary),
            Const(neighbors),
        )[1]
    end

    return box_pe_grads
end

function loss_stress_gnn(sys)
    frac_coords = sys.coords ./ side_length
    py_neighbors = Zygote.ignore() do
        py_neighbor_fn(svecs_to_jax_array(frac_coords))
    end
    params = sys.general_inters.gnn.params
    return loss_stress_gnn(frac_coords, sys.velocities, py_neighbors, params)
end

function loss_stress_gnn(frac_coords, velocities, py_neighbors, params)
    py_stress = py_loss_jit(
        svecs_to_jax_array(frac_coords),
        atom_mass,
        svecs_to_jax_array(velocities),
        py_neighbors,
        params.py_params,
        py_box_tensor,
    )
    return pyconvert(T, py_stress.item())
end

function ChainRulesCore.rrule(::typeof(loss_stress_gnn), frac_coords, velocities,
                              py_neighbors, params)
    Y = loss_stress_gnn(frac_coords, velocities, py_neighbors, params)

    function loss_stress_gnn_pullback(dy)
        py_dl_dfc, py_dl_dv, py_dl_dp = pymod_jax.grad(py_loss_jit, argnums=[0, 2, 4])(
            svecs_to_jax_array(frac_coords),
            atom_mass,
            svecs_to_jax_array(velocities),
            py_neighbors,
            params.py_params,
            py_box_tensor,
        )
        dl_dfc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dfc))) .* dy
        dl_dv  = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dv ))) .* dy
        dl_dp = GNNParams(py_dl_dp) * dy
        return NoTangent(), dl_dfc, dl_dv, NoTangent(), dl_dp
    end

    return Y, loss_stress_gnn_pullback
end

function virial_stress_tensor(sw, coords, velocities, boundary, atom_masses,
                              neighbors, vel_term=true)
    thermal_excitation_velocity = velocities .- (mean(velocities),)
    velocity_tensors = outer_products(thermal_excitation_velocity)
    kinetic_tensor = -dropdims(sum(atom_masses .* velocity_tensors; dims=1); dims=1)

    fs = forces_sw(sw, coords, boundary, neighbors)
    Ft = reshape(reinterpret(T, fs), 3, :)
    R = reshape(reinterpret(T, coords), 3, :)'
    FtR = mat_mul(Ft, R)

    bvs = boundary.basis_vectors
    box_tensor_flat = [bvs[1]..., bvs[2]..., bvs[3]...]
    box_pe_grads = dU_dh(sw, coords, boundary, neighbors)
    # Triclinic 3x3 matrix is upper triangular
    box_contribution = reshape(box_pe_grads, 3, 3)' * reshape(box_tensor_flat, 3, 3)

    virial_tensor = -FtR .+ box_contribution
    if vel_term
        return (kinetic_tensor .+ virial_tensor) ./ Ω
    else
        return virial_tensor ./ Ω
    end
end

function loss_stress_sw(sys, vel_term=true)
    return loss_stress_sw(sys.general_inters.sw, sys.coords, sys.velocities, sys.boundary,
                          masses(sys), find_neighbors(sys), vel_term)
end

function loss_stress_sw(sw, coords, velocities, boundary, atom_masses, neighbors, vel_term)
    σij = virial_stress_tensor(sw, coords, velocities, boundary, atom_masses, neighbors, vel_term)
    return sum(abs2, σij) * γσ / 9
end

function ChainRulesCore.rrule(::typeof(loss_stress_sw), sw, coords, velocities, boundary,
                              atom_masses, neighbors, vel_term)
    Y = loss_stress_sw(sw, coords, velocities, boundary, atom_masses, neighbors, vel_term)

    function loss_stress_sw_pullback(dy)
        d_coords = zero(coords)
        d_velocities = zero(velocities)
        grads = autodiff_deferred(
            Enzyme.Reverse,
            loss_stress_sw,
            Active,
            Active(sw),
            Duplicated(coords, d_coords),
            Duplicated(velocities, d_velocities),
            Const(boundary),
            Const(atom_masses),
            Const(neighbors),
            Const(vel_term),
        )[1]
        d_sw = StillingerWeber(grads[1].A * dy, grads[1].B * dy, grads[1].p, grads[1].q,
                               grads[1].a * dy, grads[1].λ * dy, grads[1].γ * dy, grads[1].σ * dy,
                               grads[1].ϵ * dy)
        return NoTangent(), d_sw, d_coords .* dy, d_velocities .* dy, NoTangent(),
               NoTangent(), NoTangent(), NoTangent()
    end

    return Y, loss_stress_sw_pullback
end

approx_stress_diag(loss_val_stress) = conv_to_GPa * sqrt((loss_val_stress / (γσ / 9)) / 3)

function potential_energy_sw_eps(sws, coords, boundary, neighbors, epsilon)
    frac_coords = coords ./ side_length
    zt, ot = zero(T), one(T)
    bv, ep = boundary.basis_vectors, epsilon
    boundary_eps = TriclinicBoundary{T, true, T, T}(
        SVector(
            SVector(
                bv[1][1] * (ep[1, 1] + ot) + bv[1][2] * ep[2, 1] + bv[1][3] * ep[3, 1],
                bv[1][1] * ep[1, 2] + bv[1][2] * (ep[2, 2] + ot) + bv[1][3] * ep[3, 2],
                bv[1][1] * ep[1, 3] + bv[1][2] * ep[2, 3] + bv[1][3] * (ep[3, 3] + ot),
            ),
            SVector(
                bv[2][1] * (ep[1, 1] + ot) + bv[2][2] * ep[2, 1] + bv[2][3] * ep[3, 1],
                bv[2][1] * ep[1, 2] + bv[2][2] * (ep[2, 2] + ot) + bv[2][3] * ep[3, 2],
                bv[2][1] * ep[1, 3] + bv[2][2] * ep[2, 3] + bv[2][3] * (ep[3, 3] + ot),
            ),
            SVector(
                bv[3][1] * (ep[1, 1] + ot) + bv[3][2] * ep[2, 1] + bv[3][3] * ep[3, 1],
                bv[3][1] * ep[1, 2] + bv[3][2] * (ep[2, 2] + ot) + bv[3][3] * ep[3, 2],
                bv[3][1] * ep[1, 3] + bv[3][2] * ep[2, 3] + bv[3][3] * (ep[3, 3] + ot),
            ),
        ),
        zt, zt, zt, boundary.reciprocal_size, zt, zt, zt, zt, zt,
    )
    return potential_energy_sw_frac_coords_inline(sws[1], frac_coords, boundary_eps, neighbors)
end

function dU_dϵ(sws, coords, boundary, neighbors, epsilon, i, j)
    d_epsilon = zeros(T, 3, 3)
    d_epsilon[i, j] = one(T)
    return autodiff_deferred(
        Enzyme.Forward,
        potential_energy_sw_eps,
        DuplicatedNoNeed,
        Const(sws),
        Const(coords),
        Const(boundary),
        Const(neighbors),
        Duplicated(epsilon, d_epsilon),
    )[1]
end

function dU_dϵ(sws, coords, boundary, neighbors, epsilon)
    dU_dϵ_out = zeros(T, 3, 3)
    for i in 1:3, j in 1:3
        dU_dϵ_out[i, j] = dU_dϵ(sws, coords, boundary, neighbors, epsilon, i, j)
    end
    return dU_dϵ_out
end

function d2U_dϵ2(sws, coords, boundary, neighbors, epsilon)
    d2U_dϵ2_out = zeros(T, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        d_epsilon = zeros(T, 3, 3)
        d_epsilon[i, j] = one(T)
        d2U_dϵ2_out[l, k, j, i] = autodiff_deferred(
            Enzyme.Forward,
            dU_dϵ,
            DuplicatedNoNeed,
            Const(sws),
            Const(coords),
            Const(boundary),
            Const(neighbors),
            Duplicated(epsilon, d_epsilon),
            Const(k),
            Const(l),
        )[1]
    end
    return d2U_dϵ2_out
end

function loss_stiff_sw_p1(sys)
    return loss_stiff_sw_p1([sys.general_inters.sw], sys.coords, sys.boundary, find_neighbors(sys))
end

function loss_stiff_sw_p1(sws, coords, boundary, neighbors)
    p1 = zeros(T, 3, 3, 3, 3)
    loss_stiff_sw_p1!(p1, sws, coords, boundary, neighbors)
    return p1
end

function loss_stiff_sw_p1!(p1, sws, coords, boundary, neighbors)
    epsilon = zeros(T, 3, 3) # No dims
    born_stiffness = d2U_dϵ2(sws, coords, boundary, neighbors, epsilon) ./ Ω # kJ * mol^-1 * nm^-3
    born_stress = dU_dϵ(sws, coords, boundary, neighbors, epsilon) ./ Ω # kJ * mol^-1 * nm^-3
    born_stress_op = stiff_av_weight .* reshape(born_stress, 3, 3, 1, 1) .* reshape(born_stress, 1, 1, 3, 3) # kJ^2 * mol^-2 * nm^-6
    p1 .= born_stiffness .- Ω .* β .* born_stress_op
    return p1
end

function ChainRulesCore.rrule(::typeof(loss_stiff_sw_p1), sws, coords, boundary, neighbors)
    Y = loss_stiff_sw_p1(sws, coords, boundary, neighbors)

    function loss_stiff_sw_p1_pullback(d_p1)
        p1 = zeros(T, 3, 3, 3, 3)
        d_sws = zero.(sws)
        d_coords = zero(coords)

        autodiff(
            Enzyme.Reverse,
            loss_stiff_sw_p1!,
            Const,
            Duplicated(p1, d_p1),
            Duplicated(sws, d_sws),
            Duplicated(coords, d_coords),
            Const(boundary),
            Const(neighbors),
        )

        return NoTangent(), d_sws, d_coords, NoTangent(), NoTangent()
    end

    return Y, loss_stiff_sw_p1_pullback
end

function loss_stiff_sw_p2(sys)
    return loss_stiff_sw_p2([sys.general_inters.sw], sys.coords, sys.boundary, find_neighbors(sys))
end

function loss_stiff_sw_p2(sws, coords, boundary, neighbors)
    p2 = zeros(T, 3, 3)
    loss_stiff_sw_p2!(p2, sws, coords, boundary, neighbors)
    return p2
end

function loss_stiff_sw_p2!(p2, sws, coords, boundary, neighbors)
    epsilon = zeros(T, 3, 3) # No dims
    p2 .= stiff_av_weight .* dU_dϵ(sws, coords, boundary, neighbors, epsilon) ./ Ω # kJ * mol^-1 * nm^-3
    return p2
end

function ChainRulesCore.rrule(::typeof(loss_stiff_sw_p2), sws, coords, boundary, neighbors)
    Y = loss_stiff_sw_p2(sws, coords, boundary, neighbors)

    function loss_stiff_sw_p2_pullback(d_p2)
        p2 = zeros(T, 3, 3)
        d_sws = zero.(sws)
        d_coords = zero(coords)

        autodiff(
            Enzyme.Reverse,
            loss_stiff_sw_p2!,
            Const,
            Duplicated(p2, d_p2),
            Duplicated(sws, d_sws),
            Duplicated(coords, d_coords),
            Const(boundary),
            Const(neighbors),
        )

        return NoTangent(), d_sws, d_coords, NoTangent(), NoTangent()
    end

    return Y, loss_stiff_sw_p2_pullback
end

const pyscript_stiff = """
import jax
global jax
from functools import partial
global partial

global GNN_energy_global
GNN_energy_global = GNN_energy
global β_global
β_global = β
global volume_global
volume_global = volume

def energy_fn_template(energy_params):
    gnn_energy = partial(GNN_energy_global, energy_params)
    def energy(R, neighbor, **dynamic_kwargs):
        return gnn_energy(R, neighbor=neighbor, **dynamic_kwargs)
    return energy

global energy_fn_template_global
energy_fn_template_global = energy_fn_template

def energy_under_strain(epsilon, energy_fn, frac_coords, neighbor, box_tensor):
    strained_box = jax.numpy.dot(box_tensor, jax.numpy.eye(3) + epsilon)
    energy = energy_fn(frac_coords, neighbor=neighbor, box=strained_box)
    return energy

global energy_under_strain_global
energy_under_strain_global = energy_under_strain

def born_stiffness_fn(frac_coords, neighbor, energy_params, box_tensor, epsilon0):
    energy_fn = energy_fn_template_global(energy_params)
    born_stiffness_contribution = jax.jacfwd(jax.jacrev(energy_under_strain_global))(
                                        epsilon0, energy_fn, frac_coords, neighbor, box_tensor)
    return born_stiffness_contribution / volume_global

global born_stiffness_fn_global
born_stiffness_fn_global = born_stiffness_fn

def born_stress_fn(frac_coords, neighbor, energy_params, box_tensor, epsilon0):
    energy_fn = energy_fn_template_global(energy_params)
    sigma_born = jax.jacrev(energy_under_strain_global)(epsilon0, energy_fn, frac_coords,
                                                        neighbor, box_tensor)
    return sigma_born / volume_global

global born_stress_fn_global
born_stress_fn_global = born_stress_fn

def stiff_p1(frac_coords, neighbor, energy_params, box_tensor):
    epsilon0 = jax.numpy.zeros((3, 3))
    born_stiffness = born_stiffness_fn_global(frac_coords, neighbor, energy_params, box_tensor, epsilon0)
    born_stress = born_stress_fn_global(frac_coords, neighbor, energy_params, box_tensor, epsilon0)
    born_stress_op = $stiff_av_weight * jax.numpy.einsum("ij,kl->ijkl", born_stress, born_stress)
    return born_stiffness - volume_global * β_global * born_stress_op

global stiff_p1_global
stiff_p1_global = stiff_p1

py_stiff_p1_jit = jax.jit(stiff_p1)

def stiff_p1_sum(frac_coords, neighbor, energy_params, box_tensor, d_p1):
    p1 = stiff_p1_global(frac_coords, neighbor, energy_params, box_tensor)
    return (p1 * d_p1).sum()

py_stiff_p1_sum_jit = jax.jit(stiff_p1_sum)

def stiff_p2(frac_coords, neighbor, energy_params, box_tensor):
    epsilon0 = jax.numpy.zeros((3, 3))
    born_stress = born_stress_fn_global(frac_coords, neighbor, energy_params, box_tensor, epsilon0)
    return $stiff_av_weight * born_stress

global stiff_p2_global
stiff_p2_global = stiff_p2

py_stiff_p2_jit = jax.jit(stiff_p2)

def stiff_p2_sum(frac_coords, neighbor, energy_params, box_tensor, d_p2):
    p2 = stiff_p2_global(frac_coords, neighbor, energy_params, box_tensor)
    return (p2 * d_p2).sum()

py_stiff_p2_sum_jit = jax.jit(stiff_p2_sum)
"""

const py_stiff_nt = pyexec(
    @NamedTuple{py_stiff_p1_jit, py_stiff_p1_sum_jit, py_stiff_p2_jit, py_stiff_p2_sum_jit},
    pyscript_stiff,
    Main,
    (GNN_energy=py_GNN_energy, β=β, volume=Ω),
)

const py_stiff_p1_jit = py_stiff_nt.py_stiff_p1_jit
const py_stiff_p1_sum_jit = py_stiff_nt.py_stiff_p1_sum_jit
const py_stiff_p2_jit = py_stiff_nt.py_stiff_p2_jit
const py_stiff_p2_sum_jit = py_stiff_nt.py_stiff_p2_sum_jit

function loss_stiff_gnn_p1(sys)
    frac_coords = sys.coords ./ side_length
    py_neighbors = Zygote.ignore() do
        py_neighbor_fn(svecs_to_jax_array(frac_coords))
    end
    params = sys.general_inters.gnn.params
    return loss_stiff_gnn_p1(frac_coords, py_neighbors, params)
end

function loss_stiff_gnn_p1(frac_coords, py_neighbors, params)
    py_p1 = py_stiff_p1_jit(svecs_to_jax_array(frac_coords), py_neighbors, params.py_params,
                            py_box_tensor)
    return pyconvert(Array{T}, py_p1)
end

function ChainRulesCore.rrule(::typeof(loss_stiff_gnn_p1), frac_coords, py_neighbors, params)
    Y = loss_stiff_gnn_p1(frac_coords, py_neighbors, params)

    function loss_stiff_gnn_p1_pullback(d_p1)
        py_d_p1 = pymod_jax.numpy.array(d_p1)
        py_dl_dfc, py_dl_dp = pymod_jax.grad(py_stiff_p1_sum_jit, argnums=[0, 2])(
            svecs_to_jax_array(frac_coords),
            py_neighbors,
            params.py_params,
            py_box_tensor,
            py_d_p1,
        )
        dl_dfc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dfc)))
        dl_dp = GNNParams(py_dl_dp)
        return NoTangent(), dl_dfc, NoTangent(), dl_dp
    end

    return Y, loss_stiff_gnn_p1_pullback
end

function loss_stiff_gnn_p2(sys)
    frac_coords = sys.coords ./ side_length
    py_neighbors = Zygote.ignore() do
        py_neighbor_fn(svecs_to_jax_array(frac_coords))
    end
    params = sys.general_inters.gnn.params
    return loss_stiff_gnn_p2(frac_coords, py_neighbors, params)
end

function loss_stiff_gnn_p2(frac_coords, py_neighbors, params)
    py_p2 = py_stiff_p2_jit(svecs_to_jax_array(frac_coords), py_neighbors, params.py_params,
                            py_box_tensor)
    return pyconvert(Matrix{T}, py_p2)
end

function ChainRulesCore.rrule(::typeof(loss_stiff_gnn_p2), frac_coords, py_neighbors, params)
    Y = loss_stiff_gnn_p2(frac_coords, py_neighbors, params)

    function loss_stiff_gnn_p2_pullback(d_p2)
        py_d_p2 = pymod_jax.numpy.array(d_p2)
        py_dl_dfc, py_dl_dp = pymod_jax.grad(py_stiff_p2_sum_jit, argnums=[0, 2])(
            svecs_to_jax_array(frac_coords),
            py_neighbors,
            params.py_params,
            py_box_tensor,
            py_d_p2,
        )
        dl_dfc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dfc)))
        dl_dp = GNNParams(py_dl_dp)
        return NoTangent(), dl_dfc, NoTangent(), dl_dp
    end

    return Y, loss_stiff_gnn_p2_pullback
end

function stiff_delta_term()
    delta_term = zeros(T, 3, 3, 3, 3) # No dims
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        δik, δjl, δil, δjk = T(i == k), T(j == l), T(i == l), T(j == k)
        delta_term[i, j, k, l] = δik * δjl + δil * δjk
    end
    return delta_term
end

ChainRulesCore.@non_differentiable stiff_delta_term()

function stiff_c_terms(C)
    c11 = (C[1, 1, 1, 1] + C[2, 2, 2, 2] + C[3, 3, 3, 3]) / 3
    c12 = (C[1, 1, 2, 2] + C[2, 2, 1, 1] + C[1, 1, 3, 3] + C[3, 3, 1, 1] +
           C[2, 2, 3, 3] + C[3, 3, 2, 2]) / 6
    c44 = (C[1, 2, 1, 2] + C[2, 1, 1, 2] + C[1, 2, 2, 1] + C[2, 1, 2, 1] +
           C[1, 3, 1, 3] + C[3, 1, 1, 3] + C[1, 3, 3, 1] + C[3, 1, 3, 1] +
           C[3, 2, 3, 2] + C[2, 3, 3, 2] + C[3, 2, 2, 3] + C[2, 3, 2, 3]) / 12
    return c11, c12, c44
end

function loss_stiff_final(p1, p2, n_blocks)
    p2_div_blocks = p2 ./ n_blocks
    born_stress_term = reshape(p2_div_blocks, 3, 3, 1, 1) .* reshape(p2_div_blocks, 1, 1, 3, 3) # kJ^2 * mol^-2 * nm^-6
    delta_term = stiff_delta_term()
    C = (p1 ./ n_blocks) .+ Ω .* β .* born_stress_term .+ delta_term .* n_atoms ./ (Ω .* β) # kJ * mol^-1 * nm^-3
    c11, c12, c44 = stiff_c_terms(C)
    loss_val = (γc / 3) * sum(abs2, [c11, c12, c44] .- c_target) # No dims
    return loss_val, c11, c12, c44
end

function loss_stiff_rev(p1, p2, p2_accum, n_blocks)
    p2_accum_div_blocks = p2_accum ./ n_blocks
    delta_term = stiff_delta_term()
    C = p1 .+ Ω .* β .* born_stress_term_rev(p2_accum_div_blocks, p2) .+
        delta_term .* n_atoms ./ (Ω .* β) # kJ * mol^-1 * nm^-3
    c11, c12, c44 = stiff_c_terms(C)
    return (γc / 3) * sum(abs2, [c11, c12, c44] .- c_target) # No dims
end

function born_stress_term_rev(p2_accum_div_blocks, p2)
    return reshape(p2_accum_div_blocks, 3, 3, 1, 1) .* reshape(p2_accum_div_blocks, 1, 1, 3, 3)
end

function ChainRulesCore.rrule(::typeof(born_stress_term_rev), p2_accum_div_blocks, p2)
    Y = born_stress_term_rev(p2_accum_div_blocks, p2)

    function born_stress_term_rev_pullback(dy)
        grads = Zygote.gradient(p2_accum_div_blocks, p2) do p2_accum_div_blocks, p2
            b1 = reshape(p2_accum_div_blocks, 3, 3, 1, 1) .* reshape(p2, 1, 1, 3, 3) # kJ^2 * mol^-2 * nm^-6
            b2 = reshape(p2, 3, 3, 1, 1) .* reshape(p2_accum_div_blocks, 1, 1, 3, 3) # kJ^2 * mol^-2 * nm^-6
            return sum((b1 .+ b2) .* dy)
        end
        return NoTangent(), NoTangent(), grads[2]
    end

    return Y, born_stress_term_rev_pullback
end

function stiff_cross_terms(sw_p2, gnn_p2)
    ct1 = reshape(sw_p2 , 3, 3, 1, 1) .* reshape(gnn_p2, 1, 1, 3, 3)
    ct2 = reshape(gnn_p2, 3, 3, 1, 1) .* reshape(sw_p2 , 1, 1, 3, 3)
    return -Ω .* β .* (ct1 .+ ct2)
end

function loss_fn(sys, loss_type, inter_type)
    if loss_type == :stress
        if inter_type == :sw
            return (stress=loss_stress_sw(sys),)
        elseif inter_type == :gnn
            return (stress=loss_stress_gnn(sys),)
        elseif inter_type == :sw_gnn
            return (stress=(loss_stress_sw(sys, false) + loss_stress_gnn(sys)),)
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    elseif loss_type == :stiff
        if inter_type == :sw
            return (stiff_p1=loss_stiff_sw_p1(sys), stiff_p2=loss_stiff_sw_p2(sys))
        elseif inter_type == :gnn
            return (stiff_p1=loss_stiff_gnn_p1(sys), stiff_p2=loss_stiff_gnn_p2(sys))
        elseif inter_type == :sw_gnn
            sw_p2, gnn_p2 = loss_stiff_sw_p2(sys), loss_stiff_gnn_p2(sys)
            return (stiff_p1=(loss_stiff_sw_p1(sys) .+ loss_stiff_gnn_p1(sys) .+
                              stiff_cross_terms(sw_p2, gnn_p2)),
                    stiff_p2=(sw_p2 .+ gnn_p2))
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    elseif loss_type == :stress_stiff
        if inter_type == :sw
            return (stress=loss_stress_sw(sys), stiff_p1=loss_stiff_sw_p1(sys),
                                                stiff_p2=loss_stiff_sw_p2(sys))
        elseif inter_type == :gnn
            return (stress=loss_stress_gnn(sys), stiff_p1=loss_stiff_gnn_p1(sys),
                                                stiff_p2=loss_stiff_gnn_p2(sys))
        elseif inter_type == :sw_gnn
            sw_p2, gnn_p2 = loss_stiff_sw_p2(sys), loss_stiff_gnn_p2(sys)
            return (stress=(loss_stress_sw(sys, false) + loss_stress_gnn(sys)),
                    stiff_p1=(loss_stiff_sw_p1(sys) .+ loss_stiff_gnn_p1(sys) .+
                              stiff_cross_terms(sw_p2, gnn_p2)),
                    stiff_p2=(sw_p2 .+ gnn_p2))
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function loss_fn_rev(sys, loss_accum_fwd, n_blocks, loss_type, inter_type)
    if loss_type == :stress
        if inter_type == :sw
            return loss_stress_sw(sys)
        elseif inter_type == :gnn
            return loss_stress_gnn(sys)
        elseif inter_type == :sw_gnn
            return loss_stress_sw(sys, false) + loss_stress_gnn(sys)
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    elseif loss_type == :stiff
        if inter_type == :sw
            return loss_stiff_rev(loss_stiff_sw_p1(sys), loss_stiff_sw_p2(sys),
                                  loss_accum_fwd.stiff_p2, n_blocks)
        elseif inter_type == :gnn
            return loss_stiff_rev(loss_stiff_gnn_p1(sys), loss_stiff_gnn_p2(sys),
                                  loss_accum_fwd.stiff_p2, n_blocks)
        elseif inter_type == :sw_gnn
            sw_p2, gnn_p2 = loss_stiff_sw_p2(sys), loss_stiff_gnn_p2(sys)
            return loss_stiff_rev(loss_stiff_sw_p1(sys) .+ loss_stiff_gnn_p1(sys) .+
                                  stiff_cross_terms(sw_p2, gnn_p2),
                                  sw_p2 .+ gnn_p2, loss_accum_fwd.stiff_p2, n_blocks)
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    elseif loss_type == :stress_stiff
        if inter_type == :sw
            return loss_stress_sw(sys) +
                   loss_stiff_rev(loss_stiff_sw_p1(sys), loss_stiff_sw_p2(sys),
                                  loss_accum_fwd.stiff_p2, n_blocks)
        elseif inter_type == :gnn
            return loss_stress_gnn(sys) +
                   loss_stiff_rev(loss_stiff_gnn_p1(sys), loss_stiff_gnn_p2(sys),
                                  loss_accum_fwd.stiff_p2, n_blocks)
        elseif inter_type == :sw_gnn
            sw_p2, gnn_p2 = loss_stiff_sw_p2(sys), loss_stiff_gnn_p2(sys)
            return loss_stress_sw(sys, false) + loss_stress_gnn(sys) +
                   loss_stiff_rev(loss_stiff_sw_p1(sys) .+ loss_stiff_gnn_p1(sys) .+
                                  stiff_cross_terms(sw_p2, gnn_p2),
                                  sw_p2 .+ gnn_p2, loss_accum_fwd.stiff_p2, n_blocks)
        else
            throw(ArgumentError("inter_type not recognised"))
        end
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function loss_fn_init(loss_type)
    if loss_type == :stress
        return (stress=zero(T),)
    elseif loss_type == :stiff
        return (stiff_p1=zeros(T, 3, 3, 3, 3), stiff_p2=zeros(T, 3, 3))
    elseif loss_type == :stress_stiff
        return (stress=zero(T), stiff_p1=zeros(T, 3, 3, 3, 3), stiff_p2=zeros(T, 3, 3))
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function loss_fn_accum(l1, l2, loss_type)
    if loss_type == :stress
        return (stress=(l1.stress + l2.stress),)
    elseif loss_type == :stiff
        return (stiff_p1=(l1.stiff_p1 .+ l2.stiff_p1), stiff_p2=(l1.stiff_p2 .+ l2.stiff_p2))
    elseif loss_type == :stress_stiff
        return (stress=(l1.stress + l2.stress), stiff_p1=(l1.stiff_p1 .+ l2.stiff_p1),
                                                stiff_p2=(l1.stiff_p2 .+ l2.stiff_p2))
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function loss_fn_final(loss_accum, n_blocks, loss_type)
    loss_val_stress, loss_val_stiff, c11, c12, c44 = zero(T), zero(T), zero(T), zero(T), zero(T)
    if loss_type == :stress
        loss_val_stress = loss_accum.stress / n_blocks
    elseif loss_type == :stiff
        loss_val_stiff, c11, c12, c44 = loss_stiff_final(loss_accum.stiff_p1,
                                                    loss_accum.stiff_p2, n_blocks)
    elseif loss_type == :stress_stiff
        loss_val_stress = loss_accum.stress / n_blocks
        loss_val_stiff, c11, c12, c44 = loss_stiff_final(loss_accum.stiff_p1,
                                                    loss_accum.stiff_p2, n_blocks)
    else
        throw(ArgumentError("loss_type not recognised"))
    end
    return loss_val_stress + loss_val_stiff, (stress=loss_val_stress, stiff=loss_val_stiff,
                                                c11=c11, c12=c12, c44=c44)
end

function loss_and_grads(sys, loss_accum_fwd, n_blocks, loss_type, inter_type)
    l, grads = Zygote.withgradient(loss_fn_rev, sys, loss_accum_fwd, n_blocks,
                                   loss_type, inter_type)
    sys_grads = grads[1]
    dl_dc = isnothing(sys_grads.coords)     ? zeros(SVector{3, T}, n_atoms) : sys_grads.coords
    dl_dv = isnothing(sys_grads.velocities) ? zeros(SVector{3, T}, n_atoms) : sys_grads.velocities
    dl_dp_sw, dl_dp_gnn = extract_grads(sys_grads.general_inters, inter_type)
    return l, dl_dc, dl_dv, dl_dp_sw, dl_dp_gnn
end

function params_to_inters(params_sw, params_gnn, inter_type)
    if inter_type == :sw
        sw = StillingerWeber(params_sw[1], params_sw[2], 4, 0, params_sw[3], params_sw[4],
                             params_sw[5], params_sw[6], params_sw[7])
        return (sw=sw,)
    elseif inter_type == :gnn
        gnn = DimeNetPP(params_gnn)
        return (gnn=gnn,)
    elseif inter_type == :sw_gnn
        sw = StillingerWeber(params_sw[1], params_sw[2], 4, 0, params_sw[3], params_sw[4],
                             params_sw[5], params_sw[6], params_sw[7])
        gnn = DimeNetPP(params_gnn)
        return (sw=sw, gnn=gnn)
    else
        throw(ArgumentError("inter_type not recognised"))
    end
end

function extract_grads(general_inters, inter_type)
    if inter_type == :sw
        sw = general_inters.sw
        return [sw.A, sw.B, sw.a, sw.λ, sw.γ, sw.σ, sw.ϵ], GNNParams(nothing)
    elseif inter_type == :gnn
        return zeros(T, n_params_sw), general_inters.gnn.params
    elseif inter_type == :sw_gnn
        sw = general_inters.sw
        return [sw.A, sw.B, sw.a, sw.λ, sw.γ, sw.σ, sw.ϵ], general_inters.gnn.params
    else
        throw(ArgumentError("inter_type not recognised"))
    end
end

function generate_noise!(noise, seed, k=T(ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)))
    rng = Xoshiro(seed)
    @inbounds for i in eachindex(noise)
        noise[i] = Molly.random_velocity_3D(atom_mass, temp, k, rng)
    end
    return noise
end

rand_seed() = rand(0:typemax(Int))

function forward_sim(coords_in, velocities_in, noise_seeds, γ, n_steps, n_steps_loss,
                     n_steps_buffer, params_sw, params_gnn,
                     dt, n_threads, inter_type, loss_type, run_grads=true, calc_loss=true)
    if isnothing(coords_in)
        sc = SimpleCrystals.Diamond(Float64(side_length) * u"nm" / 5, :C, SVector(5, 5, 5))
        sys_init = System(sc; force_units=u"kJ * nm^-1", energy_units=u"kJ")
        coords_start = SVector{3, T}.(ustrip.(sys_init.coords))
    else
        coords_start = copy(coords_in)
    end

    atoms = fill(Atom(mass=T(atom_mass), σ=zero(T), ϵ=zero(T)), n_atoms)
    general_inters = params_to_inters(params_sw, params_gnn, inter_type)

    if isnothing(velocities_in)
        velocities_start = [random_velocity(atom_mass, temp) for _ in 1:n_atoms]
    else
        velocities_start = copy(velocities_in)
    end

    sys = System(
        atoms=atoms,
        coords=copy(coords_start),
        boundary=boundary,
        velocities=copy(velocities_start),
        general_inters=general_inters,
        neighbor_finder=neighbor_finder_fwd,
        loggers=(
            # These are run manually when calculating loss
            coords=CoordCopyLogger(1),
            velocities=VelocityCopyLogger(1),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    n_blocks = 0
    vel_scale = exp(-γ * dt)
    noise_scale = sqrt(1 - vel_scale^2)

    fs = zero(sys.coords)
    fs_chunks = [zero(sys.coords) for _ in 1:n_threads]
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)

    loss_accum = loss_fn_init(loss_type)
    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "sw force" begin
            if inter_type in (:sw, :sw_gnn)
                @inbounds for ti in 1:n_threads
                    fs_chunks[ti] .= (zero(SVector{3, T}),)
                end
                forces_sw!(fs_chunks, sys.general_inters.sw, sys.coords, sys.boundary, neighbors, n_threads)
                fs .= sum(fs_chunks)
            else
                fs .= (zero(SVector{3, T}),)
            end
        end
        @timeit to "gnn force" begin
            if inter_type in (:gnn, :sw_gnn)
                fs .+= AtomsCalculators.forces(sys, sys.general_inters.gnn)
            end
        end
        @timeit to "f.1" sys.velocities .+= fs .* dt ./ masses(sys)
        @timeit to "f.2" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.3" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.4" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.5" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.6" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "neighbour finding" begin
            if inter_type in (:sw, :sw_gnn)
                neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
            end
        end
        @timeit to "f.8" if calc_loss && ((n_steps - step_n) % n_steps_loss == 0) &&
                            step_n >= n_steps_buffer
            n_blocks += 1
            loss_accum = loss_fn_accum(loss_accum, loss_fn(sys, loss_type, inter_type), loss_type)
            run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
        end
    end
    if calc_loss
        loss, loss_logs = loss_fn_final(loss_accum, n_blocks, loss_type)
    else
        loss, loss_logs = zero(T), zero(T)
    end

    if run_grads
        noises = [zeros(SVector{3, T}, n_atoms) for i in 1:n_steps]
        for step_n in 1:n_steps
            @inbounds generate_noise!(noises[step_n], noise_seeds[step_n])
        end

        function loss_forward(params_sw, params_gnn)
            general_inters = params_to_inters(params_sw, params_gnn, inter_type)
            sys_grad = System(
                atoms=atoms,
                coords=copy(coords_start),
                boundary=boundary,
                velocities=copy(velocities_start),
                general_inters=general_inters,
                neighbor_finder=neighbor_finder_fwd,
                force_units=NoUnits,
                energy_units=NoUnits,
            )
            n_blocks_grad = 0
            loss_accum_rad = loss_fn_init(loss_type)
            neighbors_grad = find_neighbors(sys_grad, sys_grad.neighbor_finder; n_threads=n_threads)
            for step_n in 1:n_steps
                # n_threads=1 to avoid possible Enzyme issue
                accels = accelerations(sys_grad, neighbors_grad; n_threads=1)
                sys_grad.velocities += accels .* dt
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.velocities = sys_grad.velocities .* vel_scale .+ noises[step_n] .* noise_scale
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.coords = wrap_coords.(sys_grad.coords, (sys_grad.boundary,))
                if inter_type in (:sw, :sw_gnn)
                    neighbors_grad = find_neighbors(sys_grad, sys_grad.neighbor_finder, neighbors_grad,
                                                    step_n; n_threads=n_threads)
                end
                if (n_steps - step_n) % n_steps_loss == 0 && step_n >= n_steps_buffer
                    n_blocks_grad += 1
                    loss_accum_rad = loss_fn_accum(loss_accum_rad,
                                            loss_fn(sys_grad, loss_type, inter_type), loss_type)
                end
            end
            loss_rad, _ = loss_fn_final(loss_accum_rad, n_blocks_grad, loss_type)
            return loss_rad
        end

        @timeit to "forward grad" begin
            loss_rad, grads_args = Zygote.withgradient(loss_forward, params_sw, params_gnn)
            grads_sw = isnothing(grads_args[1]) ? zeros(T, n_params_sw) : grads_args[1]
            grads_gnn = isnothing(grads_args[2]) ? GNNParams(nothing) : grads_args[2]
        end
        @assert isapprox(loss_rad, loss; rtol=1e-3)
        println("fwd loss ", loss_rad)
        println("fwd gradients ", grads_sw, ", ", grads_gnn)
    else
        grads_sw, grads_gnn = zeros(T, n_params_sw), GNNParams(nothing)
    end

    loss_accum_blocks = merge(loss_accum, (n_blocks=n_blocks,))
    return sys, loss, loss_accum_blocks, loss_logs, grads_sw, grads_gnn
end

function reweighting_sim(coords_in, velocities_in, noise_seeds, γ, n_steps, n_steps_loss,
                         n_steps_buffer, params_sw, params_gnn, dt, n_threads, inter_type,
                         loss_type)
    coords_start = copy(coords_in)
    atoms = fill(Atom(mass=T(atom_mass), σ=zero(T), ϵ=zero(T)), n_atoms)
    general_inters = params_to_inters(params_sw, params_gnn, inter_type)
    velocities_start = copy(velocities_in)

    sys = System(
        atoms=atoms,
        coords=copy(coords_start),
        boundary=boundary,
        velocities=copy(velocities_start),
        general_inters=general_inters,
        neighbor_finder=neighbor_finder_fwd,
        loggers=(
            # These are run manually when calculating loss
            coords=CoordCopyLogger(1),
            velocities=VelocityCopyLogger(1),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    n_blocks = 0
    vel_scale = exp(-γ * dt)
    noise_scale = sqrt(1 - vel_scale^2)

    fs = zero(sys.coords)
    fs_chunks = [zero(sys.coords) for _ in 1:n_threads]
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)

    loss_accum = loss_fn_init(loss_type)
    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "sw force" begin
            if inter_type in (:sw, :sw_gnn)
                @inbounds for ti in 1:n_threads
                    fs_chunks[ti] .= (zero(SVector{3, T}),)
                end
                forces_sw!(fs_chunks, sys.general_inters.sw, sys.coords, sys.boundary, neighbors, n_threads)
                fs .= sum(fs_chunks)
            else
                fs .= (zero(SVector{3, T}),)
            end
        end
        @timeit to "gnn force" begin
            if inter_type in (:gnn, :sw_gnn)
                fs .+= AtomsCalculators.forces(sys, sys.general_inters.gnn)
            end
        end
        @timeit to "f.1" sys.velocities .+= fs .* dt ./ masses(sys)
        @timeit to "f.2" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.3" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.4" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.5" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.6" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "neighbour finding" begin
            if inter_type in (:sw, :sw_gnn)
                neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
            end
        end
        @timeit to "f.8" if (n_steps - step_n) % n_steps_loss == 0 && step_n >= n_steps_buffer
            n_blocks += 1
            loss_accum = loss_fn_accum(loss_accum, loss_fn(sys, loss_type, inter_type), loss_type)
            run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
        end
    end
    loss_fwd, loss_logs = loss_fn_final(loss_accum, n_blocks, loss_type)

    loss_rew_accum    = zero(T)
    dl_dp_accum_sw    = zeros(T, n_params_sw)
    dl_dp_accum_gnn   = GNNParams(nothing)
    dE_dp_accum_sw    = zeros(T, n_params_sw)
    dE_dp_accum_gnn   = GNNParams(nothing)
    l_dE_dp_accum_sw  = zeros(T, n_params_sw)
    l_dE_dp_accum_gnn = GNNParams(nothing)

    @timeit to "reweighting" for snapshot_i in eachindex(values(sys.loggers.coords))
        sys.coords     .= values(sys.loggers.coords    )[snapshot_i]
        sys.velocities .= values(sys.loggers.velocities)[snapshot_i]
        neighbors = find_neighbors(sys; n_threads=n_threads)

        loss_block, _, _, dl_dp_accum_sw_block, dl_dp_accum_gnn_block = loss_and_grads(
                        sys, loss_accum, n_blocks, loss_type, inter_type)
        loss_rew_accum += loss_block
        dl_dp_accum_sw .= dl_dp_accum_sw .+ dl_dp_accum_sw_block
        dl_dp_accum_gnn = dl_dp_accum_gnn + dl_dp_accum_gnn_block

        sys_E_grads = Zygote.gradient(potential_energy, sys, neighbors)[1]
        dE_dp_block_sw, dE_dp_block_gnn = extract_grads(sys_E_grads.general_inters, inter_type)
        dE_dp_accum_sw .= dE_dp_accum_sw .+ dE_dp_block_sw
        dE_dp_accum_gnn = dE_dp_accum_gnn + dE_dp_block_gnn
        l_dE_dp_accum_sw .= l_dE_dp_accum_sw .+ dE_dp_block_sw .* loss_block
        l_dE_dp_accum_gnn = l_dE_dp_accum_gnn + (dE_dp_block_gnn * loss_block)
    end

    loss_rew = loss_rew_accum / n_blocks
    grads_sw = dl_dp_accum_sw ./ n_blocks .- β .* (l_dE_dp_accum_sw ./ n_blocks .- loss_rew .* dE_dp_accum_sw ./ n_blocks)
    grads_gnn = dl_dp_accum_gnn * inv(n_blocks) + (l_dE_dp_accum_gnn * inv(n_blocks) + dE_dp_accum_gnn * (-loss_rew / n_blocks)) * -β

    return sys, loss_rew, loss_fwd, loss_logs, grads_sw, grads_gnn
end

function sum_dldfiacc3_dot_f(dl_dfi_accum_3, atom_masses_3N, sw, coords, boundary, neighbors,
                             n_threads=Threads.nthreads())
    fs = forces_sw(sw, coords, boundary, neighbors, n_threads)
    fs_flat = reinterpret(T, fs)
    return dot(dl_dfi_accum_3, fs_flat ./ atom_masses_3N)
end

function sum_dldfiacc1_dot_f(dl_dfi_accum_1, atom_masses_3N, dt2, sw, coords, boundary, neighbors,
                             n_threads=Threads.nthreads())
    fs = forces_sw(sw, coords, boundary, neighbors, n_threads)
    fs_flat = reinterpret(T, fs)
    return dot(dl_dfi_accum_1, fs_flat .* (dt2 ./ (2 .* atom_masses_3N)))
end

function sw_force_and_grads!(fs, dl_dfi, dl_dp_i_sw, coords, boundary, sw, neighbors,
                               dl_dfi_accum_1, dl_dfi_accum_3, atom_masses_3N, dt2, n_threads)
    @timeit to "f.a" d_coords = zero(coords)

    @timeit to "f.b" autodiff(
        Enzyme.Reverse,
        sum_dldfiacc3_dot_f,
        Active,
        Const(dl_dfi_accum_3),
        Const(atom_masses_3N),
        Const(sw),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(neighbors),
        Const(n_threads),
    )[1]

    dl_dfi .= reinterpret(T, d_coords)

    @timeit to "f.c" grads = autodiff(
        Enzyme.Reverse,
        sum_dldfiacc1_dot_f,
        Active,
        Const(dl_dfi_accum_1),
        Const(atom_masses_3N),
        Const(dt2),
        Active(sw),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(neighbors),
        Const(n_threads),
    )[1]

    swg = grads[4]
    dl_dp_i_sw .= [swg.A, swg.B, swg.a, swg.λ, swg.γ, swg.σ, swg.ϵ]

    @timeit to "f.d" fs .= forces_sw(sw, coords, boundary, neighbors, n_threads)

    return fs, dl_dfi, dl_dp_i_sw
end

const pyscript_sum_dldf = """
import jax
global jax

global GNN_energy_global
GNN_energy_global = GNN_energy

def sum_dldfiacc3_dot_f(params, frac_coords, neighbors, box_tensor, dl_dfi_accum_3, atom_masses_3N):
    E_grad = jax.grad(GNN_energy_global, argnums=[1])(params, frac_coords, neighbors, box=box_tensor)
    fs_flat = -E_grad[0].flatten()
    return jax.numpy.dot(dl_dfi_accum_3, fs_flat / atom_masses_3N)

py_sum_dldfiacc3_dot_f_jit = jax.jit(sum_dldfiacc3_dot_f)

def sum_dldfiacc1_dot_f(params, frac_coords, neighbors, box_tensor, dl_dfi_accum_1, atom_masses_3N, dt2):
    E_grad = jax.grad(GNN_energy_global, argnums=[1])(params, frac_coords, neighbors, box=box_tensor)
    fs_flat = -E_grad[0].flatten()
    return jax.numpy.dot(dl_dfi_accum_1, (fs_flat * dt2) / (2 * atom_masses_3N))

py_sum_dldfiacc1_dot_f_jit = jax.jit(sum_dldfiacc1_dot_f)
"""

const py_sum_dldf_nt = pyexec(
    @NamedTuple{py_sum_dldfiacc3_dot_f_jit, py_sum_dldfiacc1_dot_f_jit},
    pyscript_sum_dldf,
    Main,
    (GNN_energy=py_GNN_energy,),
)

const py_sum_dldfiacc3_dot_f_jit = py_sum_dldf_nt.py_sum_dldfiacc3_dot_f_jit
const py_sum_dldfiacc1_dot_f_jit = py_sum_dldf_nt.py_sum_dldfiacc1_dot_f_jit

function gnn_rev_grads(coords, gnn, py_neighbors, dl_dfi_accum_1, dl_dfi_accum_3,
                       atom_masses_3N, dt2)
    @timeit to "g.c" begin
        py_frac_coords    = svecs_to_jax_array(coords ./ side_length)
        py_dl_dfi_accum_1 = pymod_jax.numpy.array(dl_dfi_accum_1)
        py_dl_dfi_accum_3 = pymod_jax.numpy.array(dl_dfi_accum_3)
        py_atom_masses_3N = pymod_jax.numpy.array(atom_masses_3N)
    end

    @timeit to "g.d" begin
        py_dl_dfc = pymod_jax.grad(py_sum_dldfiacc3_dot_f_jit, argnums=[1])(
                            gnn.params.py_params, py_frac_coords, py_neighbors, py_box_tensor,
                            py_dl_dfi_accum_3, py_atom_masses_3N)[0].flatten()
        dl_dfi_gnn = pyconvert(Vector{T}, py_dl_dfc)
    end

    @timeit to "g.e" begin
        py_dl_dp_i_gnn = pymod_jax.grad(py_sum_dldfiacc1_dot_f_jit, argnums=[0])(
                            gnn.params.py_params, py_frac_coords, py_neighbors, py_box_tensor,
                            py_dl_dfi_accum_1, py_atom_masses_3N, dt2)[0]
        dl_dp_i_gnn = GNNParams(py_dl_dp_i_gnn)
    end

    return dl_dfi_gnn, dl_dp_i_gnn
end

function reverse_sim(coords, velocities, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc,
                     n_steps_buffer, params_sw, params_gnn, dt, n_threads, fwd_loggers,
                     loss_accum_fwd, inter_type, loss_type)
    atoms = fill(Atom(mass=T(atom_mass), σ=zero(T), ϵ=zero(T)), n_atoms)
    general_inters = params_to_inters(params_sw, params_gnn, inter_type)

    sys = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        velocities=copy(velocities),
        general_inters=general_inters,
        neighbor_finder=neighbor_finder_rev,
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    exp_mγdt = exp(-γ * dt)
    noise_scale = sqrt(1 - exp_mγdt^2)
    n_blocks = loss_accum_fwd.n_blocks
    fwd_coords_logger = values(fwd_loggers.coords)
    fwd_velocities_logger = values(fwd_loggers.velocities)

    loss_accum, dl_dxn_raw, dl_dvn_raw, dl_dp_accum_sw, dl_dp_accum_gnn = loss_and_grads(
                                    sys, loss_accum_fwd, n_blocks, loss_type, inter_type)
    dl_dxn = Vector(reinterpret(T, SVector{3, T}.(dl_dxn_raw)))
    dl_dvn = Vector(reinterpret(T, SVector{3, T}.(dl_dvn_raw)))
    atom_masses_3N = repeat(masses(sys); inner=3)
    dt2 = dt^2
    dl_dp_i_sw = zeros(T, n_params_sw)
    dl_dp_i_gnn = GNNParams(nothing)
    dl_dfi = zeros(T, n_atoms_x3)
    dl_dfi_accum_1 = dl_dxn .* (1 .+ exp_mγdt) .+ dl_dvn .* exp_mγdt .* 2 ./ dt
    dl_dfi_accum_2 = dl_dxn .* (1 .+ exp_mγdt) .+ dl_dvn .* (exp_mγdt .- 1) .* 2 ./ dt
    dl_dfi_accum_3 = dl_dvn .* (1 .+ exp_mγdt) ./ dt
    dl_dfi_accum_4 = dl_dxn .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt) .+
                     dl_dvn .* (exp_mγdt .- inv(exp_mγdt)) ./ dt

    fs = zero(sys.coords)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)

    @timeit to "rev loop" for loss_count in 1:n_blocks
        step_start = n_steps-(loss_count-1)*n_steps_loss
        step_end = max(step_start-n_steps_trunc+1, 1)

        for step_n in step_start:-1:step_end
            @timeit to "loop part 1" begin
                @timeit to "1.a" sys.coords .= sys.coords .- sys.velocities .* dt ./ 2
                @timeit to "1.b" @inbounds generate_noise!(noise, noise_seeds[step_n])
                @timeit to "1.c" sys.velocities .= (sys.velocities .- noise .* noise_scale) ./ exp_mγdt
                @timeit to "1.d" sys.coords .= sys.coords .- sys.velocities .* dt ./ 2
                @timeit to "1.e" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
                @timeit to "1.f" dl_dfi_accum_4 .= exp_mγdt .* dl_dfi_accum_4 .+ dt2 .* dl_dfi .* (1 .+ exp_mγdt) ./ 2
                @timeit to "1.g" dl_dfi_accum_3 .= dl_dfi_accum_3 .+ dl_dfi_accum_4
            end
            @timeit to "neighbour finding" begin
                if inter_type in (:sw, :sw_gnn)
                    neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                               n_threads=n_threads)
                end
            end
            @timeit to "sw force and grads" begin
                if inter_type in (:sw, :sw_gnn)
                    sw_force_and_grads!(fs, dl_dfi, dl_dp_i_sw, sys.coords,
                            sys.boundary, sys.general_inters.sw, neighbors, dl_dfi_accum_1,
                            dl_dfi_accum_3, atom_masses_3N, dt2, n_threads)
                else
                    fs .= (zero(SVector{3, T}),)
                    dl_dfi .= zero(T)
                end
            end
            @timeit to "gnn force and grads" begin
                if inter_type in (:gnn, :sw_gnn)
                    @timeit to "g.a" py_neighbors = py_neighbor_fn(svecs_to_jax_array(
                                                                sys.coords ./ side_length))
                    @timeit to "g.b" fs .+= AtomsCalculators.forces(sys, sys.general_inters.gnn;
                                                                    py_neighbors=py_neighbors)
                    dl_dfi_gnn, dl_dp_i_gnn = gnn_rev_grads(sys.coords, sys.general_inters.gnn,
                                py_neighbors, dl_dfi_accum_1, dl_dfi_accum_3, atom_masses_3N, dt2)
                    dl_dfi .+= dl_dfi_gnn
                end
            end
            @timeit to "loop part 2" begin
                @timeit to "2.a" sys.velocities .= sys.velocities .- fs .* dt ./ masses(sys)
                @timeit to "2.b" dl_dp_accum_sw .= dl_dp_accum_sw .+ dl_dp_i_sw
                @timeit to "2.c" dl_dp_accum_gnn = dl_dp_accum_gnn + dl_dp_i_gnn
                @timeit to "2.d" dl_dfi_accum_2 .= exp_mγdt .* dl_dfi_accum_2 .+ dt2 .* dl_dfi
                @timeit to "2.e" dl_dfi_accum_1 .= dl_dfi_accum_1 .+ dl_dfi_accum_2
            end
        end

        @timeit to "loop part 3" begin
            if loss_count != n_blocks
                sys.coords .= fwd_coords_logger[end-loss_count]
                sys.velocities .= fwd_velocities_logger[end-loss_count]
                @timeit to "3.a" loss_block, dl_dxn_raw_block, dl_dvn_raw_block, dl_dp_accum_sw_block,
                                dl_dp_accum_gnn_block = loss_and_grads(sys,
                                                loss_accum_fwd, n_blocks, loss_type, inter_type)
                loss_accum += loss_block
                dl_dxn .= Vector(reinterpret(T, SVector{3, T}.(dl_dxn_raw_block)))
                dl_dvn .= Vector(reinterpret(T, SVector{3, T}.(dl_dvn_raw_block)))
                dl_dp_accum_sw .= dl_dp_accum_sw .+ dl_dp_accum_sw_block
                dl_dp_accum_gnn = dl_dp_accum_gnn + dl_dp_accum_gnn_block
                dl_dfi .= zeros(T, n_atoms_x3)
                dl_dfi_accum_1 .= dl_dxn .* (1 .+ exp_mγdt) .+ dl_dvn .* exp_mγdt .* 2 ./ dt
                dl_dfi_accum_2 .= dl_dxn .* (1 .+ exp_mγdt) .+ dl_dvn .* (exp_mγdt .- 1) .* 2 ./ dt
                dl_dfi_accum_3 .= dl_dvn .* (1 .+ exp_mγdt) ./ dt
                dl_dfi_accum_4 .= dl_dxn .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt) .+
                                  dl_dvn .* (exp_mγdt .- inv(exp_mγdt)) ./ dt
            end
        end
    end
    loss = loss_accum / n_blocks
    dl_dp_sw = dl_dp_accum_sw ./ n_blocks
    dl_dp_gnn = dl_dp_accum_gnn * inv(T(n_blocks))
    println("rev n_threads ", n_threads)
    println("rev loss ", loss)
    println("rev gradient ", dl_dp_sw, ", ", dl_dp_gnn)

    return sys, loss, dl_dp_sw, dl_dp_gnn
end

function grad_diff_gnn(g1::GNNParams{Py}, g2::GNNParams{Py})
    p1 = gnn_example_array(g1.py_params)
    p2 = gnn_example_array(g2.py_params)
    grad_diff_gnn_abs = pyconvert(T, abs(p1 - p2).mean().item())
    grad_diff_gnn_err = 100 * pyconvert(T, (abs(p1 - p2).mean() / abs(p1).mean()).item())
    return grad_diff_gnn_abs, grad_diff_gnn_err
end

grad_diff_gnn(::GNNParams{Nothing}, ::GNNParams{Nothing}) = zero(T), zero(T)

function grad_perc_sw(p1, p2)
    if iszero(p1) && iszero(p2)
        return zero(T)
    else
        return 100 * abs(p1 - p2) / abs(p1)
    end
end

inter_type = :sw_gnn
loss_type = :stress_stiff
if inter_type == :sw
    params_sw = copy(init_params_sw)
    params_gnn = GNNParams(nothing)
elseif inter_type == :gnn
    params_sw = zeros(T, n_params_sw)
    params_gnn = gnn_init_params
elseif inter_type == :sw_gnn
    params_sw = copy(init_params_sw)
    params_gnn = gnn_init_params
end
γ = T(4.0)
n_steps_equil = 2
n_steps       = 12 # May also want to change below
n_steps_loss  = 5
n_steps_trunc = n_steps
n_steps_buffer = 0
dt = T(0.0005) # ps
n_threads = Threads.nthreads()
noise_seeds = [rand_seed() for i in 1:n_steps]

noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
sys_equil, _, _, _, _, _ = forward_sim(nothing, nothing, noise_equil_seeds, γ, n_steps_equil,
                n_steps_equil, n_steps_buffer, params_sw, params_gnn, dt, n_threads, inter_type,
                loss_type, false, false)
coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

sys_fwd, loss_fwd, loss_accum_fwd, loss_logs_fwd, grads_fwd_sw, grads_fwd_gnn = forward_sim(
                coords_start, velocities_start, noise_seeds, γ, n_steps, n_steps_loss,
                n_steps_buffer, params_sw, params_gnn, dt, n_threads, inter_type, loss_type,
                true, true)

sys_rew, loss_rew, _, loss_logs_rew, grads_rew_sw, grads_rew_gnn = reweighting_sim(
                coords_start, velocities_start, noise_seeds, γ, n_steps, n_steps_loss,
                n_steps_buffer, params_sw, params_gnn, dt, n_threads, inter_type, loss_type)

sys_rev, loss_rev, grads_rev_sw, grads_rev_gnn = reverse_sim(sys_fwd.coords, sys_fwd.velocities,
            noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, n_steps_buffer, params_sw,
            params_gnn, dt, n_threads, sys_fwd.loggers, loss_accum_fwd, inter_type, loss_type)

# This will only be true if n_steps_loss == n_steps_trunc
@test maximum(norm.(coords_start .- sys_rev.coords)) < (T == Float64 ? 1e-8 : 1e-4)
@test isapprox(loss_rev, loss_fwd; rtol=1e-2)
@test isapprox(loss_rev, loss_rew; rtol=1e-2)

grad_diff_gnn_abs, grad_diff_gnn_err = grad_diff_gnn(grads_fwd_gnn, grads_rev_gnn)

println("Grads forward:        ", grads_fwd_sw, ", ", grads_fwd_gnn)
println("Grads reverse:        ", grads_rev_sw, ", ", grads_rev_gnn)
println("Diff forward reverse: ", abs.(grads_fwd_sw .- grads_rev_sw), ", ", grad_diff_gnn_abs)
println("%err forward reverse: ", grad_perc_sw.(grads_fwd_sw, grads_rev_sw), ", ", grad_diff_gnn_err)

function train()
    inter_type = :sw_gnn
    loss_type = :stress_stiff
    update_sw_params = true
    reweighting = false

    γ = T(4.0)
    n_steps_loss  = 500 # 250 fs
    n_steps_trunc = 200 # 100 fs
    n_steps_buffer = 0
    dt = T(0.0005) # ps
    n_threads = Threads.nthreads()
    n_epochs = 10_000
    learning_rate_sw = T(5e-4)
    learning_rate_gnn = T(2e-3)
    py_opt_gnn = pymod_optax.adam(learning_rate=learning_rate_gnn)
    first_epoch_save_gnn = 1
    only_latest_save_gnn = true
    grad_rerun_size = T(100_000.0)
    epochs_loss_rev, epochs_loss_fwd, epochs_loss_stress, epochs_loss_stiff = T[], T[], T[], T[]
    epochs_stress_GPa, epochs_c11_GPa, epochs_c12_GPa, epochs_c44_GPa = T[], T[], T[], T[]

    isdir(out_dir) || mkdir(out_dir)
    isdir("$out_dir/gnn_params") || mkdir("$out_dir/gnn_params")
    if isfile("$out_dir/train.log")
        epoch_n_start = countlines("$out_dir/train.log") + 1
        if inter_type == :sw
            params_sw = parse.(T, split(readlines("$out_dir/params_sw.txt")[end]))
            params_gnn = GNNParams(nothing)
        elseif inter_type == :gnn
            params_sw = zeros(T, n_params_sw)
            params_gnn = GNNParams(pymod_pickle.load(
                    pybuiltins.open("$out_dir/gnn_params/params_gnn_ep_$epoch_n_start.pkl", "rb")))
        elseif inter_type == :sw_gnn
            params_sw = parse.(T, split(readlines("$out_dir/params_sw.txt")[end]))
            params_gnn = GNNParams(pymod_pickle.load(
                    pybuiltins.open("$out_dir/gnn_params/params_gnn_ep_$epoch_n_start.pkl", "rb")))
        end

        if inter_type in (:sw, :sw_gnn)
            BSON.@load "$out_dir/opt_sw.bson" opt_sw
        end
        if inter_type in (:gnn, :sw_gnn)
            py_opt_state_gnn = pymod_pickle.load(pybuiltins.open("$out_dir/py_opt_state_gnn.pkl", "rb"))
        end

        for line in readlines("$out_dir/train.log")
            cols = split(line)
            push!(epochs_loss_rev   , parse(T, cols[2]))
            push!(epochs_loss_fwd   , parse(T, cols[3]))
            push!(epochs_loss_stress, parse(T, cols[4]))
            push!(epochs_loss_stiff , parse(T, cols[5]))
            push!(epochs_stress_GPa , parse(T, cols[6]))
            push!(epochs_c11_GPa    , parse(T, cols[7]))
            push!(epochs_c12_GPa    , parse(T, cols[8]))
            push!(epochs_c44_GPa    , parse(T, cols[9]))
        end
        println("Restarting training from epoch ", epoch_n_start)
    else
        epoch_n_start = 1
        if inter_type == :sw
            params_sw = copy(init_params_sw)
            params_gnn = GNNParams(nothing)
        elseif inter_type == :gnn
            params_sw = zeros(T, n_params_sw)
            params_gnn = gnn_init_params
        elseif inter_type == :sw_gnn
            params_sw = copy(init_params_sw)
            params_gnn = gnn_init_params
        end

        opt_sw = Optimisers.setup(Optimisers.Adam(learning_rate_sw), params_sw)
        py_opt_state_gnn = py_opt_gnn.init(params_gnn.py_params)
        println("Starting training")
    end

    epoch_n = epoch_n_start
    while epoch_n <= n_epochs
        time_start = now()
        n_steps_equil = 1
        n_steps = min(epoch_n * 1, 20_000)
        noise_seeds = [rand_seed() for i in 1:n_steps]

        if n_steps_equil > 0
            noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
            sys_equil, _, _, _, _, _ = forward_sim(nothing, nothing, noise_equil_seeds, γ,
                            n_steps_equil, n_steps_equil, n_steps_equil, params_sw, params_gnn,
                            dt, n_threads, inter_type, loss_type, false, false)
            coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)
        else
            # This will fail in the reweighting case
            coords_start, velocities_start = nothing, nothing
        end

        if reweighting
            sys_rev, loss_rev, loss_fwd, loss_logs_fwd, grads_rev_sw, grads_rev_gnn = reweighting_sim(
                        coords_start, velocities_start, noise_seeds, γ, n_steps,
                        min(n_steps_loss, n_steps), min(n_steps_buffer, n_steps), params_sw,
                        params_gnn, dt, n_threads, inter_type, loss_type)
        else
            sys_fwd, loss_fwd, loss_accum_fwd, loss_logs_fwd, _, _ = forward_sim(coords_start,
                        velocities_start, noise_seeds, γ, n_steps, min(n_steps_loss, n_steps),
                        min(n_steps_buffer, n_steps), params_sw, params_gnn, dt, n_threads,
                        inter_type, loss_type, false, true)

            sys_rev, loss_rev, grads_rev_sw, grads_rev_gnn = reverse_sim(sys_fwd.coords,
                        sys_fwd.velocities, noise_seeds, γ, n_steps, min(n_steps_loss, n_steps),
                        min(n_steps_trunc, n_steps), min(n_steps_buffer, n_steps), params_sw,
                        params_gnn, dt, n_threads, sys_fwd.loggers, loss_accum_fwd, inter_type,
                        loss_type)
        end

        # Restart the epoch if the gradients are too large
        if sum(abs, grads_rev_sw) > grad_rerun_size
            println("Re-running epoch ", epoch_n, ", sum(abs, grads_rev_sw) = ", sum(abs, grads_rev_sw))
            continue
        end

        if isnan(loss_rev) || any(isnan, grads_rev_sw) ||
                        (inter_type in (:gnn, :sw_gnn) &&
                        any(isnan, pyconvert(Matrix, gnn_example_array(grads_rev_gnn.py_params))))
            error("NaN encountered")
        end

        if inter_type in (:gnn, :sw_gnn)
            params_gnn_ex = pyconvert(T, abs(gnn_example_array(params_gnn.py_params)).mean().item())
            grads_gnn_ex  = pyconvert(T, abs(gnn_example_array(grads_rev_gnn.py_params)).mean().item())
        else
            params_gnn_ex = zero(T)
            grads_gnn_ex  = zero(T)
        end

        loss_val_stress, loss_val_stiff, c11, c12, c44 = loss_logs_fwd
        stress_GPa = approx_stress_diag(loss_val_stress)
        c11_GPa, c12_GPa, c44_GPa = conv_to_GPa * c11, conv_to_GPa * c12, conv_to_GPa * c44
        push!(epochs_loss_rev   , loss_rev)
        push!(epochs_loss_fwd   , loss_fwd)
        push!(epochs_loss_stress, loss_val_stress)
        push!(epochs_loss_stiff , loss_val_stiff)
        push!(epochs_stress_GPa , stress_GPa)
        push!(epochs_c11_GPa    , c11_GPa)
        push!(epochs_c12_GPa    , c12_GPa)
        push!(epochs_c44_GPa    , c44_GPa)

        open("$out_dir/train.log", "a") do of
            println(
                of,
                "$epoch_n $loss_rev $loss_fwd $loss_val_stress $loss_val_stiff ",
                "$stress_GPa $c11_GPa $c12_GPa $c44_GPa ",
                join(params_sw, " "), " ",
                join(grads_rev_sw, " "), " ",
                "$params_gnn_ex $grads_gnn_ex ",
                round(now() - time_start, Dates.Minute),
            )
        end

        f = Figure()
        ax = f[1, 1] = Axis(f, xlabel="Epoch number", ylabel="Loss")
        lines!(1:epoch_n, epochs_loss_rev, label="Reverse")
        lines!(1:epoch_n, epochs_loss_fwd, label="Forward")
        lines!(1:epoch_n, epochs_loss_stress, label="Stress forward")
        lines!(1:epoch_n, epochs_loss_stiff, label="Stiffness forward")
        xlims!(0, epoch_n)
        ylims!(0, min(maximum(epochs_loss_rev) + 1, 100))
        f[1, 2] = Legend(f, ax, framevisible=false)
        save("$out_dir/plot_loss.pdf", f)

        f = Figure()
        ax = f[1, 1] = Axis(f, xlabel="Epoch number", ylabel="Approximate σ1 in GPa")
        lines!([0, epoch_n], [0, 0], color=:black, linestyle=:dash)
        lines!(1:epoch_n, epochs_stress_GPa)
        xlims!(0, epoch_n)
        save("$out_dir/plot_stress.pdf", f)

        f = Figure()
        ax = f[1, 1] = Axis(f, xlabel="Epoch number", ylabel="Cij in GPa")
        for Cij in c_target
            Cij_GPa = conv_to_GPa * Cij
            lines!([0, epoch_n], [Cij_GPa, Cij_GPa], color=:black, linestyle=:dash)
        end
        lines!(1:epoch_n, epochs_c11_GPa, label="C11")
        lines!(1:epoch_n, epochs_c12_GPa, label="C12")
        lines!(1:epoch_n, epochs_c44_GPa, label="C44")
        xlims!(0, epoch_n)
        f[1, 2] = Legend(f, ax, framevisible=false)
        save("$out_dir/plot_stiff.pdf", f)

        if inter_type in (:sw, :sw_gnn)
            if epoch_n == 1
                open("$out_dir/params_sw.txt", "a") do of
                    println(of, join(params_sw, " "), " ")
                end
            end

            if update_sw_params
                opt_sw, params_sw = Optimisers.update!(opt_sw, params_sw, grads_rev_sw)
            end

            open("$out_dir/params_sw.txt", "a") do of
                println(of, join(params_sw, " "), " ")
            end
            BSON.@save "$out_dir/opt_sw.bson" opt_sw
        end
        if inter_type in (:gnn, :sw_gnn)
            if epoch_n == 1 && first_epoch_save_gnn == 1
                pywith(pybuiltins.open("$out_dir/gnn_params/params_gnn_ep_$epoch_n.pkl", "wb")) do py_of
                    pymod_pickle.dump(params_gnn.py_params, py_of)
                end
            end

            py_updates, py_opt_state_gnn = py_opt_gnn.update(grads_rev_gnn.py_params,
                                                py_opt_state_gnn, params_gnn.py_params)
            params_gnn = GNNParams(pymod_optax.apply_updates(params_gnn.py_params, py_updates))

            if (epoch_n + 1) >= first_epoch_save_gnn
                pywith(pybuiltins.open("$out_dir/gnn_params/params_gnn_ep_$(epoch_n + 1).pkl", "wb")) do py_of
                    pymod_pickle.dump(params_gnn.py_params, py_of)
                end
                pywith(pybuiltins.open("$out_dir/py_opt_state_gnn.pkl", "wb")) do py_of
                    pymod_pickle.dump(py_opt_state_gnn, py_of)
                end
            end
            if only_latest_save_gnn && isfile("$out_dir/gnn_params/params_gnn_ep_$epoch_n.pkl")
                rm("$out_dir/gnn_params/params_gnn_ep_$epoch_n.pkl")
            end
        end

        epoch_n += 1
    end
end

#=
# Benchmark
for rep in 1:3
    reset_timer!(to)
    forward_sim(coords_start, velocities_start, noise_seeds, γ, 10, 10, 0, params_sw, params_gnn,
                dt, n_threads, inter_type, loss_type, false, false) # Don't time loss with 10 steps
    println()
    show(to)
    println()
end
for rep in 1:3
    reset_timer!(to)
    reverse_sim(sys_fwd.coords, sys_fwd.velocities, noise_seeds, γ, 10, 10, 10, 0,
                params_sw, params_gnn, dt, n_threads, sys_fwd.loggers, loss_accum_fwd,
                inter_type, loss_type)
    println()
    show(to)
    println()
end
=#
#=
# Run longer simulations for validation (Figure 4C-D)
model_dir = "trained_model"
epoch_n = 1072
for n_steps in [
            2_000,   # 1 ps
            10_000,  # 5 ps
            20_000,  # 10 ps
            50_000,  # 25 ps
            100_000, # 50 ps
            200_000, # 100 ps
        ]
    params_sw = parse.(T, split(readlines("$model_dir/params_sw.txt")[epoch_n]))
    params_gnn = GNNParams(pymod_pickle.load(
                pybuiltins.open("$model_dir/params_gnn_ep_$epoch_n.pkl", "rb")))
    γ = T(4.0)
    n_steps_loss  = 500 # 250 fs
    n_steps_buffer = 0
    dt = T(0.0005) # ps
    n_threads = Threads.nthreads()
    inter_type = :sw_gnn
    loss_type = :stress_stiff
    noise_seeds = [rand_seed() for i in 1:n_steps]

    sys_fwd, loss_fwd, loss_accum_fwd, loss_logs_fwd, _, _ = forward_sim(nothing,
                    nothing, noise_seeds, γ, n_steps, min(n_steps_loss, n_steps),
                    min(n_steps_buffer, n_steps), params_sw, params_gnn, dt, n_threads,
                    inter_type, loss_type, false, true)

    loss_val_stress, loss_val_stiff, c11, c12, c44 = loss_logs_fwd
    stress_GPa = approx_stress_diag(loss_val_stress)
    c11_GPa, c12_GPa, c44_GPa = conv_to_GPa * c11, conv_to_GPa * c12, conv_to_GPa * c44
    println(
        "$n_steps $loss_fwd $loss_val_stress $loss_val_stiff ",
        "$stress_GPa $c11_GPa $c12_GPa $c44_GPa",
    )
end
=#

train()
