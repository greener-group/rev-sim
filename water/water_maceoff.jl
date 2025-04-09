# Reversible simulation for MACE-OFF water
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT
# An appropriate conda environment should be activated
# Use CUDA_VISIBLE_DEVICES to set the GPU to run on

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Molly
using Zygote
import Enzyme
import AtomsCalculators
using PythonCall
using UnitfulAtomic
using ChainRulesCore
using TimerOutputs
using LinearAlgebra
using Random
using Test

Enzyme.API.runtimeActivity!(true)

const out_dir        = (length(ARGS) >= 1 ? ARGS[1] : "train_water_maceoff")
const run_inter_type = (length(ARGS) >= 2 ? ARGS[2] : "maceoff")

const pymod_torch = pyimport("torch")
const pymod_mc = pyimport("mace.calculators")
const py_ase_calc = pymod_mc.mace_off(model="small", device="cuda", default_dtype="float32")

const T = Float64
const ff = MolecularForceField(
    T,
    joinpath(dirname(pathof(Molly)), "..", "data", "force_fields", "tip3p_standard.xml");
    units=false,
)
const structure_file = "tip3p.pdb"
const n_molecules = 895
const n_atoms = n_molecules * 3
const n_atoms_x3 = n_atoms * 3
const rdf_exp_file = "soper_2013_water_rdf.txt"
const n_steps_log = 400
const O_mass = T(15.99943)
const H_mass = T(1.007947)
const atom_masses = repeat([O_mass, H_mass, H_mass]; outer=n_molecules)
const temp = T(295.15) # K
const two_pow_1o6 = T(2^(1/6))
const conv_eV_to_kJpmol = T(96.48533212331)
const conv_eVpÅ_to_kJpmolpnm = T(964.8533212331)
const rdf_loss_weighting = T(1.0)
const to = TimerOutput()

coords_copy(sys, neighbors=nothing; kwargs...) = copy(sys.coords)
velocities_copy(sys, neighbors=nothing; kwargs...) = copy(sys.velocities)
CoordCopyLogger() = GeneralObservableLogger(coords_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)
VelocityCopyLogger() = GeneralObservableLogger(velocities_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)

const py_script = """
import ase
global ase
import torch
global torch

global atoms
atoms = ase.io.read("$structure_file")

global calc_global
calc_global = calc
global model_global
model_global = calc_global.models[0]
nn_model = model_global
for param in model_global.parameters():
    param.requires_grad = True

def nn_params():
    return [p.detach().cpu() for p in model_global.parameters()]

def zero_params():
    return [torch.zeros_like(p, device="cpu") for p in model_global.parameters()]

# This should be called before calling potential_energy etc. from Julia
def update_params(params_new):
    with torch.no_grad():
        for param, param_new in zip(model_global.parameters(), params_new):
            param.copy_(param_new)

def add_parameters(params_1, params_2):
    return [p1 + p2 for p1, p2 in zip(params_1, params_2)]

def multiply_parameters(params, n):
    return [p * n for p in params]

def forces(coords, box_sides, detach):
    atoms.set_cell(box_sides)
    # Coordinates set here for edges and later for gradients
    atoms.set_positions(coords.detach().cpu().numpy())
    batch = calc_global._atoms_to_batch(atoms)
    batch.positions = coords
    fs = model_global(
        batch.to_dict(),
        compute_stress=False,
        training=True,
    )["forces"]
    if detach:
        return fs.detach().cpu()
    else:
        return fs

global forces_global
forces_global = forces

def potential_energy(coords, box_sides, detach):
    atoms.set_cell(box_sides)
    atoms.set_positions(coords.detach().cpu().numpy())
    batch = calc_global._atoms_to_batch(atoms)
    batch.positions = coords
    pe = model_global(
        batch.to_dict(),
        compute_stress=False,
        training=True,
    )["energy"]
    if detach:
        return pe.item()
    else:
        return pe

global potential_energy_global
potential_energy_global = potential_energy

def grad_forces_sum(coords, box_sides, d_fs):
    fs = forces_global(coords, box_sides, False)
    l = (fs * d_fs).sum()
    l.backward()
    model_grads = [p.grad.detach().cpu() for p in model_global.parameters()]
    model_global.zero_grad()
    coords_grads = coords.grad.detach().cpu()
    return model_grads, coords_grads

def grad_potential_energy_sum(coords, box_sides, d_pe):
    pe = potential_energy_global(coords, box_sides, False)
    l = pe * d_pe
    l.backward()
    model_grads = [p.grad.detach().cpu() for p in model_global.parameters()]
    model_global.zero_grad()
    coords_grads = coords.grad.detach().cpu()
    return model_grads, coords_grads

def rev_grad_1(coords, box_sides, accum_C, atom_masses_3N):
    fs = forces_global(coords, box_sides, False) * $conv_eVpÅ_to_kJpmolpnm
    as_flat = fs.reshape(-1) / atom_masses_3N
    l = (as_flat * accum_C).sum()
    l.backward()
    model_global.zero_grad()
    coords_grads = coords.grad.detach().cpu()
    return coords_grads.reshape(-1)

def rev_grad_2(coords, box_sides, accum_A, atom_masses_3N, dt2):
    fs = forces_global(coords, box_sides, False) * $conv_eVpÅ_to_kJpmolpnm
    as_flat_dt = (fs.reshape(-1) / atom_masses_3N) * (dt2 / 2)
    l = (as_flat_dt * accum_A).sum()
    l.backward()
    model_grads = [p.grad.detach().cpu() for p in model_global.parameters()]
    model_global.zero_grad()
    return model_grads

def optimiser_step(opt, grads):
    model_global.zero_grad()
    for param, grad in zip(model_global.parameters(), grads):
        param.grad = grad.to("cuda")
    opt.step()
    model_global.zero_grad()
"""

const py_functions = pyexec(
    @NamedTuple{nn_model, nn_params, zero_params, update_params, add_parameters,
                multiply_parameters, forces, potential_energy, grad_forces_sum,
                grad_potential_energy_sum, rev_grad_1, rev_grad_2, optimiser_step},
    py_script,
    Main,
    (calc=py_ase_calc,),
)

const py_nn_params_start = py_functions.nn_params()

struct NNParams
    py_params::Py
end

Base.show(io::IO, p::NNParams) = print(io, "NNParams")

function Base.:+(p1::NNParams, p2::NNParams)
    return NNParams(py_functions.add_parameters(p1.py_params, p2.py_params))
end

function Base.:*(p::NNParams, n::Real)
    return NNParams(py_functions.multiply_parameters(p.py_params, n))
end
Base.:*(n::Real, p::NNParams) = p * n

struct MACEOFF
    params::NNParams
end

function AtomsCalculators.forces(sys, inter::MACEOFF; kwargs...)
    return forces_nn(inter.params, sys.coords, sys.boundary) .* conv_eVpÅ_to_kJpmolpnm
end

function AtomsCalculators.potential_energy(sys, inter::MACEOFF; kwargs...)
    return potential_energy_nn(inter.params, sys.coords, sys.boundary) * conv_eV_to_kJpmol
end

function svecs_to_pytorch_array(svecs, requires_grad=true)
    arr = Array(transpose(reshape(reinterpret(T, svecs), 3, length(svecs))))
    py_arr = pymod_torch.tensor(Py(arr).to_numpy(), dtype=pymod_torch.float,
                                device="cuda", requires_grad=requires_grad)
    return py_arr
end

function forces_nn(params, coords, boundary)
    py_coords = svecs_to_pytorch_array(coords .* 10) # Convert to Å
    py_box_lengths = pylist(boundary.side_lengths .* 10)
    py_fs = py_functions.forces(py_coords, py_box_lengths, true)
    fs = pyconvert(Matrix{T}, py_fs)
    return SVector{3, T}.(eachrow(fs)) # This is in eV/Å
end

function ChainRulesCore.rrule(::typeof(forces_nn), params, coords, boundary)
    Y = forces_nn(params, coords, boundary)

    function forces_nn_pullback(d_fs)
        py_coords = svecs_to_pytorch_array(coords .* 10)
        py_box_lengths = pylist(boundary.side_lengths .* 10)
        py_d_fs = svecs_to_pytorch_array(d_fs, false)
        py_dl_dp, py_dl_dc = py_functions.grad_forces_sum(py_coords, py_box_lengths, py_d_fs)
        dl_dp = NNParams(py_dl_dp)
        dl_dc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dc)))
        return NoTangent(), dl_dp, dl_dc, NoTangent()
    end

    return Y, forces_nn_pullback
end

function potential_energy_nn(params, coords, boundary)
    py_coords = svecs_to_pytorch_array(coords .* 10)
    py_box_lengths = pylist(boundary.side_lengths .* 10)
    py_pe = py_functions.potential_energy(py_coords, py_box_lengths, true)
    return pyconvert(T, py_pe) # This is in eV
end

function ChainRulesCore.rrule(::typeof(potential_energy_nn), params, coords, boundary)
    Y = potential_energy_nn(params, coords, boundary)

    function potential_energy_nn_pullback(d_pe)
        py_coords = svecs_to_pytorch_array(coords .* 10)
        py_box_lengths = pylist(boundary.side_lengths .* 10)
        py_dl_dp, py_dl_dc = py_functions.grad_potential_energy_sum(py_coords, py_box_lengths, d_pe)
        dl_dp = NNParams(py_dl_dp)
        dl_dc = SVector{3, T}.(eachrow(pyconvert(Matrix{T}, py_dl_dc)))
        return NoTangent(), dl_dp, dl_dc, NoTangent()
    end

    return Y, potential_energy_nn_pullback
end

function read_rdf_data(rdf_exp_file)
    rdf_OO, rdf_OH, rdf_HH = T[], T[], T[]
    for line in readlines(rdf_exp_file)[3:end] # Skip 0.00 and 0.01 lines
        cols = split(line)
        push!(rdf_OO, parse(T, cols[2]))
        push!(rdf_OH, parse(T, cols[4]))
        push!(rdf_HH, parse(T, cols[6]))
    end
    return rdf_OO[87:334], # 248 values in 0.261:0.003:1.002 (2.61 Å -> 10.02 Å)
           rdf_OH[54:334], # 281 values in 0.162:0.003:1.002 (1.62 Å -> 10.02 Å)
           rdf_HH[74:334]  # 261 values in 0.222:0.003:1.002 (2.22 Å -> 10.02 Å)
end

const rdf_exp_OO, rdf_exp_OH, rdf_exp_HH = read_rdf_data(rdf_exp_file)

function rdf_differentiable(coords_1::Vector{SVector{3, T}}, coords_2::Vector{SVector{3, T}},
                            boundary, d=T(0.0005), x_min=T(0.1), x_max=T(2.5),
                            exp_cutoff=T(0.04)) where T
    n_atoms_1, n_atoms_2 = length(coords_1), length(coords_2)
    dists = T[]
    @inbounds for i in 1:n_atoms_1
        for j in 1:n_atoms_2
            dist = norm(vector(coords_1[i], coords_2[j], boundary))
            if !iszero(dist)
                push!(dists, dist)
            end
        end
    end
    sort!(dists)
    d_min, d_max = dists[1], dists[end]

    # See Wang2023 equations 4 and 6
    xs = [x_min + d * (i - 1) for i in 1:(Int(cld(x_max - x_min, d)) + 1)] # Range has Enzyme issue
    # Pre-calculate sum in denominator for each distance
    denoms = zero(dists)
    n_chunks = Threads.nthreads() * 10
    offset = Int(x_min / d) - 1
    Threads.@threads for chunk_i in 1:n_chunks
        @inbounds for di in chunk_i:n_chunks:length(dists)
            sum_pair = zero(T)
            dist = dists[di]
            start_i = max(Int(cld(dist - exp_cutoff, d)) - offset, 1)
            end_i   = min(Int(fld(dist + exp_cutoff, d)) - offset, length(xs))
            for xi in start_i:end_i
                sum_pair += exp(-((xs[xi] - dist)^2) / d)
            end
            denoms[di] = sum_pair
        end
    end

    # Calculate p(μ_k)
    # Equation 4 is missing d term?
    pre_fac = box_volume(boundary) / (4 * T(π) * n_atoms_1 * n_atoms_2 * d)
    hist = zero(xs)
    Threads.@threads for chunk_i in 1:n_chunks
        @inbounds for xi in chunk_i:n_chunks:length(xs)
            sum_pair = zero(T)
            x = xs[xi]
            x_p_cutoff, x_m_cutoff = x + exp_cutoff, x - exp_cutoff
            reading = false
            for di in eachindex(dists)
                dist = dists[di]
                if dist > x_p_cutoff
                    break
                elseif !reading && dist >= x_m_cutoff
                    reading = true
                end
                if reading
                    sum_pair += exp(-((x - dist)^2) / d) / denoms[di]
                end
            end
            hist[xi] = pre_fac * sum_pair / x^2
        end
    end
    return xs, hist
end

function loss_rdf_part(coords_1, coords_2, boundary, rdf_exp, rng)
    rdf_sim = rdf_differentiable(coords_1, coords_2, boundary)[2][rng]
    @assert length(rdf_sim) == length(rdf_exp)
    return sum(abs.(rdf_sim .- rdf_exp))
end

const coords_O_inds = 1:3:n_atoms
const coords_H_inds = [i for i in 1:n_atoms if (i + 2) % 3 != 0]

function loss_rdf(coords, boundary)
    coords_O = coords[coords_O_inds]
    coords_H = coords[coords_H_inds]
    # Slices only works for this d and x_min
    loss_OO = loss_rdf_part(coords_O, coords_O, boundary, rdf_exp_OO, 323:6:1805)
    loss_OH = loss_rdf_part(coords_O, coords_H, boundary, rdf_exp_OH, 125:6:1805)
    # HH RDF gives OOM error
    loss_HH = zero(T) # loss_rdf_part(coords_H, coords_H, boundary, rdf_exp_HH, 245:6:1805)
    return loss_OO + loss_OH + loss_HH
end

loss_rdf(sys) = loss_rdf(sys.coords, sys.boundary)

function ChainRulesCore.rrule(::typeof(loss_rdf_part), coords_1, coords_2, boundary, rdf_exp, rng)
    Y = loss_rdf_part(coords_1, coords_2, boundary, rdf_exp, rng)

    function loss_rdf_part_pullback(dy)
        d_coords_1 = zero(coords_1)
        d_coords_2 = zero(coords_2)
        Enzyme.autodiff(
            Enzyme.Reverse,
            loss_rdf_part,
            Enzyme.Active,
            Enzyme.Duplicated(coords_1, d_coords_1),
            Enzyme.Duplicated(coords_2, d_coords_2),
            Enzyme.Const(boundary),
            Enzyme.Const(rdf_exp),
            Enzyme.Const(rng),
        )
        return NoTangent(), dy .* d_coords_1, dy .* d_coords_2, NoTangent(),
               NoTangent(), NoTangent()
    end

    return Y, loss_rdf_part_pullback
end

function enthalpy_vaporization(sys)
    # See https://docs.openforcefield.org/projects/evaluator/en/stable/properties/properties.html
    RT = T(2.45) # uconvert(u"kJ/mol", Unitful.R * 295.15u"K")
    mean_U_gas = T(-200794.0)
    snapshot_U_liquid = potential_energy(sys) / n_molecules
    ΔH_vap = mean_U_gas - snapshot_U_liquid + RT
    return ΔH_vap
end

function loss_enth_vap(sys)
    ΔH_vap = enthalpy_vaporization(sys)
    # See Glattli2002 and https://www.engineeringtoolbox.com/water-properties-d_1573.html
    # 44.12 kJ/mol at 295.15 K plus amount to account for not using bond/angle constraints
    ΔH_vap_exp = T(44.12)
    return (ΔH_vap - ΔH_vap_exp) ^ 2
end

function loss_fn(sys, loss_type)
    l_rdf, l_ev = zero(T), zero(T)
    if loss_type == :rdf
        l_rdf = rdf_loss_weighting * loss_rdf(sys)
    elseif loss_type == :enth_vap
        l_ev = loss_enth_vap(sys)
    elseif loss_type == :both
        l_rdf = rdf_loss_weighting * loss_rdf(sys)
        l_ev = loss_enth_vap(sys)
    else
        throw(ArgumentError("loss_type not recognised"))
    end
    return l_rdf + l_ev, l_rdf, l_ev
end

function loss_and_grads(sys, loss_type)
    (l, l_rdf, l_ev), grads = withgradient(loss_fn, sys, loss_type)
    sys_grads = grads[1]
    dl_dc = sys_grads.coords
    if isnothing(sys_grads.general_inters)
        dl_dp = NNParams(py_functions.zero_params())
    else
        dl_dp = sys_grads.general_inters[1].params
    end
    return l, l_rdf, l_ev, dl_dc, dl_dp
end

mutable struct MonteCarloBarostatNoUnits{T, P, K, V}
    pressure::P
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Int
    n_accepted::Int
end

function MonteCarloBarostatNoUnits(P, T, boundary; n_steps=30, n_iterations=1, scale_factor=0.01,
                        scale_increment=1.1, max_volume_frac=0.3, trial_find_neighbors=false)
    volume_scale = box_volume(boundary) * float_type(boundary)(scale_factor)
    return MonteCarloBarostatNoUnits(P, T, n_steps, n_iterations, volume_scale, scale_increment,
                                     max_volume_frac, trial_find_neighbors, 0, 0)
end

function Molly.apply_coupling!(sys::System{D, G, T}, barostat::MonteCarloBarostatNoUnits, sim,
                        neighbors=nothing, step_n::Integer=0;
                        n_threads::Integer=Threads.nthreads()) where {D, G, T}
    if !iszero(step_n % barostat.n_steps)
        return false
    end

    kT = sys.k * barostat.temperature * T(ustrip(u"kJ", 1.0u"u * nm^2 * ps^-2")) # kJ
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false

    for attempt_n in 1:barostat.n_iterations
        E = potential_energy(sys, neighbors; n_threads=n_threads)
        V = box_volume(sys.boundary)
        dV = barostat.volume_scale * (2 * rand(T) - 1)
        v_scale = (V + dV) / V
        l_scale = (D == 2 ? sqrt(v_scale) : cbrt(v_scale))
        old_coords = copy(sys.coords)
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

        if barostat.trial_find_neighbors
            neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                             n_threads=n_threads)
        else
            # Assume neighbors are unchanged by the change in coordinates
            # This may not be valid for larger changes
            neighbors_trial = neighbors
        end
        E_trial = potential_energy(sys, neighbors_trial; n_threads=n_threads)
        dE = (E_trial - E) / T(ustrip(Unitful.Na)) # kJ
        dW = dE + uconvert(unit(dE), barostat.pressure * dV) - n_molecules * kT * log(v_scale)
        if dW <= zero(dW) || rand(T) < exp(-dW / kT)
            recompute_forces = true
            barostat.n_accepted += 1
        else
            sys.coords = old_coords
            sys.boundary = old_boundary
        end
        barostat.n_attempted += 1

        # Modify size of volume change to keep accept/reject ratio roughly equal
        if barostat.n_attempted >= 10
            if barostat.n_accepted < 0.25 * barostat.n_attempted
                barostat.volume_scale /= barostat.scale_increment
                barostat.n_attempted = 0
                barostat.n_accepted = 0
            elseif barostat.n_accepted > 0.75 * barostat.n_attempted
                barostat.volume_scale = min(barostat.volume_scale * barostat.scale_increment,
                                            V * barostat.max_volume_frac)
                barostat.n_attempted = 0
                barostat.n_accepted = 0
            end
        end
    end
    return recompute_forces
end

function generate_noise!(noise, seed, k=T(ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)))
    rng = Xoshiro(seed)
    @inbounds for i in eachindex(noise)
        noise[i] = Molly.random_velocity_3D(atom_masses[i], temp, k, rng)
    end
    return noise
end

rand_seed() = rand(0:typemax(Int))

function forward_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, params,
                     dt, n_threads, inter_type, loss_type, run_grads=true, calc_loss=true,
                     use_barostat=false)
    sys_init = System(structure_file, ff; units=false)
    sys_init.coords .= wrap_coords.(sys_init.coords, (sys_init.boundary,))
    py_functions.update_params(params.py_params)
    general_inters = (MACEOFF(params),)
    sys = System(
        atoms=sys_init.atoms,
        coords=copy(isnothing(coords) ? sys_init.coords : coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        general_inters=general_inters,
        loggers=(
            coords=CoordCopyLogger(),
            velocities=VelocityCopyLogger(),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    n_blocks = (n_steps == 0 ? 1 : n_steps ÷ n_steps_loss)
    if use_barostat
        coupling = MonteCarloBarostatNoUnits(
            T(ustrip(u"kJ * nm^-3", 1.0u"bar")), # kJ * nm^-3
            temp,
            sys.boundary,
        )
    else
        coupling = NoCoupling()
    end

    vel_scale = exp(-γ * dt)
    noise_scale = sqrt(1 - vel_scale^2)

    fs = zero(sys.coords)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = nothing
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    loss_accum = zero(T)
    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "f.1" fs .= AtomsCalculators.forces(sys, sys.general_inters[1])
        @timeit to "f.2" sys.velocities .+= fs .* dt ./ masses(sys)
        @timeit to "f.3" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.4" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.5" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.6" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.7" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "f.8" run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
        @timeit to "f.9" if calc_loss && (step_n % n_steps_loss == 0)
            loss_accum += loss_fn(sys, loss_type)[1]
        end
        @timeit to "f.10" apply_coupling!(sys, coupling, nothing, neighbors, step_n;
                                          n_threads=n_threads)
    end
    loss = loss_accum / n_blocks
    if use_barostat
        println("fwd box size ", sys.boundary[1], " nm")
        println("fwd density ", 2.677405863226618e-23 / (box_volume(sys.boundary) * 1e-27), " kg / m^3")
    end

    if run_grads
        noises = [zeros(SVector{3, T}, n_atoms) for i in 1:n_steps]
        for step_n in 1:n_steps
            @inbounds generate_noise!(noises[step_n], noise_seeds[step_n])
        end

        function loss_forward(params)
            general_inters = (MACEOFF(params),)
            sys_grad = System(
                atoms=sys_init.atoms,
                coords=copy(isnothing(coords) ? sys_init.coords : coords),
                boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
                velocities=copy(velocities),
                general_inters=general_inters,
                force_units=NoUnits,
                energy_units=NoUnits,
            )
            loss_accum_rad = zero(T)
            neighbors_grad = nothing
            for step_n in 1:n_steps
                fs_grad = AtomsCalculators.forces(sys_grad, sys_grad.general_inters[1])
                sys_grad.velocities += fs_grad .* dt ./ masses(sys_grad)
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.velocities = sys_grad.velocities .* vel_scale .+ noises[step_n] .* noise_scale
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.coords = wrap_coords.(sys_grad.coords, (sys_grad.boundary,))
                if step_n % n_steps_loss == 0
                    loss_accum_rad += loss_fn(sys_grad, loss_type)[1]
                end
            end
            loss_rad = loss_accum_rad / n_blocks
            return loss_rad
        end

        @timeit to "forward grad" begin
            loss_rad, grads_tup = withgradient(loss_forward, params)
            grads = grads_tup[1]
        end
        # Loss can vary between runs so the losses might not match
        println("fwd loss ", loss_rad)
        println("fwd loss primal ", loss)
        println("fwd gradients ", grads)
    else
        grads = NNParams(py_functions.zero_params())
    end

    return sys, loss, grads
end

function reweighting_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, params,
                         dt, n_threads, inter_type, loss_type, use_barostat=false)
    sys_init = System(structure_file, ff; units=false)
    sys_init.coords .= wrap_coords.(sys_init.coords, (sys_init.boundary,))
    py_functions.update_params(params.py_params)
    general_inters = (MACEOFF(params),)
    sys = System(
        atoms=sys_init.atoms,
        coords=copy(isnothing(coords) ? sys_init.coords : coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        general_inters=general_inters,
        loggers=(
            coords=CoordCopyLogger(),
            velocities=VelocityCopyLogger(),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    n_blocks = (n_steps == 0 ? 1 : n_steps ÷ n_steps_loss)
    if use_barostat
        coupling = MonteCarloBarostatNoUnits(
            T(ustrip(u"kJ * nm^-3", 1.0u"bar")), # kJ * nm^-3
            temp,
            sys.boundary,
        )
    else
        coupling = NoCoupling()
    end

    vel_scale = exp(-γ * dt)
    noise_scale = sqrt(1 - vel_scale^2)

    fs = zero(sys.coords)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = nothing
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    loss_accum     = zero(T)
    loss_accum_rdf = zero(T)
    loss_accum_ev  = zero(T)
    dl_dp_accum    = NNParams(py_functions.zero_params())
    dE_dp_accum    = NNParams(py_functions.zero_params())
    l_dE_dp_accum  = NNParams(py_functions.zero_params())

    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "f.1" fs .= AtomsCalculators.forces(sys, sys.general_inters[1])
        @timeit to "f.2" sys.velocities .+= fs .* dt ./ masses(sys)
        @timeit to "f.3" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.4" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.5" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.6" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.7" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "f.8" run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
        @timeit to "f.9" if step_n % n_steps_loss == 0
            loss_block, loss_block_rdf, loss_block_ev, _, dl_dp_accum_block = loss_and_grads(sys, loss_type)
            loss_accum     += loss_block
            loss_accum_rdf += loss_block_rdf
            loss_accum_ev  += loss_block_ev
            dl_dp_accum = dl_dp_accum + dl_dp_accum_block

            sys_grads = gradient(potential_energy, sys, neighbors)[1]
            dE_dp_block = sys_grads.general_inters[1].params
            dE_dp_accum = dE_dp_accum + dE_dp_block
            l_dE_dp_accum = l_dE_dp_accum + loss_block * dE_dp_block
        end
        @timeit to "f.10" apply_coupling!(sys, coupling, nothing, neighbors, step_n;
                                          n_threads=n_threads)
    end
    if use_barostat
        println("fwd box size ", sys.boundary[1], " nm")
        println("fwd density ", 2.677405863226618e-23 / (box_volume(sys.boundary) * 1e-27), " kg / m^3")
    end

    loss  = loss_accum / n_blocks
    l_rdf = loss_accum_rdf / n_blocks
    l_ev  = loss_accum_ev / n_blocks
    β = inv(sys.k * temp)
    grads = dl_dp_accum * T(inv(n_blocks)) + -β * (l_dE_dp_accum * T(inv(n_blocks)) +
                                                  (-loss * T(inv(n_blocks))) * dE_dp_accum)

    return sys, loss, grads, l_rdf, l_ev
end

function nn_rev_grads!(dl_dfi, coords, boundary, accum_A, accum_C, atom_masses_3N, dt2)
    py_coords = svecs_to_pytorch_array(coords .* 10)
    py_box_lengths = pylist(boundary.side_lengths .* 10)
    py_accum_A = pymod_torch.tensor(Py(accum_A).to_numpy(), dtype=pymod_torch.float,
                                    device="cuda", requires_grad=false)
    py_accum_C = pymod_torch.tensor(Py(accum_C).to_numpy(), dtype=pymod_torch.float,
                                    device="cuda", requires_grad=false)
    py_atom_masses_3N = pymod_torch.tensor(Py(atom_masses_3N).to_numpy(), dtype=pymod_torch.float,
                                           device="cuda", requires_grad=false)
    py_dl_dfi_nn = py_functions.rev_grad_1(py_coords, py_box_lengths, py_accum_C,
                                           py_atom_masses_3N)
    dl_dfi .= pyconvert(Vector{T}, py_dl_dfi_nn)
    py_dl_dp_i_nn = py_functions.rev_grad_2(py_coords, py_box_lengths, py_accum_A,
                                            py_atom_masses_3N, dt2)
    dl_dp_i_nn = NNParams(py_dl_dp_i_nn)
    return dl_dp_i_nn
end

function reverse_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc,
                     params, dt, n_threads, fwd_loggers, clip_norm_threshold, inter_type, loss_type)
    if n_steps_trunc > n_steps_loss
        throw(ArgumentError("n_steps_trunc must not be greater than n_steps_loss"))
    end

    sys_init = System(structure_file, ff; units=false)
    py_functions.update_params(params.py_params)
    general_inters = (MACEOFF(params),)
    sys = System(
        atoms=sys_init.atoms,
        coords=copy(coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        general_inters=general_inters,
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    exp_mγdt = exp(-γ * dt)
    noise_scale = sqrt(1 - exp_mγdt^2)
    n_blocks = n_steps ÷ n_steps_loss
    fwd_coords_logger = values(fwd_loggers.coords)
    fwd_velocities_logger = values(fwd_loggers.velocities)

    loss_accum, loss_accum_rdf, loss_accum_ev, dl_dfc_raw, dl_dp_accum = loss_and_grads(sys, loss_type)
    dl_dfc = Vector(reinterpret(T, dl_dfc_raw))
    atom_masses_3N = repeat(masses(sys); inner=3)
    dt2 = dt^2
    dl_dfi = zeros(T, n_atoms_x3)
    accum_A = dl_dfc .* (1 .+ exp_mγdt)
    accum_B = dl_dfc .* (1 .+ exp_mγdt)
    accum_C = zeros(T, n_atoms_x3)
    accum_D = dl_dfc .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt)

    fs = zero(sys.coords)
    noise = zeros(SVector{3, T}, n_atoms)

    trunc_count, log_count = 0, 0
    @timeit to "rev loop" for step_n in n_steps:-1:1
        trunc_count += 1
        if trunc_count <= n_steps_trunc
            @timeit to "loop part 1" begin
                @timeit to "1.a" sys.coords .= sys.coords .- sys.velocities .* dt ./ 2
                @timeit to "1.b" @inbounds generate_noise!(noise, noise_seeds[step_n])
                @timeit to "1.c" sys.velocities .= (sys.velocities .- noise .* noise_scale) ./ exp_mγdt
                @timeit to "1.d" sys.coords .= sys.coords .- sys.velocities .* dt ./ 2
                @timeit to "1.e" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
                @timeit to "1.f" accum_D .= exp_mγdt .* accum_D .+ dt2 .* dl_dfi .* (1 .+ exp_mγdt) ./ 2
                @timeit to "1.g" accum_C .= accum_C .+ accum_D
            end
            @timeit to "g.a" fs .= AtomsCalculators.forces(sys, sys.general_inters[1])
            @timeit to "g.b" dl_dp_i_nn = nn_rev_grads!(dl_dfi, sys.coords, sys.boundary,
                                    accum_A, accum_C, atom_masses_3N, dt2)
            @timeit to "loop part 2" begin
                @timeit to "2.a" begin
                    norm_dl_dfi = norm(dl_dfi)
                    if norm_dl_dfi > clip_norm_threshold
                        dl_dfi .= dl_dfi .* clip_norm_threshold ./ norm_dl_dfi
                    end
                end
                @timeit to "2.b" sys.velocities .= sys.velocities .- fs .* dt ./ masses(sys)
                @timeit to "2.c" dl_dp_accum = dl_dp_accum + dl_dp_i_nn
                @timeit to "2.d" accum_B .= exp_mγdt .* accum_B .+ dt2 .* dl_dfi
                @timeit to "2.e" accum_A .= accum_A .+ accum_B
            end
        end

        @timeit to "loop part 3" begin
            if (step_n - 1) % n_steps_log == 0 && step_n > 1
                log_count += 1
                if trunc_count <= n_steps_trunc || (step_n - 1) % n_steps_loss == 0
                    sys.coords .= fwd_coords_logger[end - log_count]
                    sys.velocities .= fwd_velocities_logger[end - log_count]
                end
            end
            if (step_n - 1) % n_steps_loss == 0 && step_n > 1 # Don't accum loss at step 0
                trunc_count = 0
                @timeit to "3.a" loss_block, loss_block_rdf, loss_block_ev, dl_dfc_raw_block, dl_dp_accum_block = loss_and_grads(sys, loss_type)
                loss_accum     += loss_block
                loss_accum_rdf += loss_block_rdf
                loss_accum_ev  += loss_block_ev
                dl_dfc .= Vector(reinterpret(T, dl_dfc_raw_block))
                dl_dp_accum = dl_dp_accum + dl_dp_accum_block
                dl_dfi .= zeros(T, n_atoms_x3)
                accum_A .= dl_dfc .* (1 .+ exp_mγdt)
                accum_B .= dl_dfc .* (1 .+ exp_mγdt)
                accum_C .= zeros(T, n_atoms_x3)
                accum_D .= dl_dfc .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt)
            end
        end
    end
    l     = loss_accum / n_blocks
    l_rdf = loss_accum_rdf / n_blocks
    l_ev  = loss_accum_ev / n_blocks
    dl_dp = dl_dp_accum * T(inv(n_blocks))
    println("rev loss ", l)
    println("rev gradient ", dl_dp)

    return sys, l, dl_dp, l_rdf, l_ev
end

inter_type = :maceoff
loss_type = :both
params = NNParams(py_nn_params_start)
γ = T(1.0)
n_steps_equil = 10
n_steps       = 10 # May also want to change below
n_steps_loss  = n_steps # Should be a factor of n_steps
n_steps_trunc = n_steps
dt = T(0.0005) # ps
n_threads = Threads.nthreads()
noise_seeds = [rand_seed() for i in 1:n_steps]

noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
velocities_equil = [random_velocity(m, temp) for m in atom_masses]

sys_equil, _, _ = forward_sim(nothing, velocities_equil, nothing, noise_equil_seeds, γ,
                    n_steps_equil, n_steps_equil, params, dt, n_threads, inter_type,
                    loss_type, false, false)
coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

sys_forward, loss_fwd, grads_forward = forward_sim(coords_start, velocities_start, nothing,
                    noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads,
                    inter_type, loss_type)

sys_rew, loss_rew, grads_rew, _, _ = reweighting_sim(coords_start, velocities_start, nothing,
                    noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads,
                    inter_type, loss_type)

sys_reverse, loss_rev, grads_reverse, _, _ = reverse_sim(sys_forward.coords, sys_forward.velocities,
                    nothing, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, params, dt,
                    n_threads, sys_forward.loggers, T(Inf), inter_type, loss_type)

# This will only be true if n_steps_loss == n_steps_trunc
@test maximum(norm.(coords_start .- sys_reverse.coords)) < (T == Float64 ? 1e-6 : 1e-3)

function train()
    reweighting = false
    inter_type = :maceoff
    loss_type = :both
    γ = T(1.0)
    dt = T(0.0005) # ps
    n_steps = 20_000 # 10 ps
    n_steps_equil = 10_000 # 5 ps
    n_steps_loss = 2_000 # 1 ps
    n_steps_trunc = 200
    use_barostat = true
    clip_norm_threshold = T(Inf)
    n_threads = Threads.nthreads()
    n_epochs = 10_000
    learning_rate = T(1e-3)
    py_opt = pymod_torch.optim.Adam(py_functions.nn_model.parameters(), lr=learning_rate)

    if isdir(out_dir) && isfile("$out_dir/train.log")
        py_functions.nn_model.load_state_dict(pymod_torch.load("$out_dir/model.pt"))
        py_opt.load_state_dict(pymod_torch.load("$out_dir/optim.pt"))
        params = NNParams(py_functions.nn_params())
        start_epoch_n = countlines("$out_dir/train.log") + 1
    else
        isdir(out_dir) || mkdir(out_dir)
        params = NNParams(py_nn_params_start)
        start_epoch_n = 1
    end

    for epoch_n in start_epoch_n:n_epochs
        noise_seeds = [rand_seed() for i in 1:n_steps]
        noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]

        velocities_equil = [random_velocity(m, temp) for m in atom_masses]
        sys_equil, _, _ = forward_sim(nothing, velocities_equil, nothing, noise_equil_seeds, γ,
                                    n_steps_equil, n_steps_equil, params, dt,
                                    n_threads, inter_type, loss_type, false, false, use_barostat)
        coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)
        boundary_start = deepcopy(sys_equil.boundary)

        if reweighting
            sys_reverse, loss_rev, grads_reverse, l_rdf, l_ev = reweighting_sim(
                    coords_start, velocities_start, boundary_start, noise_seeds, γ, n_steps, n_steps_loss,
                    params, dt, n_threads, inter_type, loss_type)
        else
            sys_forward, _, _ = forward_sim(
                    coords_start, velocities_start, boundary_start, noise_seeds, γ, n_steps, n_steps_loss,
                    params, dt, n_threads, inter_type, loss_type, false, false)

            sys_reverse, loss_rev, grads_reverse, l_rdf, l_ev = reverse_sim(
                    sys_forward.coords, sys_forward.velocities, boundary_start, noise_seeds, γ, n_steps,
                    n_steps_loss, n_steps_trunc, params, dt,
                    n_threads, sys_forward.loggers, clip_norm_threshold, inter_type, loss_type)
        end

        open("$out_dir/train.log", "a") do of
            println(of, "$epoch_n $loss_rev $l_rdf $l_ev")
        end

        if epoch_n > 1
            rm("$out_dir/model_ep_$(epoch_n-1).pt")
            rm("$out_dir/optim_ep_$(epoch_n-1).pt")
        end
        pymod_torch.save(py_functions.nn_model.state_dict(), "$out_dir/model_ep_$epoch_n.pt")
        pymod_torch.save(py_opt.state_dict(), "$out_dir/optim_ep_$epoch_n.pt")

        py_functions.optimiser_step(py_opt, grads_reverse.py_params)
        params = NNParams(py_functions.nn_params())

        pymod_torch.save(py_functions.nn_model.state_dict(), "$out_dir/model.pt")
        pymod_torch.save(py_opt.state_dict(), "$out_dir/optim.pt")
    end
end

if run_inter_type == "maceoff"
    train()
end

#=
# Benchmark
loss_type = :enth_vap
for rep in 1:3
    reset_timer!(to)
    forward_sim(coords_start, velocities_start, nothing, noise_seeds, γ, 100, 100, params, dt, n_threads, inter_type, loss_type, false, false)
    println()
    show(to)
    println()
end
for rep in 1:3
    reset_timer!(to)
    reverse_sim(sys_forward.coords, sys_forward.velocities, nothing, noise_seeds, γ, 100, 100, 100, params, dt, n_threads, sys_forward.loggers, T(Inf), inter_type, loss_type)
    println()
    show(to)
    println()
end
=#
