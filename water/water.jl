# Reversible simulation for 3-point water model
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT

using Molly
using Zygote
import Enzyme
using UnitfulAtomic
using ChainRulesCore
using Optimisers
using Optim
using FiniteDifferences
using LoopVectorization
using Polyester
using TimerOutputs
using LinearAlgebra
using Random
using Statistics
using Test

Enzyme.API.runtimeActivity!(true)

const out_dir        = (length(ARGS) >= 1 ? ARGS[1] : "train_water")
const run_inter_type = (length(ARGS) >= 2 ? ARGS[2] : "lj")
const run_n          = (length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1)

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
const dist_cutoff    = T(1.0) # nm
const dist_neighbors = T(1.2) # nm
const n_steps_nf = 20
const crf_const =  T((1 / (dist_cutoff^3)) * ((78.3 - 1) / (2 * 78.3 + 1)))
const n_steps_log = 1000
const O_mass = T(15.99943)
const H_mass = T(1.007947)
const atom_masses = repeat([O_mass, H_mass, H_mass]; outer=n_molecules)
const temp = T(295.15) # K
const two_pow_1o6 = T(2^(1/6))
const n_bonded_params = 4
const rdf_loss_weighting = T(1.0)
const to = TimerOutput()

coords_copy(sys, neighbors=nothing; kwargs...) = copy(sys.coords)
velocities_copy(sys, neighbors=nothing; kwargs...) = copy(sys.velocities)
CoordCopyLogger() = GeneralObservableLogger(coords_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)
VelocityCopyLogger() = GeneralObservableLogger(velocities_copy, Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)

# New interactions need to define an entry in generate_inter, inter_n_params, force_inter,
#   force_grads_inter!, inject_atom_grads, extract_grads, Molly.force, Molly.potential_energy,
#   a struct, Molly.use_neighbors, Base.:+ and possibly an atom type with zero and + functions

struct DoubleExponential{T, C} <: PairwiseInteraction
    α::T
    β::T
    cutoff::C
end

Molly.use_neighbors(::DoubleExponential) = true

function Base.:+(i1::DoubleExponential, i2::DoubleExponential)
    return DoubleExponential(i1.α + i2.α, i1.β + i2.β, i1.cutoff)
end

struct Buckingham{C} <: PairwiseInteraction
    cutoff::C
end

Molly.use_neighbors(::Buckingham) = true

function Base.:+(i1::Buckingham, i2::Buckingham)
    return Buckingham(i1.cutoff)
end

struct BuckinghamAtom{T}
    index::Int
    charge::T
    mass::T
    A::T
    B::T
    C::T
end

function Base.zero(::Type{BuckinghamAtom{T}}) where T
    z = zero(T)
    return BuckinghamAtom(0, z, z, z, z, z)
end

function Base.:+(x::BuckinghamAtom, y::BuckinghamAtom)
    return BuckinghamAtom(0, x.charge + y.charge, x.mass + y.mass, x.A + y.A, x.B + y.B, x.C + y.C)
end

function ChainRulesCore.rrule(T::Type{<:BuckinghamAtom}, vs...)
    Y = T(vs...)
    function BuckinghamAtom_pullback(Ȳ)
        return NoTangent(), Ȳ.index, Ȳ.charge, Ȳ.mass, Ȳ.A, Ȳ.B, Ȳ.C
    end
    return Y, BuckinghamAtom_pullback
end

struct Buffered147{T, C} <: PairwiseInteraction
    δ::T
    γ::T
    cutoff::C
end

Molly.use_neighbors(::Buffered147) = true

function Base.:+(i1::Buffered147, i2::Buffered147)
    return Buffered147(i1.δ + i2.δ, i1.γ + i2.γ, i1.cutoff)
end

function generate_inter(inter_type, params)
    # Needs to be Zygote-friendly
    cutoff = DistanceCutoff(dist_cutoff)
    if inter_type == :lj
        # Params are σ, ϵ
        return LennardJones{false, typeof(cutoff), Int, Int, typeof(NoUnits), typeof(NoUnits)}(
                    cutoff, true, true, 1, 1, NoUnits, NoUnits)
    elseif inter_type == :dexp
        # Params are σ, ϵ, α, β
        α, β = params[3], params[4]
        return DoubleExponential(α, β, cutoff)
    elseif inter_type == :buck
        # Params are A, B, C
        return Buckingham(cutoff)
    elseif inter_type == :ljsc
        # Params are σ, ϵ, α, λ
        α, λ = params[3], params[4]
        p = 2
        σ6_fac = α * λ^p
        return LennardJonesSoftCore{false, typeof(cutoff), T, T, typeof(p), T, Int, Int,
                                    typeof(NoUnits), typeof(NoUnits)}(
                    cutoff, α, λ, p, σ6_fac, true, true, 1, 1, NoUnits, NoUnits)
    elseif inter_type == :buff
        # Params are σ, ϵ, δ, γ
        δ, γ = params[3], params[4]
        return Buffered147(δ, γ, cutoff)
    else
        throw(ArgumentError("inter_type not recognised"))
    end
end

inter_n_params(::CoulombReactionField) = 1
inter_n_params(::LennardJones) = 2
inter_n_params(::DoubleExponential) = 4
inter_n_params(::Buckingham) = 3
inter_n_params(::LennardJonesSoftCore) = 4
inter_n_params(::Buffered147) = 4

# Cutoff already applied
function force_inter(::LennardJones, atom_i, atom_j, r, invr2)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        six_term = (σ^2 * invr2)^3
        twelve_term = six_term^2
        return (24ϵ / r) * (2 * twelve_term - six_term)
    end
end

function force_grads_inter!(dF_dp_threads, ::LennardJones, atom_i, atom_j, r,
                            invr2, dr, ndr, i3, j3, chunk_i)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T), zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        six_term = (σ^2 * invr2)^3
        twelve_term = six_term^2
        F = (24ϵ / r) * (2 * twelve_term - six_term)
        f_dFi_dxjn_norm = -24ϵ * invr2 * (26 * twelve_term - 7 * six_term)
        f_dF_dσ = (144ϵ * invr2 / σ) * (4 * twelve_term - six_term) * dr
        f_dF_dϵ = F * ndr / ϵ
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui]
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui]
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui]
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui]
        end
        return F, f_dFi_dxjn_norm
    end
end

function force_inter(inter::DoubleExponential, atom_i, atom_j, r, invr2)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        α, β = inter.α, inter.β
        k1 = β * exp(α) / (α - β)
        k2 = α * exp(β) / (α - β)
        f = ϵ * (α * k1 * exp(-α * r / rm) - β * k2 * exp(-β * r / rm)) / rm
        return f
    end
end

function force_grads_inter!(dF_dp_threads, inter::DoubleExponential, atom_i, atom_j, r,
                            invr2, dr, ndr, i3, j3, chunk_i)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T), zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        α, β = inter.α, inter.β
        exp_1 = exp(-α * r / rm) * exp(α) / (α - β)
        exp_2 = exp(-β * r / rm) * exp(β) / (α - β)
        ϵαβ = ϵ * α * β
        F = ϵαβ * (exp_1 - exp_2) / rm
        f_dFi_dxjn_norm = -ϵαβ * (α * exp_1 - β * exp_2) / rm^2
        f_dF_drm = ϵαβ * ndr * ((α * r - rm) * exp_1 - (β * r - rm) * exp_2) / rm^3
        f_dF_dσ = f_dF_drm * two_pow_1o6
        f_dF_dϵ = F * ndr / ϵ
        f_dF_dα = -ϵ * β * ndr * ((α * (α - β) * (r - rm) + β * rm) * exp_1 - (β * rm) * exp_2) / (rm^2 * (α - β))
        f_dF_dβ = ϵ * α * ndr * ((α * rm) * exp_1 - (β * (β - α) * (r - rm) + α * rm) * exp_2) / (rm^2 * (α - β))
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui]
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui]
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui]
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui]
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dα[ui]
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dα[ui]
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dβ[ui]
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dβ[ui]
        end
        return F, f_dFi_dxjn_norm
    end
end

function force_inter(::Buckingham, atom_i, atom_j, r, invr2)
    if iszero(atom_i.A) || iszero(atom_j.A)
        return zero(T)
    else
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        return A * B * exp(-B * r) - 6 * C * invr2^3 / r
    end
end

function force_grads_inter!(dF_dp_threads, ::Buckingham, atom_i, atom_j, r,
                            invr2, dr, ndr, i3, j3, chunk_i)
    if iszero(atom_i.A) || iszero(atom_j.A)
        return zero(T), zero(T)
    else
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        invr7 = invr2^3 / r
        exp_mBr = exp(-B * r)
        F = A * B * exp_mBr - 6 * C * invr7
        f_dFi_dxjn_norm = -A * B^2 * exp_mBr + 42 * C * invr7 / r
        f_dF_dA = B * exp_mBr * ndr
        f_dF_dB = A * (1 - B * r) * exp_mBr * ndr
        f_dF_dC = -6 * invr7 * ndr
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dA[ui]
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dA[ui]
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dB[ui]
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dB[ui]
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dC[ui]
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dC[ui]
        end
        return F, f_dFi_dxjn_norm
    end
end

function force_inter(inter::LennardJonesSoftCore, atom_i, atom_j, r, invr2)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        σ6 = σ^6
        inv_rsc6 = inv(r^6 + σ6 * inter.σ6_fac)
        inv_rsc = √cbrt(inv_rsc6)
        six_term = σ6 * inv_rsc6
        return (24ϵ * inv_rsc) * (2 * six_term^2 - six_term) * (r * inv_rsc)^5
    end
end

function force_grads_inter!(dF_dp_threads, inter::LennardJonesSoftCore, atom_i, atom_j, r,
                            invr2, dr, ndr, i3, j3, chunk_i)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T), zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        α, λ, p, σ6_fac = inter.α, inter.λ, inter.p, inter.σ6_fac
        σ5 = σ^5
        σ6 = σ5 * σ
        σ12 = σ6^2
        r4 = r^4
        r5 = r4 * r
        r6 = r5 * r
        rsc6  = r6 + σ6 * σ6_fac
        rsc6m = r6 - σ6 * σ6_fac
        rsc12 = rsc6^2
        rsc24 = rsc12^2
        inv_rsc6 = inv(rsc6)
        inv_rsc = √cbrt(inv_rsc6)
        six_term = σ6 * inv_rsc6
        F = (24ϵ * inv_rsc) * (2 * six_term^2 - six_term) * (r * inv_rsc)^5
        f_dFi_dxjn_norm = (24ϵ * σ6 * r4) * (rsc12 + 6 * rsc6 * rsc6m - 36 * r6 * σ6 + 10 * σ6 * rsc6) / rsc24
        f_dF_dσ = -144ϵ * r5 * σ5 * (rsc6 * rsc6m + 2 * α * σ12 * λ^p - 4 * r6 * σ6) * ndr / rsc24
        f_dF_dϵ = F * ndr / ϵ
        f_dF_dα = 48ϵ * r5 * σ12 * λ^p * (rsc6 - 3 * σ6) * ndr / rsc24
        f_dF_dλ = 48ϵ * r5 * σ12 * α * p * λ^(p - 1) * (rsc6 - 3 * σ6) * ndr / rsc24
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui]
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui]
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui]
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui]
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dα[ui]
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dα[ui]
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dλ[ui]
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dλ[ui]
        end
        return F, f_dFi_dxjn_norm
    end
end

function force_inter(inter::Buffered147, atom_i, atom_j, r, invr2)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        δ, γ = inter.δ, inter.γ
        r_div_rm = r / rm
        r_div_rm_6 = r_div_rm^6
        r_div_rm_7 = r_div_rm_6 * r_div_rm
        γ7_term = (1 + γ) / (r_div_rm_7 + γ)
        f = (7ϵ / rm) * ((1 + δ) / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
        return f
    end
end

function force_grads_inter!(dF_dp_threads, inter::Buffered147, atom_i, atom_j, r,
                            invr2, dr, ndr, i3, j3, chunk_i)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T), zero(T)
    else
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        δ, γ = inter.δ, inter.γ
        γ2, δ2 = γ^2, δ^2
        r5 = r^5
        r6 = r5 * r
        r7 = r6 * r
        r8 = r7 * r
        r12 = r6 * r6
        r13 = r12 * r
        r14 = r13 * r
        r15 = r14 * r
        r21 = r14 * r7
        rm6 = rm^6
        rm7 = rm6 * rm
        rm8 = rm7 * rm
        rm9 = rm8 * rm
        rm14 = rm7 * rm7
        rm15 = rm14 * rm
        rm16 = rm15 * rm
        rm21 = rm14 * rm7
        r_div_rm = r / rm
        r_div_rm_6 = r6 / rm6
        r_div_rm_7 = r7 / rm7
        δp1 = δ + 1
        δp1_6 = δp1^6
        δp1_7 = δp1_6 * δp1
        γp1, γm1 = γ + 1, γ - 1
        γ7_term = γp1 / (r_div_rm_7 + γ)
        F = (7ϵ / rm) * (δp1 / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
        f_dFi_dxjn_norm = (14ϵ * δp1_7 * rm7 * (3 * δ2 * γ * γp1 * rm16 * r5 - 4 * δ2 * γp1 * rm9 * r12 - δ * γ * γp1 * rm15 * r6 - 15 * δ * γp1 * rm8 * r13 + 4 * γm1 * γ2 * rm21 + 12 * γ * γm1 * rm14 * r7 + 3 * (3γ - 5) * rm7 * r14 + 8 * r21)) / ((r + δ * rm)^9 + (γ * rm7 + r7)^3)
        f_dF_drm = ndr * -7ϵ * δp1_7 * rm6 * (7 * δ2 * γp1 * rm9 * r6 * (γ * rm7 - r7) + δ * rm * (-γm1 * γ2 * rm21 + 2 * (1 - 2γ) * γ * rm14 * r7 - 3 * (11γ + 9) * rm7 * r14 - 2 * r21) + 7r * (γm1 * γ2 * rm21 + 3 * γm1 * γ * rm14 * r7 + 2 * (γ - 2) * rm7 * r14 + 2r21)) / ((r + δ * rm)^9 * (r7 + γ * rm7)^3)
        f_dF_dσ = f_dF_drm * two_pow_1o6
        f_dF_dϵ = F * ndr / ϵ
        f_dF_dδ = ndr * 7ϵ * δp1_6 * rm7 * ((δ + 8) * γm1 * γ * rm15 - 7δ * γp1 * rm9 * r6 + rm8 * r7 * (2δ * (5γ + 3) + 17γ - 15) + 2 * (δ + 8) * rm * r14 - 7 * γm1 * γ * rm14 * r - 14 * γm1 * rm7 * r8 - 14r15) / ((r + δ * rm)^9 * (r7 + γ * rm7)^2)
        f_dF_dγ = ndr * -7ϵ * δp1_7 * rm14 * (δ * rm8 * r6 * (γ + 2) - δ * rm * r13 + γ * rm14 + 3 * rm7 * r7 - 2 * r14) / ((r + δ * rm)^8 * (r7 + γ * rm7)^3)
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui]
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui]
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui]
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui]
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dδ[ui]
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dδ[ui]
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dγ[ui]
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dγ[ui]
        end
        return F, f_dFi_dxjn_norm
    end
end

# @inline required here and below for gradient accuracy, possibly an Enzyme bug
@inline function Molly.force(inter::DoubleExponential, dr, coord_i, coord_j,
                             atom_i, atom_j, boundary)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(coord_i)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        r = sqrt(r2)
        α, β = inter.α, inter.β
        k1 = β * exp(α) / (α - β)
        k2 = α * exp(β) / (α - β)
        f = ϵ * (α * k1 * exp(-α * r / rm) - β * k2 * exp(-β * r / rm)) / rm
        return f * dr / r
    else
        return zero(coord_i)
    end
end

@inline function Molly.potential_energy(inter::DoubleExponential, dr, coord_i, coord_j,
                                        atom_i, atom_j, boundary)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        r = sqrt(r2)
        α, β = inter.α, inter.β
        k1 = β * exp(α) / (α - β)
        k2 = α * exp(β) / (α - β)
        pe = ϵ * (k1 * exp(-α * r / rm) - k2 * exp(-β * r / rm))
        return pe
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::Buckingham, dr, coord_i, coord_j,
                             atom_i, atom_j, boundary)
    if iszero(atom_i.A) || iszero(atom_j.A)
        return zero(coord_i)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        r = sqrt(r2)
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        return (A * B * exp(-B * r) / r - 6 * C / r2^4) * dr
    else
        return zero(coord_i)
    end
end

@inline function Molly.potential_energy(inter::Buckingham, dr, coord_i, coord_j,
                                        atom_i, atom_j, boundary)
    if iszero(atom_i.A) || iszero(atom_j.A)
        return zero(T)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        r = sqrt(r2)
        A = (atom_i.A + atom_j.A) / 2
        B = (atom_i.B + atom_j.B) / 2
        C = (atom_i.C + atom_j.C) / 2
        return A * exp(-B * r) - C / r2^3
    else
        return zero(T)
    end
end

@inline function Molly.force(inter::Buffered147, dr, coord_i, coord_j,
                             atom_i, atom_j, boundary)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(coord_i)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        δ, γ = inter.δ, inter.γ
        r = sqrt(r2)
        r_div_rm = r / rm
        r_div_rm_6 = r_div_rm^6
        r_div_rm_7 = r_div_rm_6 * r_div_rm
        γ7_term = (1 + γ) / (r_div_rm_7 + γ)
        f = (7ϵ / rm) * ((1 + δ) / (r_div_rm + δ))^7 * (inv(r_div_rm + δ) * (γ7_term - 2) + inv(r_div_rm_7 + γ) * r_div_rm_6 * γ7_term)
        return f * dr / r
    else
        return zero(coord_i)
    end
end

@inline function Molly.potential_energy(inter::Buffered147, dr, coord_i, coord_j,
                                        atom_i, atom_j, boundary)
    if iszero(atom_i.ϵ) || iszero(atom_j.ϵ)
        return zero(T)
    end
    r2 = sum(abs2, dr)
    if r2 <= inter.cutoff.sqdist_cutoff
        σ = (atom_i.σ + atom_j.σ) / 2
        rm = σ * two_pow_1o6
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
        δ, γ = inter.δ, inter.γ
        r_div_rm = sqrt(r2) / rm
        pe = ϵ * ((1 + δ) / (r_div_rm + δ))^7 * (((1 + γ) / (r_div_rm^7 + γ)) - 2)
        return pe
    else
        return zero(T)
    end
end

function inject_si_grads(sis, params)
    bonds = InteractionList2Atoms(
        sis[1].is,
        sis[1].js,
        [HarmonicBond(params[end - 3], params[end - 2]) for _ in eachindex(sis[1].inters)],
        sis[1].types,
    )
    angles = InteractionList3Atoms(
        sis[2].is,
        sis[2].js,
        sis[2].ks,
        [HarmonicAngle(params[end - 1], params[end]) for _ in eachindex(sis[2].inters)],
        sis[2].types,
    )
    return (bonds, angles)
end

# Needs to be Zygote-friendly
function inject_atom_grads(::Union{LennardJones, DoubleExponential, LennardJonesSoftCore, Buffered147},
                           params, n_atoms_sys)
    σ, ϵ, O_charge = params[1], params[2], params[end - n_bonded_params]
    H_charge = -O_charge / 2
    return map(1:n_atoms_sys) do i
        if i % 3 == 1
            Atom(i, O_charge, O_mass, σ, ϵ, false)
        else
            Atom(i, H_charge, H_mass, zero(T), zero(T), false)
        end
    end
end

function inject_atom_grads(::Buckingham, params, n_atoms_sys)
    A, B, C, O_charge = params[1], params[2], params[3], params[end - n_bonded_params]
    H_charge = -O_charge / 2
    return map(1:n_atoms_sys) do i
        if i % 3 == 1
            BuckinghamAtom(i, O_charge, O_mass, A, B, C)
        else
            BuckinghamAtom(i, H_charge, H_mass, zero(T), zero(T), zero(T))
        end
    end
end

function extract_grads(::LennardJones, sys_grads, sys)
    return [
        sum(at -> at.σ, sys_grads.atoms),
        sum(at -> at.ϵ, sys_grads.atoms),
    ]
end

function extract_grads(inter_grads::DoubleExponential, sys_grads, sys)
    return [
        sum(at -> at.σ, sys_grads.atoms),
        sum(at -> at.ϵ, sys_grads.atoms),
        inter_grads.α,
        inter_grads.β,
    ]
end

function extract_grads(::Buckingham, sys_grads, sys)
    return [
        sum(at -> at.A, sys_grads.atoms),
        sum(at -> at.B, sys_grads.atoms),
        sum(at -> at.C, sys_grads.atoms),
    ]
end

function extract_grads(inter_grads::LennardJonesSoftCore, sys_grads, sys)
    α_g, λ_g, p_g, σ6_fac_g = inter_grads.α, inter_grads.λ, inter_grads.p, inter_grads.σ6_fac
    inter = sys.pairwise_inters[1]
    α, λ, p, σ6_fac = inter.α, inter.λ, inter.p, inter.σ6_fac
    return [
        sum(at -> at.σ, sys_grads.atoms),
        sum(at -> at.ϵ, sys_grads.atoms),
        α_g + σ6_fac_g * λ^p,
        λ_g + σ6_fac_g * α * p * λ^(p - 1),
    ]
end

function extract_grads(inter_grads::Buffered147, sys_grads, sys)
    return [
        sum(at -> at.σ, sys_grads.atoms),
        sum(at -> at.ϵ, sys_grads.atoms),
        inter_grads.δ,
        inter_grads.γ,
    ]
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
    mean_U_gas = T(3.69)
    # n_threads=1 to avoid possible Enzyme issue
    snapshot_U_liquid = potential_energy(sys, find_neighbors(sys); n_threads=1) / n_molecules
    ΔH_vap = mean_U_gas - snapshot_U_liquid + RT
    return ΔH_vap
end

function loss_enth_vap(sys)
    ΔH_vap = enthalpy_vaporization(sys)
    # See Glattli2002 and https://www.engineeringtoolbox.com/water-properties-d_1573.html
    # 44.12 kJ/mol at 295.15 K plus amount to account for not using bond/angle constraints
    ΔH_vap_exp = T(44.12 + 2.8)
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
    if isnothing(sys_grads.atoms)
        dl_dp = zeros(T, sum(inter_n_params, sys.pairwise_inters) + n_bonded_params)
    else
        dl_dp = extract_grads(sys_grads.pairwise_inters[1], sys_grads, sys)
        push!(dl_dp, sum(ad.element == "O" ? at.charge : (-at.charge / 2)
                         for (at, ad) in zip(sys_grads.atoms, sys.atoms_data)))
        push!(dl_dp, sum(i -> i.k , sys_grads.specific_inter_lists[1].inters))
        push!(dl_dp, sum(i -> i.r0, sys_grads.specific_inter_lists[1].inters))
        push!(dl_dp, sum(i -> i.k , sys_grads.specific_inter_lists[2].inters))
        push!(dl_dp, sum(i -> i.θ0, sys_grads.specific_inter_lists[2].inters))
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

function calc_F!(forces_flat, forces_flat_threads, coords, boundary, atoms, pis, sis, neighbors,
                 n_threads)
    @timeit to "zeroing" begin
        @assert n_threads == Threads.nthreads()
        @batch per=thread for ti in 1:n_threads
            @inbounds for i in axes(forces_flat_threads, 1)
                forces_flat_threads[i, ti] = zero(T)
            end
        end
        inter_1, coul = pis
        bonds, angles = sis
        # Assume the same cutoff distance for both interactions
        sqdist_cutoff = inter_1.cutoff.sqdist_cutoff
    end

    @timeit to "non-bonded" @batch per=thread for ni in eachindex(neighbors)
        chunk_i = Threads.threadid()
        @inbounds i, j, _ = neighbors[ni]
        @inbounds dr = vector(coords[i], coords[j], boundary)
        r2 = sum(abs2, dr)
        if r2 <= sqdist_cutoff
            @inbounds atom_i, atom_j = atoms[i], atoms[j]
            r = sqrt(r2)
            invr2 = inv(r2)
            ndr = dr / r
            charge_prefac = coul.coulomb_const * atom_i.charge * atom_j.charge
            F = charge_prefac * (invr2 - 2 * crf_const * r)
            F += force_inter(inter_1, atom_i, atom_j, r, invr2)
            i3 = 3 * (i - 1)
            j3 = 3 * (j - 1)

            @turbo for ui in 1:3
                forces_flat_threads[i3 + ui, chunk_i] -= F * ndr[ui]
                forces_flat_threads[j3 + ui, chunk_i] += F * ndr[ui]
            end
        end
    end

    @timeit to "bonds" begin
        @inbounds begin
            @batch per=thread for bi in eachindex(bonds.is)
                chunk_i = Threads.threadid()
                i, j, bond = bonds.is[bi], bonds.js[bi], bonds.inters[bi]
                dr = vector(coords[i], coords[j], boundary)
                r2 = sum(abs2, dr)
                r = sqrt(r2)
                ndr = dr / r
                F = -bond.k * (r - bond.r0)
                i3 = 3 * (i - 1)
                j3 = 3 * (j - 1)

                for ui in 1:3
                    forces_flat_threads[i3 + ui, chunk_i] -= F * ndr[ui]
                    forces_flat_threads[j3 + ui, chunk_i] += F * ndr[ui]
                end
            end
        end
    end

    @timeit to "angles" begin
        @inbounds begin
            @batch per=thread for ai in eachindex(angles.is)
                chunk_i = Threads.threadid()
                i, j, k, ang = angles.is[ai], angles.js[ai], angles.ks[ai], angles.inters[ai]
                dr_ji = vector(coords[j], coords[i], boundary)
                dr_jk = vector(coords[j], coords[k], boundary)
                r2_ji, r2_jk = sum(abs2, dr_ji), sum(abs2, dr_jk)
                r_ji, r_jk = sqrt(r2_ji), sqrt(r2_jk)
                cross_ji_jk = dr_ji × dr_jk
                iszero(cross_ji_jk) && continue
                p1_unnorm =  dr_ji × cross_ji_jk
                p3_unnorm = -dr_jk × cross_ji_jk
                p1_norm2, p3_norm2 = sum(abs2, p1_unnorm), sum(abs2, p3_unnorm)
                p1_norm, p3_norm = sqrt(p1_norm2), sqrt(p3_norm2)
                p1, p3 = p1_unnorm / p1_norm, p3_unnorm / p3_norm
                p1_divnorm, p3_divnorm = p1 / r_ji, p3 / r_jk
                dot_ji_jk = dot(dr_ji, dr_jk)
                dot_prod_norm = dot_ji_jk / (r_ji * r_jk)
                θ = acos(dot_prod_norm)
                F = -ang.k * (θ - ang.θ0)
                i3 = 3 * (i - 1)
                j3 = 3 * (j - 1)
                k3 = 3 * (k - 1)

                for ui in 1:3
                    forces_flat_threads[i3 + ui, chunk_i] += F * p1_divnorm[ui]
                    forces_flat_threads[k3 + ui, chunk_i] += F * p3_divnorm[ui]
                    forces_flat_threads[j3 + ui, chunk_i] -= F * (p1_divnorm[ui] + p3_divnorm[ui])
                end
            end
        end
    end

    @timeit to "nb reduction" @batch per=thread for i in eachindex(forces_flat)
        @inbounds forces_flat[i] = sum(@view forces_flat_threads[i, :])
    end

    return forces_flat
end

function forward_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, params,
                     dt, n_threads, inter_type, loss_type, run_grads=true, calc_loss=true,
                     use_barostat=false)
    sys_init = System(
        structure_file,
        ff;
        loggers=(
            coords=CoordCopyLogger(),
            velocities=VelocityCopyLogger(),
        ),
        units=false,
        dist_cutoff=dist_cutoff,
        dist_neighbors=dist_neighbors,
    )
    sys_init.neighbor_finder.n_steps = n_steps_nf
    sys_init.coords .= wrap_coords.(sys_init.coords, (sys_init.boundary,))
    pis = (
        generate_inter(inter_type, params),
        sys_init.pairwise_inters[2],
    )
    sis = inject_si_grads(sys_init.specific_inter_lists, params)
    atoms = inject_atom_grads(pis[1], params, n_atoms)
    sys = System(
        deepcopy(sys_init);
        atoms=atoms,
        coords=copy(isnothing(coords) ? sys_init.coords : coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        pairwise_inters=pis,
        specific_inter_lists=sis,
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

    forces_flat = zeros(T, n_atoms_x3)
    forces_flat_threads = zeros(T, n_atoms_x3, n_threads)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    loss_accum = zero(T)
    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "f.1" calc_F!(forces_flat, forces_flat_threads, sys.coords, sys.boundary,
                sys.atoms, sys.pairwise_inters, sys.specific_inter_lists, neighbors, n_threads)
        @timeit to "f.2" sys.velocities .+= reinterpret(SVector{3, T}, forces_flat) .* dt ./ masses(sys)
        @timeit to "f.3" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.4" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.5" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.6" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.7" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "neighbour finding" neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
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
            pis = (
                generate_inter(inter_type, params),
                sys_init.pairwise_inters[2],
            )
            sis = inject_si_grads(sys_init.specific_inter_lists, params)
            atoms_grad = inject_atom_grads(pis[1], params, n_atoms)
            sys_grad = System(
                deepcopy(sys_init);
                atoms=atoms_grad,
                coords=copy(isnothing(coords) ? sys_init.coords : coords),
                boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
                velocities=copy(velocities),
                pairwise_inters=pis,
                specific_inter_lists=sis,
            )
            loss_accum_rad = zero(T)
            neighbors_grad = find_neighbors(sys_grad, sys_grad.neighbor_finder; n_threads=n_threads)
            for step_n in 1:n_steps
                # n_threads=1 to avoid possible Enzyme issue
                accels = accelerations(sys_grad, neighbors_grad; n_threads=1)
                sys_grad.velocities += accels .* dt
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.velocities = sys_grad.velocities .* vel_scale .+ noises[step_n] .* noise_scale
                sys_grad.coords += sys_grad.velocities .* dt ./ 2
                sys_grad.coords = wrap_coords.(sys_grad.coords, (sys_grad.boundary,))
                neighbors_grad = find_neighbors(sys_grad, sys_grad.neighbor_finder, neighbors_grad,
                                                step_n; n_threads=n_threads)
                if step_n % n_steps_loss == 0
                    loss_accum_rad += loss_fn(sys_grad, loss_type)[1]
                end
            end
            loss_rad = loss_accum_rad / n_blocks
            return loss_rad
        end

        @timeit to "forward grad" begin
            loss_rad, grads = withgradient(loss_forward, params)
        end
        @assert loss_rad ≈ loss
        println("fwd loss ", loss_rad)
        println("fwd gradients ", grads[1])
    else
        grads = fill(nothing, sum(inter_n_params, sys.pairwise_inters) + n_bonded_params)
    end

    return sys, loss, grads[1]
end

function reweighting_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, params,
                         dt, n_threads, inter_type, loss_type, use_barostat=false)
    sys_init = System(
        structure_file,
        ff;
        loggers=(
            coords=CoordCopyLogger(),
            velocities=VelocityCopyLogger(),
        ),
        units=false,
        dist_cutoff=dist_cutoff,
        dist_neighbors=dist_neighbors,
    )
    sys_init.neighbor_finder.n_steps = n_steps_nf
    sys_init.coords .= wrap_coords.(sys_init.coords, (sys_init.boundary,))
    pis = (
        generate_inter(inter_type, params),
        sys_init.pairwise_inters[2],
    )
    sis = inject_si_grads(sys_init.specific_inter_lists, params)
    atoms = inject_atom_grads(pis[1], params, n_atoms)
    sys = System(
        deepcopy(sys_init);
        atoms=atoms,
        coords=copy(isnothing(coords) ? sys_init.coords : coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        pairwise_inters=pis,
        specific_inter_lists=sis,
    )
    n_blocks = (n_steps == 0 ? 1 : n_steps ÷ n_steps_loss)
    n_params = sum(inter_n_params, sys.pairwise_inters) + n_bonded_params
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

    forces_flat = zeros(T, n_atoms_x3)
    forces_flat_threads = zeros(T, n_atoms_x3, n_threads)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    loss_accum     = zero(T)
    loss_accum_rdf = zero(T)
    loss_accum_ev  = zero(T)
    dl_dp_accum    = zeros(T, n_params)
    dE_dp_accum    = zeros(T, n_params)
    l_dE_dp_accum  = zeros(T, n_params)

    @timeit to "forward loop" for step_n in 1:n_steps
        @timeit to "f.1" calc_F!(forces_flat, forces_flat_threads, sys.coords, sys.boundary,
                sys.atoms, sys.pairwise_inters, sys.specific_inter_lists, neighbors, n_threads)
        @timeit to "f.2" sys.velocities .+= reinterpret(SVector{3, T}, forces_flat) .* dt ./ masses(sys)
        @timeit to "f.3" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.4" @inbounds generate_noise!(noise, noise_seeds[step_n])
        @timeit to "f.5" sys.velocities .= sys.velocities .* vel_scale .+ noise .* noise_scale
        @timeit to "f.6" sys.coords .+= sys.velocities .* dt ./ 2
        @timeit to "f.7" sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        @timeit to "neighbour finding" neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
        @timeit to "f.8" run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
        @timeit to "f.9" if step_n % n_steps_loss == 0
            loss_block, loss_block_rdf, loss_block_ev, _, dl_dp_accum_block = loss_and_grads(sys, loss_type)
            loss_accum     += loss_block
            loss_accum_rdf += loss_block_rdf
            loss_accum_ev  += loss_block_ev
            dl_dp_accum .= dl_dp_accum .+ dl_dp_accum_block

            sys_grads = gradient(potential_energy, sys, neighbors)[1]
            dE_dp_block = extract_grads(sys_grads.pairwise_inters[1], sys_grads, sys)
            push!(dE_dp_block, sum(ad.element == "O" ? at.charge : (-at.charge / 2)
                                   for (at, ad) in zip(sys_grads.atoms, sys.atoms_data)))
            push!(dE_dp_block, sum(i -> i.k , sys_grads.specific_inter_lists[1].inters))
            push!(dE_dp_block, sum(i -> i.r0, sys_grads.specific_inter_lists[1].inters))
            push!(dE_dp_block, sum(i -> i.k , sys_grads.specific_inter_lists[2].inters))
            push!(dE_dp_block, sum(i -> i.θ0, sys_grads.specific_inter_lists[2].inters))

            dE_dp_accum .= dE_dp_accum .+ dE_dp_block
            l_dE_dp_accum .= l_dE_dp_accum .+ loss_block .* dE_dp_block
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
    grads = dl_dp_accum ./ n_blocks .- β .* (l_dE_dp_accum ./ n_blocks .- loss .* dE_dp_accum ./ n_blocks)

    return sys, loss, grads, l_rdf, l_ev
end

function calc_F_and_grads!(forces_flat, forces_flat_threads, dl_dfi, dl_dfi_threads, dF_dp, dF_dp_threads,
                           coords, boundary, atoms, pis, sis, neighbors, dl_dfi_prev, atom_masses_3N,
                           O_charge, n_params, n_threads)
    @timeit to "zeroing" begin
        @assert n_threads == Threads.nthreads()
        @batch per=thread for ti in 1:n_threads
            @inbounds for i in axes(forces_flat_threads, 1)
                forces_flat_threads[i, ti] = zero(T)
                dl_dfi_threads[i, ti] = zero(T)
                for j in axes(dF_dp_threads, 2)
                    dF_dp_threads[i, j, ti] = zero(T)
                end
            end
        end
        inter_1, coul = pis
        bonds, angles = sis
        # Assume the same cutoff distance for both interactions
        sqdist_cutoff = inter_1.cutoff.sqdist_cutoff
    end

    @timeit to "non-bonded" @batch per=thread for ni in eachindex(neighbors)
        chunk_i = Threads.threadid()
        @inbounds i, j, _ = neighbors[ni]
        @inbounds dr = vector(coords[i], coords[j], boundary)
        r2 = sum(abs2, dr)
        if r2 <= sqdist_cutoff
            @inbounds atom_i, atom_j = atoms[i], atoms[j]
            r = sqrt(r2)
            invr2 = inv(r2)
            invr3 = invr2 / r
            ndr = dr / r
            charge_prefac = coul.coulomb_const * atom_i.charge * atom_j.charge
            f_dFi_dxjn_norm = -2 * charge_prefac * (invr3 + crf_const)
            F = charge_prefac * (invr2 - 2 * crf_const * r)
            f_dF_dc = 2 * F * ndr / O_charge

            i3 = 3 * (i - 1)
            j3 = 3 * (j - 1)
            F_inter, f_dFi_dxjn_norm_inter = force_grads_inter!(dF_dp_threads, inter_1,
                    atom_i, atom_j, r, invr2, dr, ndr, i3, j3, chunk_i)
            F += F_inter
            f_dFi_dxjn_norm += f_dFi_dxjn_norm_inter
            f_dFi_dxjn = f_dFi_dxjn_norm * ndr
            F_div_r3 = F * invr3

            @turbo for ui in 1:3
                forces_flat_threads[i3 + ui, chunk_i] -= F * ndr[ui]
                forces_flat_threads[j3 + ui, chunk_i] += F * ndr[ui]
            end

            @turbo for ui in 1:3, uj in 1:3
                dr_term = (ui == uj ? dr[ui]^2 - r2 : dr[ui] * dr[uj])
                df_term = -f_dFi_dxjn[ui] * ndr[uj] + F_div_r3 * dr_term
                df_term_divi = df_term / atom_masses_3N[i3 + ui]
                df_term_divj = df_term / atom_masses_3N[j3 + ui]
                dl_term = dl_dfi_prev[i3 + ui] * df_term_divi - dl_dfi_prev[j3 + ui] * df_term_divj
                dl_dfi_threads[i3 + uj, chunk_i] -= dl_term
                dl_dfi_threads[j3 + uj, chunk_i] += dl_term
            end

            @turbo for ui in 1:3
                dF_dp_threads[i3 + ui, n_params - n_bonded_params, chunk_i] -= f_dF_dc[ui]
                dF_dp_threads[j3 + ui, n_params - n_bonded_params, chunk_i] += f_dF_dc[ui]
            end
        end
    end

    @timeit to "bonds" begin
        @inbounds begin
            @batch per=thread for bi in eachindex(bonds.is)
                chunk_i = Threads.threadid()
                i, j, bond = bonds.is[bi], bonds.js[bi], bonds.inters[bi]
                dr = vector(coords[i], coords[j], boundary)
                r2 = sum(abs2, dr)
                r = sqrt(r2)
                ndr = dr / r
                disp = r - bond.r0
                F = -bond.k * disp
                F_div_r3 = F / (r2 * r)
                f_dFi_dxjn = -bond.k * ndr
                i3 = 3 * (i - 1)
                j3 = 3 * (j - 1)

                for ui in 1:3
                    forces_flat_threads[i3 + ui, chunk_i] -= F * ndr[ui]
                    forces_flat_threads[j3 + ui, chunk_i] += F * ndr[ui]
                    dF_dp_threads[i3 + ui, n_params - 3, chunk_i] += disp * ndr[ui]
                    dF_dp_threads[j3 + ui, n_params - 3, chunk_i] -= disp * ndr[ui]
                    dF_dp_threads[i3 + ui, n_params - 2, chunk_i] += f_dFi_dxjn[ui]
                    dF_dp_threads[j3 + ui, n_params - 2, chunk_i] -= f_dFi_dxjn[ui]
                    for uj in 1:3
                        dr_term = (ui == uj ? dr[ui]^2 - r2 : dr[ui] * dr[uj])
                        df_term = -f_dFi_dxjn[ui] * ndr[uj] + F_div_r3 * dr_term
                        df_term_divi = df_term / atom_masses_3N[i3 + ui]
                        df_term_divj = df_term / atom_masses_3N[j3 + ui]
                        dl_term = dl_dfi_prev[i3 + ui] * df_term_divi - dl_dfi_prev[j3 + ui] * df_term_divj
                        dl_dfi_threads[i3 + uj, chunk_i] -= dl_term
                        dl_dfi_threads[j3 + uj, chunk_i] += dl_term
                    end
                end
            end
        end
    end

    @timeit to "angles" begin
        @inbounds begin
            @batch per=thread for ai in eachindex(angles.is)
                chunk_i = Threads.threadid()
                i, j, k, ang = angles.is[ai], angles.js[ai], angles.ks[ai], angles.inters[ai]
                dr_ji = vector(coords[j], coords[i], boundary)
                dr_jk = vector(coords[j], coords[k], boundary)
                r2_ji, r2_jk = sum(abs2, dr_ji), sum(abs2, dr_jk)
                r_ji, r_jk = sqrt(r2_ji), sqrt(r2_jk)
                r3_ji, r3_jk = r2_ji * r_ji, r2_jk * r_jk
                cross_ji_jk = dr_ji × dr_jk
                iszero(cross_ji_jk) && continue
                p1_unnorm =  dr_ji × cross_ji_jk
                p3_unnorm = -dr_jk × cross_ji_jk
                p1_norm2, p3_norm2 = sum(abs2, p1_unnorm), sum(abs2, p3_unnorm)
                p1_norm, p3_norm = sqrt(p1_norm2), sqrt(p3_norm2)
                p1, p3 = p1_unnorm / p1_norm, p3_unnorm / p3_norm
                p1_divnorm, p3_divnorm = p1 / r_ji, p3 / r_jk
                dot_ji_jk = dot(dr_ji, dr_jk)
                dot_prod_norm = dot_ji_jk / (r_ji * r_jk)
                θ = acos(dot_prod_norm)
                disp = ang.θ0 - θ
                F = ang.k * disp
                dF_ddpn = ang.k / sqrt(1 - dot_prod_norm^2)
                dF_dxi = dF_ddpn * (dr_jk * r2_ji - dr_ji * dot_ji_jk) / (r3_ji * r_jk)
                dF_dxk = dF_ddpn * (dr_ji * r2_jk - dr_jk * dot_ji_jk) / (r3_jk * r_ji)
                dF_dxj = -dF_dxi - dF_dxk
                i3 = 3 * (i - 1)
                j3 = 3 * (j - 1)
                k3 = 3 * (k - 1)

                for ui in 1:3
                    forces_flat_threads[i3 + ui, chunk_i] += F * p1_divnorm[ui]
                    forces_flat_threads[k3 + ui, chunk_i] += F * p3_divnorm[ui]
                    forces_flat_threads[j3 + ui, chunk_i] -= F * (p1_divnorm[ui] + p3_divnorm[ui])
                    dF_dp_threads[i3 + ui, n_params - 1, chunk_i] += disp * p1_divnorm[ui]
                    dF_dp_threads[k3 + ui, n_params - 1, chunk_i] += disp * p3_divnorm[ui]
                    dF_dp_threads[j3 + ui, n_params - 1, chunk_i] -= disp * (p1_divnorm[ui] + p3_divnorm[ui])
                    dF_dp_threads[i3 + ui, n_params    , chunk_i] += ang.k * p1_divnorm[ui]
                    dF_dp_threads[k3 + ui, n_params    , chunk_i] += ang.k * p3_divnorm[ui]
                    dF_dp_threads[j3 + ui, n_params    , chunk_i] -= ang.k * (p1_divnorm[ui] + p3_divnorm[ui])
                end

                for ui in 1:3, vi in 1:3
                    mass_i = atom_masses_3N[i3 + ui]
                    mass_j = atom_masses_3N[j3 + ui]
                    mass_k = atom_masses_3N[k3 + ui]

                    if ui == vi
                        uj = ui % 3 + 1
                        uk = (ui + 1) % 3 + 1

                        dp1udn_du1 = (dr_ji[uj]*dr_jk[uj] + dr_ji[uk]*dr_jk[uk]) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^2*r_ji^2)) * ((dr_ji[ui]*p1_norm/r_ji) + (r_ji / p1_norm) * (p1_unnorm[uj]*(dr_ji[uj]*dr_jk[ui] - 2*dr_ji[ui]*dr_jk[uj]) + p1_unnorm[uk]*(dr_ji[uk]*dr_jk[ui] - 2*dr_ji[ui]*dr_jk[uk]) + p1_unnorm[ui]*(dr_ji[uj]*dr_jk[uj] + dr_ji[uk]*dr_jk[uk])))
                        dp3udn_du1 = (-dr_jk[uj]^2 - dr_jk[uk]^2) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^3*r_jk)) * (p3_unnorm[uj]*dr_jk[ui]*dr_jk[uj] + p3_unnorm[uk]*dr_jk[ui]*dr_jk[uk] + p3_unnorm[ui]*(-dr_jk[uj]^2 - dr_jk[uk]^2))

                        gf1 = dF_dxi[ui] * p1_divnorm[ui] + F * dp1udn_du1
                        gf3 = dF_dxi[ui] * p3_divnorm[ui] + F * dp3udn_du1
                        dl_dfi_threads[i3 + ui, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j

                        dp3udn_du3 = (dr_ji[uj]*dr_jk[uj] + dr_ji[uk]*dr_jk[uk]) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^2*r_jk^2)) * ((dr_jk[ui]*p3_norm/r_jk) + (r_jk / p3_norm) * (p3_unnorm[uj]*(dr_jk[uj]*dr_ji[ui] - 2*dr_jk[ui]*dr_ji[uj]) + p3_unnorm[uk]*(dr_jk[uk]*dr_ji[ui] - 2*dr_jk[ui]*dr_ji[uk]) + p3_unnorm[ui]*(dr_jk[uj]*dr_ji[uj] + dr_jk[uk]*dr_ji[uk])))
                        dp1udn_du3 = (-dr_ji[uj]^2 - dr_ji[uk]^2) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^3*r_ji)) * (p1_unnorm[uj]*dr_ji[ui]*dr_ji[uj] + p1_unnorm[uk]*dr_ji[ui]*dr_ji[uk] + p1_unnorm[ui]*(-dr_ji[uj]^2 - dr_ji[uk]^2))

                        gf1 = dF_dxk[ui] * p1_divnorm[ui] + F * dp1udn_du3
                        gf3 = dF_dxk[ui] * p3_divnorm[ui] + F * dp3udn_du3
                        dl_dfi_threads[k3 + ui, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j

                        dp1udn_du2 = ((dr_ji[uj] - dr_jk[uj])*dr_ji[uj] + (dr_ji[uk] - dr_jk[uk])*dr_ji[uk]) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^2*r_ji^2)) * ((-dr_ji[ui]*p1_norm/r_ji) + (r_ji / p1_norm) * (p1_unnorm[uj]*(cross_ji_jk[uk] + dr_ji[ui]*dr_jk[uj] - dr_ji[ui]*dr_ji[uj]) + p1_unnorm[uk]*(-cross_ji_jk[uj] - dr_ji[ui]*dr_ji[uk] + dr_ji[ui]*dr_jk[uk]) + p1_unnorm[ui]*((dr_ji[uj] - dr_jk[uj])*dr_ji[uj] + (dr_ji[uk] - dr_jk[uk])*dr_ji[uk])))
                        dp3udn_du2 = ((dr_jk[uj] - dr_ji[uj])*dr_jk[uj] + (dr_jk[uk] - dr_ji[uk])*dr_jk[uk]) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^2*r_jk^2)) * ((-dr_jk[ui]*p3_norm/r_jk) + (r_jk / p3_norm) * (p3_unnorm[uj]*(-cross_ji_jk[uk] + dr_jk[ui]*dr_ji[uj] - dr_jk[ui]*dr_jk[uj]) + p3_unnorm[uk]*(cross_ji_jk[uj] - dr_jk[ui]*dr_jk[uk] + dr_jk[ui]*dr_ji[uk]) + p3_unnorm[ui]*((dr_jk[uj] - dr_ji[uj])*dr_jk[uj] + (dr_jk[uk] - dr_ji[uk])*dr_jk[uk])))

                        gf1 = dF_dxj[ui] * p1_divnorm[ui] + F * dp1udn_du2
                        gf3 = dF_dxj[ui] * p3_divnorm[ui] + F * dp3udn_du2
                        dl_dfi_threads[j3 + ui, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j
                    else
                        oi = 3 - (ui + vi) % 3

                        dp1udn_dv1 = (dr_ji[ui]*dr_jk[vi] - 2*dr_ji[vi]*dr_jk[ui]) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^2*r_ji^2)) * ((dr_ji[vi]*p1_norm/r_ji) + (r_ji / p1_norm) * (p1_unnorm[vi]*(dr_ji[ui]*dr_jk[ui] + dr_ji[oi]*dr_jk[oi]) + p1_unnorm[oi]*(dr_ji[oi]*dr_jk[vi] - 2*dr_ji[vi]*dr_jk[oi]) + p1_unnorm[ui]*(dr_ji[ui]*dr_jk[vi] - 2*dr_ji[vi]*dr_jk[ui])))
                        dp3udn_dv1 = (dr_ji[ui]*dr_ji[vi]) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^3*r_ji)) * (p1_unnorm[vi]*(-dr_ji[ui]^2 - dr_ji[oi]^2) + p1_unnorm[ui]*dr_ji[ui]*dr_ji[vi] + p1_unnorm[oi]*dr_ji[vi]*dr_ji[oi])

                        gf1 = dF_dxi[vi] * p1_divnorm[ui] + F * dp1udn_dv1
                        gf3 = dF_dxi[vi] * p3_divnorm[ui] + F * dp3udn_dv1
                        dl_dfi_threads[i3 + vi, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j

                        dp3udn_dv3 = (dr_jk[ui]*dr_ji[vi] - 2*dr_jk[vi]*dr_ji[ui]) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^2*r_jk^2)) * ((dr_jk[vi]*p3_norm/r_jk) + (r_jk / p3_norm) * (p3_unnorm[vi]*(dr_jk[ui]*dr_ji[ui] + dr_jk[oi]*dr_ji[oi]) + p3_unnorm[oi]*(dr_jk[oi]*dr_ji[vi] - 2*dr_jk[vi]*dr_ji[oi]) + p3_unnorm[ui]*(dr_jk[ui]*dr_ji[vi] - 2*dr_jk[vi]*dr_ji[ui])))
                        dp1udn_dv3 = (dr_jk[ui]*dr_jk[vi]) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^3*r_jk)) * (p3_unnorm[vi]*(-dr_jk[ui]^2 - dr_jk[oi]^2) + p3_unnorm[ui]*dr_jk[ui]*dr_jk[vi] + p3_unnorm[oi]*dr_jk[vi]*dr_jk[oi])

                        gf1 = dF_dxk[vi] * p1_divnorm[ui] + F * dp1udn_dv3
                        gf3 = dF_dxk[vi] * p3_divnorm[ui] + F * dp3udn_dv3
                        dl_dfi_threads[k3 + vi, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j

                        if vi > ui
                            direction = (vi - ui == 2 ? -1 : 1)
                            dp1udn_dv2 = (-direction*cross_ji_jk[oi] - dr_ji[ui]*dr_ji[vi] + dr_jk[ui]*dr_ji[vi]) / (p1_norm*r_ji) - (p1_unnorm[ui] / (p1_norm^2*r_ji^2)) * ((-dr_ji[vi]*p1_norm/r_ji) + (r_ji / p1_norm) * (p1_unnorm[vi]*((dr_ji[ui] - dr_jk[ui])*dr_ji[ui] + (dr_ji[oi] - dr_jk[oi])*dr_ji[oi]) + p1_unnorm[oi]*(direction*cross_ji_jk[ui] + dr_ji[vi]*dr_jk[oi] - dr_ji[vi]*dr_ji[oi]) + p1_unnorm[ui]*(-direction*cross_ji_jk[oi] - dr_ji[ui]*dr_ji[vi] + dr_jk[ui]*dr_ji[vi])))
                            dp3udn_dv2 = (direction*cross_ji_jk[oi] - dr_jk[ui]*dr_jk[vi] + dr_ji[ui]*dr_jk[vi]) / (p3_norm*r_jk) - (p3_unnorm[ui] / (p3_norm^2*r_jk^2)) * ((-dr_jk[vi]*p3_norm/r_jk) + (r_jk / p3_norm) * (p3_unnorm[vi]*((dr_jk[ui] - dr_ji[ui])*dr_jk[ui] + (dr_jk[oi] - dr_ji[oi])*dr_jk[oi]) + p3_unnorm[oi]*(-direction*cross_ji_jk[ui] + dr_jk[vi]*dr_ji[oi] - dr_jk[vi]*dr_jk[oi]) + p3_unnorm[ui]*(direction*cross_ji_jk[oi] - dr_jk[ui]*dr_jk[vi] + dr_ji[ui]*dr_jk[vi])))

                            gf1 = dF_dxj[vi] * p1_divnorm[ui] + F * dp1udn_dv2
                            gf3 = dF_dxj[vi] * p3_divnorm[ui] + F * dp3udn_dv2
                            dl_dfi_threads[j3 + vi, chunk_i] += dl_dfi_prev[i3 + ui] * gf1 / mass_i + dl_dfi_prev[k3 + ui] * gf3 / mass_k - dl_dfi_prev[j3 + ui] * (gf1 + gf3) / mass_j

                            gf1 = dF_dxj[ui] * p1_divnorm[vi] + F * dp1udn_dv2
                            gf3 = dF_dxj[ui] * p3_divnorm[vi] + F * dp3udn_dv2
                            dl_dfi_threads[j3 + ui, chunk_i] += dl_dfi_prev[i3 + vi] * gf1 / atom_masses_3N[i3 + vi] + dl_dfi_prev[k3 + vi] * gf3 / atom_masses_3N[k3 + vi] - dl_dfi_prev[j3 + vi] * (gf1 + gf3) / atom_masses_3N[j3 + vi]
                        end
                    end
                end
            end
        end
    end

    @timeit to "nb reduction" @batch per=thread for i in eachindex(forces_flat)
        @inbounds forces_flat[i] = sum(@view forces_flat_threads[i, :])
        @inbounds dl_dfi[i] = sum(@view dl_dfi_threads[i, :])
        @inbounds for j in axes(dF_dp, 2)
            dF_dp[i, j] = sum(@view dF_dp_threads[i, j, :])
        end
    end

    return forces_flat, dl_dfi, dF_dp
end

function reverse_sim(coords, velocities, boundary, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc,
                     params, dt, n_threads, fwd_loggers, clip_norm_threshold, inter_type, loss_type)
    if n_steps_trunc > n_steps_loss
        throw(ArgumentError("n_steps_trunc must not be greater than n_steps_loss"))
    end

    sys_init = System(
        structure_file,
        ff;
        units=false,
        dist_cutoff=dist_cutoff,
        dist_neighbors=dist_neighbors,
    )
    sys_init.neighbor_finder.n_steps = n_steps_nf
    pis = (
        generate_inter(inter_type, params),
        sys_init.pairwise_inters[2],
    )
    sis = inject_si_grads(sys_init.specific_inter_lists, params)
    atoms = inject_atom_grads(pis[1], params, n_atoms)
    sys = System(
        deepcopy(sys_init);
        atoms=atoms,
        coords=copy(coords),
        boundary=(isnothing(boundary) ? sys_init.boundary : boundary),
        velocities=copy(velocities),
        pairwise_inters=pis,
        specific_inter_lists=sis,
    )
    exp_mγdt = exp(-γ * dt)
    noise_scale = sqrt(1 - exp_mγdt^2)
    n_blocks = n_steps ÷ n_steps_loss
    n_params = sum(inter_n_params, sys.pairwise_inters) + n_bonded_params
    fwd_coords_logger = values(fwd_loggers.coords)
    fwd_velocities_logger = values(fwd_loggers.velocities)

    loss_accum, loss_accum_rdf, loss_accum_ev, dl_dfc_raw, dl_dp_accum = loss_and_grads(sys, loss_type)
    dl_dfc = Vector(reinterpret(T, SVector{3, T}.(dl_dfc_raw)))
    atom_masses_3N = repeat(masses(sys); inner=3)
    dt2 = dt^2
    dF_dp = zeros(T, n_atoms_x3, n_params)
    dF_dp_threads = zeros(T, n_atoms_x3, n_params, n_threads)
    dF_dp_sum = zeros(T, 1, n_params)
    dl_dfi = zeros(T, n_atoms_x3)
    dl_dfi_threads = zeros(T, n_atoms_x3, n_threads)
    accum_A = dl_dfc .* (1 .+ exp_mγdt)
    accum_B = dl_dfc .* (1 .+ exp_mγdt)
    accum_C = zeros(T, n_atoms_x3)
    accum_D = dl_dfc .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt)

    forces_flat = zeros(T, n_atoms_x3)
    forces_flat_threads = zeros(T, n_atoms_x3, n_threads)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    O_charge = params[end - n_bonded_params]

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
            @timeit to "neighbour finding" neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
            @timeit to "calc_F_and_grads!" calc_F_and_grads!(forces_flat, forces_flat_threads,
                    dl_dfi, dl_dfi_threads, dF_dp, dF_dp_threads, sys.coords, sys.boundary,
                    sys.atoms, sys.pairwise_inters, sys.specific_inter_lists, neighbors,
                    accum_C, atom_masses_3N, O_charge, n_params, n_threads)
            @timeit to "loop part 2" begin
                @timeit to "2.a" begin
                    norm_dl_dfi = norm(dl_dfi)
                    if norm_dl_dfi > clip_norm_threshold
                        dl_dfi .= dl_dfi .* clip_norm_threshold ./ norm_dl_dfi
                    end
                end
                @timeit to "2.b" sys.velocities .= sys.velocities .- reinterpret(SVector{3, T}, forces_flat) .* dt ./ masses(sys)
                @timeit to "2.c" dF_dp .= accum_A .* (dt2 ./ (2 .* atom_masses_3N)) .* dF_dp
                @timeit to "2.d" sum!(dF_dp_sum, dF_dp)
                @timeit to "2.e" dl_dp_accum .= dl_dp_accum .+ dropdims(dF_dp_sum; dims=1)
                @timeit to "2.f" accum_B .= exp_mγdt .* accum_B .+ dt2 .* dl_dfi
                @timeit to "2.g" accum_A .= accum_A .+ accum_B
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
                dl_dfc .= Vector(reinterpret(T, SVector{3, T}.(dl_dfc_raw_block)))
                dl_dp_accum .= dl_dp_accum .+ dl_dp_accum_block
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
    dl_dp = dl_dp_accum ./ n_blocks
    println("rev n_threads ", n_threads)
    println("rev loss ", l)
    println("rev gradient ", dl_dp)

    return sys, l, dl_dp, l_rdf, l_ev
end

inter_type = :lj
loss_type = :both
σ = T(0.315)
ϵ = T(0.636)
α, β = T(16.766), T(4.427)
A, B, C = T(133996.0), T(27.6), T(0.0168849)
α_ljsc, λ = T(0.2), T(0.1)
δ, γ_buff = T(0.07), T(0.12)
O_charge = T(-0.834)
kb, r0 = T(462750.4), T(0.09572)
ka, θ0 = T(836.8), T(1.82421813418)
if inter_type == :lj
    params = [σ, ϵ, O_charge, kb, r0, ka, θ0]
elseif inter_type == :dexp
    params = [σ, ϵ, α, β, O_charge, kb, r0, ka, θ0]
elseif inter_type == :buck
    params = [A, B, C, O_charge, kb, r0, ka, θ0]
elseif inter_type == :ljsc
    params = [σ, ϵ, α_ljsc, λ, O_charge, kb, r0, ka, θ0]
elseif inter_type == :buff
    params = [σ, ϵ, δ, γ_buff, O_charge, kb, r0, ka, θ0]
end

γ = T(1.0)
n_steps_equil = 100
n_steps       = 100 # May also want to change below
n_steps_loss  = n_steps # Should be a factor of n_steps
n_steps_trunc = n_steps
dt = T(0.001) # ps
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
@test maximum(norm.(coords_start .- sys_reverse.coords)) < (T == Float64 ? 1e-9 : 1e-4)
@test abs(loss_rev - loss_fwd) < 1e-7
@test abs(loss_rev - loss_rew) < 1e-7

println("Grads forward:        ", grads_forward)
println("Grads reweighting:    ", grads_rew)
println("Grads reverse:        ", grads_reverse)
println("Diff forward reverse: ", abs.(grads_forward .- grads_reverse))
println("%err forward reverse: ", 100 .* abs.(grads_forward .- grads_reverse) ./ abs.(grads_forward))

function train(inter_type, loss_type, params_start)
    reweighting = false
    γ = T(1.0)
    dt = T(0.001) # ps
    n_steps = 50_000 # 50 ps
    n_steps_equil = 10_000 # 10 ps
    n_steps_loss = 2_000 # 2 ps
    n_steps_trunc = 200
    use_barostat = true
    clip_norm_threshold = T(Inf)
    n_threads = Threads.nthreads()
    n_epochs = 10_000
    grad_clamp_val = T(1e3)
    learning_rate = T(2e-3)
    scale_params = true
    params = copy(params_start)
    if scale_params
        params_scaled = ones(T, length(params))
        opt = Optimisers.setup(Optimisers.Adam(learning_rate), params_scaled)
    else
        opt = Optimisers.setup(Optimisers.Adam(learning_rate), params)
    end

    for epoch_n in 1:n_epochs
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

        isdir(out_dir) || mkdir(out_dir)
        open("$out_dir/train_$run_n.log", "a") do of
            println(
                of,
                "$epoch_n $loss_rev $l_rdf $l_ev ",
                join(params, " "), " ",
                join(grads_reverse, " "),
            )
        end

        if scale_params
            scaled_grads = grads_reverse .* params_start
            opt, params_scaled = Optimisers.update!(
                opt,
                params_scaled,
                clamp.(scaled_grads, -grad_clamp_val, grad_clamp_val),
            )
            params .= params_scaled .* params_start
        else
            opt, params = Optimisers.update!(
                opt,
                params,
                clamp.(grads_reverse, -grad_clamp_val, grad_clamp_val),
            )
        end
    end
end

function train_lj()
    σs        = T.([0.315])
    ϵs        = T.([0.636])
    O_charges = T.([-0.834])

    inds = Iterators.product(eachindex(σs), eachindex(O_charges))
    σ_ind, O_charge_ind = collect(inds)[run_n]
    σ = σs[σ_ind]
    ϵ = ϵs[1]
    O_charge = O_charges[O_charge_ind]
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, O_charge, kb, r0, ka, θ0]

    inter_type = :lj
    loss_type = :both
    train(inter_type, loss_type, params)
end

function train_dexp()
    σs        = T.([0.315])
    ϵs        = T.([0.636])
    αs        = T.([16.766])
    βs        = T.([4.427])
    O_charges = T.([-0.834])

    inds = Iterators.product(eachindex(σs), eachindex(ϵs), eachindex(αs),
                             eachindex(βs), eachindex(O_charges))
    σ_ind, ϵ_ind, α_ind, β_ind, O_charge_ind = collect(inds)[run_n]
    σ = σs[σ_ind]
    ϵ = ϵs[ϵ_ind]
    α = αs[α_ind]
    β = βs[β_ind]
    O_charge = O_charges[O_charge_ind]
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, α, β, O_charge, kb, r0, ka, θ0]

    inter_type = :dexp
    loss_type = :both
    train(inter_type, loss_type, params)
end

function train_buck()
    As        = T.([359999.2])
    Bs        = T.([37.795])
    Cs        = T.([0.002343])
    O_charges = T.([-0.834])

    inds = Iterators.product(eachindex(As), eachindex(Bs), eachindex(Cs), eachindex(O_charges))
    A_ind, B_ind, C_ind, O_charge_ind = collect(inds)[run_n]
    A = As[A_ind]
    B = Bs[B_ind]
    C = Cs[C_ind]
    O_charge = O_charges[O_charge_ind]
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [A, B, C, O_charge, kb, r0, ka, θ0]

    inter_type = :buck
    loss_type = :both
    train(inter_type, loss_type, params)
end

function train_ljsc()
    σs        = T.([0.315])
    ϵs        = T.([0.636])
    αs        = T.([0.1])
    λs        = T.([0.1])
    O_charges = T.([-0.834])

    inds = Iterators.product(eachindex(σs), eachindex(ϵs), eachindex(αs),
                             eachindex(λs), eachindex(O_charges))
    σ_ind, ϵ_ind, α_ind, λ_ind, O_charge_ind = collect(inds)[run_n]
    σ = σs[σ_ind]
    ϵ = ϵs[ϵ_ind]
    α = αs[α_ind]
    λ = λs[λ_ind]
    O_charge = O_charges[O_charge_ind]
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, α, λ, O_charge, kb, r0, ka, θ0]

    inter_type = :ljsc
    loss_type = :both
    train(inter_type, loss_type, params)
end

function train_buff()
    σs        = T.([0.315])
    ϵs        = T.([0.636])
    δs        = T.([0.155])
    γs        = T.([0.096])
    O_charges = T.([-0.834])

    inds = Iterators.product(eachindex(σs), eachindex(ϵs), eachindex(δs),
                             eachindex(γs), eachindex(O_charges))
    σ_ind, ϵ_ind, δ_ind, γ_ind, O_charge_ind = collect(inds)[run_n]
    σ = σs[σ_ind]
    ϵ = ϵs[ϵ_ind]
    δ = δs[δ_ind]
    γ = γs[γ_ind]
    O_charge = O_charges[O_charge_ind]
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, δ, γ, O_charge, kb, r0, ka, θ0]

    inter_type = :buff
    loss_type = :both
    train(inter_type, loss_type, params)
end

if run_inter_type == "lj"
    train_lj()
elseif run_inter_type == "dexp"
    train_dexp()
elseif run_inter_type == "buck"
    train_buck()
elseif run_inter_type == "ljsc"
    train_ljsc()
elseif run_inter_type == "buff"
    train_buff()
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
#=
# Accuracy compared to RAD (Figure 1B)
for n_steps in (1, 2, 5, 10, 25, 50, 100, 200)
    inter_type = :lj
    loss_type = :both
    σ = T(0.315)
    ϵ = T(0.636)
    O_charge = T(-0.834)
    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, O_charge, kb, r0, ka, θ0]
    γ = T(1.0)
    n_steps_equil = 100
    n_steps_loss  = n_steps # Should be a factor of n_steps
    n_steps_trunc = n_steps
    dt = T(0.001) # ps
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

    sys_reverse, loss_rev, grads_reverse, _, _ = reverse_sim(sys_forward.coords, sys_forward.velocities,
                        nothing, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, params, dt,
                        n_threads, sys_forward.loggers, T(Inf), inter_type, loss_type)

    @test maximum(norm.(coords_start .- sys_reverse.coords)) < (T == Float64 ? 1e-9 : 1e-3)
    @test abs(loss_rev - loss_fwd) < (T == Float64 ? 1e-7 : 1e-2)

    println("n_steps $n_steps %err forward reverse: ",
            100 .* abs.(grads_forward .- grads_reverse) ./ abs.(grads_forward))
end
=#
#=
# Longer run gradient accuracy, variance and reversible/reweighting comparison (Figure 1C-E)
for rep_n in 1:2
    inter_type = :lj
    loss_type = :both
    σs        = T.([0.301, 0.308, 0.315, 0.322, 0.329])
    ϵs        = T.([0.436, 0.536, 0.636, 0.736, 0.836])
    O_charges = T.([-0.934, -0.884, -0.834, -0.784, -0.734])

    inds = Iterators.product(eachindex(σs), eachindex(ϵs), eachindex(O_charges))
    σ_ind, ϵ_ind, O_charge_ind = collect(inds)[run_n]
    σ = σs[σ_ind]
    ϵ = ϵs[ϵ_ind]
    O_charge = O_charges[O_charge_ind]

    kb, r0 = T(462750.4), T(0.09572)
    ka, θ0 = T(836.8), T(1.82421813418)
    params = [σ, ϵ, O_charge, kb, r0, ka, θ0]
    γ = T(1.0)
    n_steps = 50_000 # 50 ps
    n_steps_equil = 10_000 # 10 ps
    n_steps_loss  = 2_000 # 2 ps
    n_steps_trunc = 200
    dt = T(0.001) # ps
    n_threads = Threads.nthreads()
    noise_seeds = [rand_seed() for i in 1:n_steps]

    noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
    velocities_equil = [random_velocity(m, temp) for m in atom_masses]

    sys_equil, _, _ = forward_sim(nothing, velocities_equil, nothing, noise_equil_seeds, γ,
                        n_steps_equil, n_steps_equil, params, dt, n_threads, inter_type,
                        loss_type, false, false, true)
    coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)
    boundary_start = deepcopy(sys_equil.boundary)

    sys_forward, _, _ = forward_sim(coords_start, velocities_start, boundary_start,
                        noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads,
                        inter_type, loss_type, false, false)

    sys_rew, loss_rew, grads_rew, _, _ = reweighting_sim(coords_start, velocities_start, boundary_start,
                        noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads,
                        inter_type, loss_type)

    sys_reverse, loss_rev, grads_reverse, _, _ = reverse_sim(sys_forward.coords, sys_forward.velocities,
                        boundary_start, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, params, dt,
                        n_threads, sys_forward.loggers, T(Inf), inter_type, loss_type)

    println("gradvar ", rep_n, " loss ", loss_rev)
    println("gradvar ", rep_n, " rev " , grads_reverse)
    println("gradvar ", rep_n, " rew " , grads_rew)
end
=#
#=
# Effect of truncation (Figure S1)
n_steps_trunc_list = [50, 100, 200, 500, 750, 1000]
losses, grads_rev_list = Dict(i => [] for i in n_steps_trunc_list), Dict(i => [] for i in n_steps_trunc_list)
for rep_n in 1:125
    for n_steps_trunc in n_steps_trunc_list
        inter_type = :lj
        loss_type = :both
        σs        = T.([0.301, 0.308, 0.315, 0.322, 0.329])
        ϵs        = T.([0.436, 0.536, 0.636, 0.736, 0.836])
        O_charges = T.([-0.934, -0.884, -0.834, -0.784, -0.734])

        inds = Iterators.product(eachindex(σs), eachindex(ϵs), eachindex(O_charges))
        σ_ind, ϵ_ind, O_charge_ind = collect(inds)[rep_n]
        σ = σs[σ_ind]
        ϵ = ϵs[ϵ_ind]
        O_charge = O_charges[O_charge_ind]

        kb, r0 = T(462750.4), T(0.09572)
        ka, θ0 = T(836.8), T(1.82421813418)
        params = [σ, ϵ, O_charge, kb, r0, ka, θ0]
        γ = T(1.0)
        n_steps_equil = 100
        n_steps = 1000
        n_steps_loss  = n_steps # Should be a factor of n_steps
        dt = T(0.001) # ps
        n_threads = Threads.nthreads()
        noise_seeds = [rand_seed() for i in 1:n_steps]

        noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
        velocities_equil = [random_velocity(m, temp) for m in atom_masses]

        sys_equil, _, _ = forward_sim(nothing, velocities_equil, nothing, noise_equil_seeds, γ,
                            n_steps_equil, n_steps_equil, params, dt, n_threads, inter_type,
                            loss_type, false, false)
        coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

        sys_forward, _, _ = forward_sim(coords_start, velocities_start, nothing,
                            noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads,
                            inter_type, loss_type, false, false)

        sys_reverse, loss_rev, grads_reverse, _, _ = reverse_sim(sys_forward.coords, sys_forward.velocities,
                            nothing, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, params, dt,
                            n_threads, sys_forward.loggers, T(Inf), inter_type, loss_type)

        push!(losses[n_steps_trunc], loss_rev)
        push!(grads_rev_list[n_steps_trunc], grads_reverse[1])
        println(rep_n)
    end
end
=#
