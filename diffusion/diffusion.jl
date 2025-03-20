# Reversible simulation for gas diffusion in water
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT

using Molly
using Zygote
import Enzyme
import BioStructures
using UnitfulAtomic
using ChainRulesCore
using Optimisers
using FiniteDifferences
using LoopVectorization
using Polyester
using TimerOutputs
using CairoMakie
using LinearAlgebra
using Random
using Statistics
using Test

Enzyme.API.runtimeActivity!(true)

const out_dir        = (length(ARGS) >= 1 ? ARGS[1] : "train_diffusion")
const run_inter_type = (length(ARGS) >= 2 ? ARGS[2] : "lj")
const run_n          = (length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1)

const T = Float64
const structure_file = "tip3p.pdb"
const n_molecules = 895
const n_gas_molecules = 10
const n_atoms = n_molecules * 3 - n_gas_molecules
const n_atoms_x3 = n_atoms * 3
const dist_cutoff    = T(1.2) # nm
const dist_neighbors = T(1.35) # nm
const n_steps_nf = 20
const crf_const =  T((1 / (dist_cutoff^3)) * ((78.3 - 1) / (2 * 78.3 + 1)))
const O_mass = T(15.99943)
const H_mass = T(1.007947)
const atom_masses = vcat(
    repeat([O_mass, H_mass, H_mass]; outer=(n_molecules-n_gas_molecules)),
    repeat([O_mass, O_mass]; outer=n_gas_molecules),
)
const temp = T(295.15) # K
const two_pow_1o6 = T(2^(1/6))
const bond_length_O2 = T(0.12074)
const kcal_to_kJ = T(4.184)
const boundary = CubicBoundary(T(3.0))
const to = TimerOutput()

coords_copy(sys, neighbors=nothing; kwargs...) = copy(sys.coords)
velocities_copy(sys, neighbors=nothing; kwargs...) = copy(sys.velocities)
CoordCopyLogger(n_steps_log) = GeneralObservableLogger(coords_copy,
                                        Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)
VelocityCopyLogger(n_steps_log) = GeneralObservableLogger(velocities_copy,
                                        Array{SArray{Tuple{3}, T, 1, 3}, 1}, n_steps_log)

# New interactions need to define an entry in generate_inter, inter_n_params, force_inter,
#   force_grads_inter!, inject_atom_grads, Molly.force, Molly.potential_energy,
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
        # Params are σ, ϵ, σ_gas, ϵ_gas
        return LennardJones{false, typeof(cutoff), Int, Int, typeof(NoUnits), typeof(NoUnits)}(
                    cutoff, true, true, 1, 1, NoUnits, NoUnits)
    elseif inter_type == :dexp
        # Params are σ, ϵ, σ_gas, ϵ_gas, α, β
        α, β = params[5], params[6]
        return DoubleExponential(α, β, cutoff)
    elseif inter_type == :buck
        # Params are A, B, C, A_gas, B_gas, C_gas
        return Buckingham(cutoff)
    elseif inter_type == :ljsc
        # Params are σ, ϵ, σ_gas, ϵ_gas, α, λ
        α, λ = params[5], params[6]
        p = 2
        σ6_fac = α * λ^p
        return LennardJonesSoftCore{false, typeof(cutoff), T, T, typeof(p), T, Int, Int,
                                    typeof(NoUnits), typeof(NoUnits)}(
                    cutoff, α, λ, p, σ6_fac, true, true, 1, 1, NoUnits, NoUnits)
    elseif inter_type == :buff
        # Params are σ, ϵ, σ_gas, ϵ_gas, δ, γ
        δ, γ = params[5], params[6]
        return Buffered147(δ, γ, cutoff)
    else
        throw(ArgumentError("inter_type not recognised"))
    end
end

inter_n_params(::CoulombReactionField) = 1
inter_n_params(::LennardJones) = 4
inter_n_params(::DoubleExponential) = 6
inter_n_params(::Buckingham) = 6
inter_n_params(::LennardJonesSoftCore) = 6
inter_n_params(::Buffered147) = 6

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
        n_boundary = n_atoms_x3 - n_gas_molecules * 2 * 3
        wat_i, wat_j = (i3 + 3) <= n_boundary, (j3 + 3) <= n_boundary
        dσ_dσw = (T(wat_i) + T(wat_j)) / 2
        dσ_dσg = 1 - dσ_dσw
        dϵ_dϵw = (wat_i && wat_j) ? one(T) : ((!wat_i && !wat_j) ? zero(T) : (wat_i ? (atom_j.ϵ / 2ϵ) : (atom_i.ϵ / 2ϵ)))
        dϵ_dϵg = (wat_i && wat_j) ? zero(T) : ((!wat_i && !wat_j) ? one(T) : (wat_i ? (atom_i.ϵ / 2ϵ) : (atom_j.ϵ / 2ϵ)))
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵg
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
        n_boundary = n_atoms_x3 - n_gas_molecules * 2 * 3
        wat_i, wat_j = (i3 + 3) <= n_boundary, (j3 + 3) <= n_boundary
        dσ_dσw = (T(wat_i) + T(wat_j)) / 2
        dσ_dσg = 1 - dσ_dσw
        dϵ_dϵw = (wat_i && wat_j) ? one(T) : ((!wat_i && !wat_j) ? zero(T) : (wat_i ? (atom_j.ϵ / 2ϵ) : (atom_i.ϵ / 2ϵ)))
        dϵ_dϵg = (wat_i && wat_j) ? zero(T) : ((!wat_i && !wat_j) ? one(T) : (wat_i ? (atom_i.ϵ / 2ϵ) : (atom_j.ϵ / 2ϵ)))
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[i3 + ui, 5, chunk_i] -= f_dF_dα[ui]
            dF_dp_threads[j3 + ui, 5, chunk_i] += f_dF_dα[ui]
            dF_dp_threads[i3 + ui, 6, chunk_i] -= f_dF_dβ[ui]
            dF_dp_threads[j3 + ui, 6, chunk_i] += f_dF_dβ[ui]
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
        n_boundary = n_atoms_x3 - n_gas_molecules * 2 * 3
        wat_i, wat_j = (i3 + 3) <= n_boundary, (j3 + 3) <= n_boundary
        dp_dpw = (T(wat_i) + T(wat_j)) / 2
        dp_dpg = 1 - dp_dpw
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dA[ui] * dp_dpw
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dA[ui] * dp_dpw
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dB[ui] * dp_dpw
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dB[ui] * dp_dpw
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dC[ui] * dp_dpw
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dC[ui] * dp_dpw
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dA[ui] * dp_dpg
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dA[ui] * dp_dpg
            dF_dp_threads[i3 + ui, 5, chunk_i] -= f_dF_dB[ui] * dp_dpg
            dF_dp_threads[j3 + ui, 5, chunk_i] += f_dF_dB[ui] * dp_dpg
            dF_dp_threads[i3 + ui, 6, chunk_i] -= f_dF_dC[ui] * dp_dpg
            dF_dp_threads[j3 + ui, 6, chunk_i] += f_dF_dC[ui] * dp_dpg
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
        n_boundary = n_atoms_x3 - n_gas_molecules * 2 * 3
        wat_i, wat_j = (i3 + 3) <= n_boundary, (j3 + 3) <= n_boundary
        dσ_dσw = (T(wat_i) + T(wat_j)) / 2
        dσ_dσg = 1 - dσ_dσw
        dϵ_dϵw = (wat_i && wat_j) ? one(T) : ((!wat_i && !wat_j) ? zero(T) : (wat_i ? (atom_j.ϵ / 2ϵ) : (atom_i.ϵ / 2ϵ)))
        dϵ_dϵg = (wat_i && wat_j) ? zero(T) : ((!wat_i && !wat_j) ? one(T) : (wat_i ? (atom_i.ϵ / 2ϵ) : (atom_j.ϵ / 2ϵ)))
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[i3 + ui, 5, chunk_i] -= f_dF_dα[ui]
            dF_dp_threads[j3 + ui, 5, chunk_i] += f_dF_dα[ui]
            dF_dp_threads[i3 + ui, 6, chunk_i] -= f_dF_dλ[ui]
            dF_dp_threads[j3 + ui, 6, chunk_i] += f_dF_dλ[ui]
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
        n_boundary = n_atoms_x3 - n_gas_molecules * 2 * 3
        wat_i, wat_j = (i3 + 3) <= n_boundary, (j3 + 3) <= n_boundary
        dσ_dσw = (T(wat_i) + T(wat_j)) / 2
        dσ_dσg = 1 - dσ_dσw
        dϵ_dϵw = (wat_i && wat_j) ? one(T) : ((!wat_i && !wat_j) ? zero(T) : (wat_i ? (atom_j.ϵ / 2ϵ) : (atom_i.ϵ / 2ϵ)))
        dϵ_dϵg = (wat_i && wat_j) ? zero(T) : ((!wat_i && !wat_j) ? one(T) : (wat_i ? (atom_i.ϵ / 2ϵ) : (atom_j.ϵ / 2ϵ)))
        @turbo for ui in 1:3
            dF_dp_threads[i3 + ui, 1, chunk_i] -= f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[j3 + ui, 1, chunk_i] += f_dF_dσ[ui] * dσ_dσw
            dF_dp_threads[i3 + ui, 2, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[j3 + ui, 2, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵw
            dF_dp_threads[i3 + ui, 3, chunk_i] -= f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[j3 + ui, 3, chunk_i] += f_dF_dσ[ui] * dσ_dσg
            dF_dp_threads[i3 + ui, 4, chunk_i] -= f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[j3 + ui, 4, chunk_i] += f_dF_dϵ[ui] * dϵ_dϵg
            dF_dp_threads[i3 + ui, 5, chunk_i] -= f_dF_dδ[ui]
            dF_dp_threads[j3 + ui, 5, chunk_i] += f_dF_dδ[ui]
            dF_dp_threads[i3 + ui, 6, chunk_i] -= f_dF_dγ[ui]
            dF_dp_threads[j3 + ui, 6, chunk_i] += f_dF_dγ[ui]
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

# Needs to be Zygote-friendly
function inject_atom_grads(::Union{LennardJones, DoubleExponential, LennardJonesSoftCore, Buffered147},
                           params)
    σ, ϵ, σ_gas, ϵ_gas, O_charge = params[1], params[2], params[3], params[4], params[end]
    H_charge = -O_charge / 2
    return map(1:n_atoms) do i
        if i <= (n_atoms - n_gas_molecules * 2)
            if i % 3 == 1
                Atom(i, O_charge, O_mass, σ, ϵ, false)
            else
                Atom(i, H_charge, H_mass, zero(T), zero(T), false)
            end
        else
            Atom(i, zero(T), O_mass, σ_gas, ϵ_gas, false)
        end
    end
end

function inject_atom_grads(::Buckingham, params)
    A, B, C, A_gas, B_gas, C_gas, O_charge = params
    H_charge = -O_charge / 2
    return map(1:n_atoms) do i
        if i <= (n_atoms - n_gas_molecules * 2)
            if i % 3 == 1
                BuckinghamAtom(i, O_charge, O_mass, A, B, C)
            else
                BuckinghamAtom(i, H_charge, H_mass, zero(T), zero(T), zero(T))
            end
        else
            BuckinghamAtom(i, zero(T), O_mass, A_gas, B_gas, C_gas)
        end
    end
end

function loss_fn_init(loss_type, n_blocks)
    if loss_type == :mse || loss_type == :mae
        return [zero(SVector{3, T}) for _ in 1:(n_blocks*n_gas_molecules)]
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function gas_coords(sys)
    all_coord_term = sum(sys.coords) * 0 # Makes coordinate gradients zero not nothing
    # Using map here gave gradients that were 10x off
    inds1 = [length(sys) - 2*n_gas_molecules + 2*i - 1 for i in 1:n_gas_molecules]
    inds2 = inds1 .+ 1
    vecs = vector.((sys.coords[inds1] .+ (all_coord_term,)), sys.coords[inds2], (sys.boundary,))
    centers = sys.coords[inds1] .+ (vecs ./ 2)
    return wrap_coords.(centers, (boundary,))
end

function loss_fn_accum(loss_accum, sys, loss_type, block_n)
    if loss_type == :mse || loss_type == :mae
        return vcat(
            loss_accum[1:((block_n-1)*n_gas_molecules)],
            gas_coords(sys),
            loss_accum[((block_n*n_gas_molecules)+1):end],
        )
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function calculate_D(loss_accum, loss_frequency)
    D = zero(T)
    n_start_frames = length(loss_accum) ÷ (2 * n_gas_molecules)
    for start_i in 1:n_start_frames
        coords_t0 = loss_accum[(((start_i-1)*n_gas_molecules)+1):(start_i*n_gas_molecules)]
        coords_t = copy(coords_t0)
        diffs = similar(coords_t0)
        for j in (start_i+1):(start_i+n_start_frames)
            c = loss_accum[(((j-1)*n_gas_molecules)+1):(j*n_gas_molecules)]
            diffs .= vector.(coords_t, c, (boundary,))
            coords_t .+= diffs
        end
        msd = mean(sum.(abs2, coords_t .- coords_t0))
        D += msd / (n_start_frames * loss_frequency)
    end
    return D / (6 * n_start_frames)
end

function ChainRulesCore.rrule(::typeof(calculate_D), loss_accum, loss_frequency)
    Y = calculate_D(loss_accum, loss_frequency)

    function calculate_D_pullback(dy)
        d_loss_accum = zero(loss_accum)
        Enzyme.autodiff(
            Enzyme.Reverse,
            calculate_D,
            Enzyme.Active,
            Enzyme.Duplicated(loss_accum, d_loss_accum),
            Enzyme.Const(loss_frequency),
        )
        return NoTangent(), d_loss_accum .* dy, NoTangent()
    end

    return Y, calculate_D_pullback
end

function loss_fn_final(loss_accum, loss_type, loss_frequency)
    D = calculate_D(loss_accum, loss_frequency)
    D_conv = D * T(1e-6) # Convert from nm^2 * ps^-1 to m^2 * s^-1
    D_exp = T(2.0e-9) # m^2 * s^-1
    if loss_type == :mse
        loss_weight = 1e18 # m^-4 * s^2
        return loss_weight * (D_conv - D_exp)^2
    elseif loss_type == :mae
        loss_weight = 1e9 # m^-2 * s^1
        return loss_weight * abs(D_conv - D_exp)
    else
        throw(ArgumentError("loss_type not recognised"))
    end
end

function loss_and_grads(sys, loss_accum, loss_type, loss_frequency)
    l, grads = withgradient(loss_fn_final, loss_accum, loss_type, loss_frequency)
    D = calculate_D(loss_accum, loss_frequency)
    D_conv = D * T(1e-6) # Convert from nm^2 * ps^-1 to m^2 * s^-1
    dl_dgc_snapshots = grads[1]
    dl_dp = zeros(T, sum(inter_n_params, sys.pairwise_inters))
    return l, D_conv, dl_dgc_snapshots, dl_dp
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

function setup_system(use_loggers, n_steps_log=0)
    inds = collect(1:n_molecules)
    shuffle!(inds)
    struc = read(structure_file, BioStructures.PDBFormat)
    mols = BioStructures.collectresidues(struc)
    atoms, atoms_data, coords = [], [], []
    bond_is, bond_js, bond_inters = [], [], []
    angle_is, angle_js, angle_ks, angle_inters = [], [], [], []

    for (ri, i) in enumerate(inds[1:(n_molecules-n_gas_molecules)])
        for atom_name in ("O", "H1", "H2")
            # TIP3P parameters
            if atom_name == "O"
                at = Atom(charge=T(-0.834), mass=O_mass, σ=T(0.31507524065751241), ϵ=T(0.635968))
                at_data = AtomData(atom_name=atom_name, res_number=ri, res_name="HOH", element="O")
            else
                at = Atom(charge=T(0.417), mass=H_mass, σ=zero(T), ϵ=zero(T))
                at_data = AtomData(atom_name=atom_name, res_number=ri, res_name="HOH", element="H")
            end
            push!(atoms, at)
            push!(atoms_data, at_data)
            push!(coords, SVector{3, T}(BioStructures.coords(mols[i][atom_name]) ./ 10))
        end
        append!(bond_is, [length(atoms) - 2, length(atoms) - 2])
        append!(bond_js, [length(atoms) - 1, length(atoms)])
        for _ in 1:2
            push!(bond_inters, HarmonicBond(k=T(462750.4), r0=T(0.09572)))
        end
        push!(angle_is, length(atoms) - 1)
        push!(angle_js, length(atoms) - 2)
        push!(angle_ks, length(atoms))
        push!(angle_inters, HarmonicAngle(k=T(836.8), θ0=T(1.82421813418)))
    end

    for (ri, i) in enumerate(inds[(n_molecules-n_gas_molecules+1):end])
        # Parameters from https://pubs.acs.org/doi/10.1021/acs.jctc.0c01132
        for _ in 1:2
            push!(atoms, Atom(charge=zero(T), mass=O_mass, σ=T(0.3297), ϵ=T(0.1047 * kcal_to_kJ)))
        end
        for atom_name in ("O1", "O2")
            push!(atoms_data, AtomData(atom_name=atom_name, res_number=(ri+n_molecules-n_gas_molecules),
                                       res_name="OXY", element="O"))
        end
        coord_O  = SVector{3, T}(BioStructures.coords(mols[i]["O" ]) ./ 10)
        coord_H1 = SVector{3, T}(BioStructures.coords(mols[i]["H1"]) ./ 10)
        coord_O2 = coord_O + normalize(coord_H1 - coord_O) * bond_length_O2
        push!(coords, coord_O)
        push!(coords, coord_O2)
        push!(bond_is, length(atoms) - 1)
        push!(bond_js, length(atoms))
        push!(bond_inters, HarmonicBond(k=T(1640.4 * kcal_to_kJ * 100), r0=bond_length_O2))
    end

    atoms, atoms_data = [atoms...], [atoms_data...]
    coords = wrap_coords.([coords...], (boundary,))

    lj = LennardJones(
        cutoff=DistanceCutoff(dist_cutoff),
        use_neighbors=true,
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    crf = CoulombReactionField(
        dist_cutoff=T(dist_cutoff),
        solvent_dielectric=T(Molly.crf_solvent_dielectric),
        use_neighbors=true,
        coulomb_const=T(ustrip(Molly.coulombconst)),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    bonds = InteractionList2Atoms([bond_is...], [bond_js...], [bond_inters...])
    angles = InteractionList3Atoms([angle_is...], [angle_js...], [angle_ks...], [angle_inters...])

    n_atoms = length(atoms)
    eligible = trues(n_atoms, n_atoms)
    for (i, j, k) in zip(angle_is, angle_js, angle_ks)
        eligible[i, j] = false
        eligible[j, i] = false
        eligible[i, k] = false
        eligible[k, i] = false
        eligible[j, k] = false
        eligible[k, j] = false
    end
    for bi in (length(bond_inters)-n_gas_molecules+1):length(bond_inters)
        i, j = bond_is[bi], bond_js[bi]
        eligible[i, j] = false
        eligible[j, i] = false
    end

    neighbor_finder = CellListMapNeighborFinder(
        eligible=eligible,
        n_steps=n_steps_nf,
        x0=coords,
        unit_cell=boundary,
        dist_cutoff=dist_neighbors,
    )

    if use_loggers
        loggers = (
            coords=CoordCopyLogger(n_steps_log),
            velocities=VelocityCopyLogger(n_steps_log),
        )
    else
        loggers = nothing
    end

    sys = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        atoms_data=atoms_data,
        pairwise_inters=(lj, crf),
        specific_inter_lists=(bonds, angles),
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    return sys
end

function forward_sim(coords_in, velocities_in, noise_seeds, γ, n_steps, n_steps_loss, params,
                     dt, n_threads, inter_type, loss_type, run_grads=true, calc_loss=true)
    sys_init = setup_system(true, n_steps_loss)
    pis = (
        generate_inter(inter_type, params),
        sys_init.pairwise_inters[2],
    )
    atoms = inject_atom_grads(pis[1], params)

    if isnothing(coords_in)
        coords_start = copy(sys_init.coords)
    else
        coords_start = copy(coords_in)
    end
    if isnothing(velocities_in)
        velocities_start = [random_velocity(m, temp) for m in atom_masses]
    else
        velocities_start = copy(velocities_in)
    end

    sys = System(
        deepcopy(sys_init);
        atoms=atoms,
        coords=copy(coords_start),
        velocities=copy(velocities_start),
        pairwise_inters=pis,
    )

    if isnothing(coords_in)
        minimizer = SteepestDescentMinimizer(step_size=T(0.01), tol=T(1000.0))
        simulate!(sys, minimizer; run_loggers=false)
    end

    n_blocks = (n_steps == 0 ? 1 : n_steps ÷ n_steps_loss)
    loss_frequency = n_steps_loss * dt
    vel_scale = exp(-γ * dt)
    noise_scale = sqrt(1 - vel_scale^2)

    forces_flat = zeros(T, n_atoms_x3)
    forces_flat_threads = zeros(T, n_atoms_x3, n_threads)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    loss_accum = loss_fn_init(loss_type, n_blocks)
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
            loss_accum = loss_fn_accum(loss_accum, sys, loss_type, step_n ÷ n_steps_loss)
        end
    end
    if calc_loss
        loss = loss_fn_final(loss_accum, loss_type, loss_frequency)
    else
        loss = zero(T)
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
            atoms_grad = inject_atom_grads(pis[1], params)
            sys_grad = System(
                deepcopy(sys_init);
                atoms=atoms_grad,
                coords=copy(coords_start),
                velocities=copy(velocities_start),
                pairwise_inters=pis,
            )
            neighbors_grad = find_neighbors(sys_grad, sys_grad.neighbor_finder; n_threads=n_threads)
            loss_accum_rad = loss_fn_init(loss_type, n_blocks)
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
                    loss_accum_rad = loss_fn_accum(loss_accum_rad, sys_grad, loss_type, step_n ÷ n_steps_loss)
                end
            end
            loss_rad = loss_fn_final(loss_accum_rad, loss_type, loss_frequency)
            return loss_rad
        end

        @timeit to "forward grad" begin
            loss_rad, grads_tup = withgradient(loss_forward, params)
            grads = grads_tup[1]
        end
        @assert isapprox(loss_rad, loss; rtol=1e-2)
        println("fwd loss ", loss_rad)
        println("fwd gradients ", grads)
    else
        grads = fill(nothing, sum(inter_n_params, sys.pairwise_inters))
    end

    return sys, loss, loss_accum, grads
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
                dF_dp_threads[i3 + ui, n_params, chunk_i] -= f_dF_dc[ui]
                dF_dp_threads[j3 + ui, n_params, chunk_i] += f_dF_dc[ui]
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
                F_div_r3 = F / (r2 * r)
                f_dFi_dxjn = -bond.k * ndr
                i3 = 3 * (i - 1)
                j3 = 3 * (j - 1)

                for ui in 1:3
                    forces_flat_threads[i3 + ui, chunk_i] -= F * ndr[ui]
                    forces_flat_threads[j3 + ui, chunk_i] += F * ndr[ui]
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
                F = -ang.k * (θ - ang.θ0)
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

function reverse_sim(coords, velocities, noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc,
                     params, dt, n_threads, fwd_loggers, loss_accum_fwd, inter_type, loss_type)
    sys_init = setup_system(false)
    pis = (
        generate_inter(inter_type, params),
        sys_init.pairwise_inters[2],
    )
    atoms = inject_atom_grads(pis[1], params)
    sys = System(
        deepcopy(sys_init);
        atoms=atoms,
        coords=copy(coords),
        velocities=copy(velocities),
        pairwise_inters=pis,
    )
    exp_mγdt = exp(-γ * dt)
    noise_scale = sqrt(1 - exp_mγdt^2)
    n_blocks = n_steps ÷ n_steps_loss
    loss_frequency = n_steps_loss * dt
    n_params = sum(inter_n_params, sys.pairwise_inters)
    fwd_coords_logger = values(fwd_loggers.coords)
    fwd_velocities_logger = values(fwd_loggers.velocities)

    loss, D, dl_dgc_snapshots, dl_dp_accum = loss_and_grads(sys, loss_accum_fwd,
                                                            loss_type, loss_frequency)
    dl_dfc = zeros(T, n_atoms_x3)
    # Gas coordinates are average of the two atom coordinates
    dl_dfc_gas = repeat(dl_dgc_snapshots[(end-n_gas_molecules+1):end]; inner=2) ./ 2
    dl_dfc[(end-(n_gas_molecules*3*2)+1):end] .= reinterpret(T, dl_dfc_gas)
    atom_masses_3N = repeat(masses(sys); inner=3)
    dt2 = dt^2
    dF_dp = zeros(T, n_atoms_x3, n_params)
    dF_dp_threads = zeros(T, n_atoms_x3, n_params, n_threads)
    dF_dp_sum = zeros(T, 1, n_params)
    dl_dfi = zeros(T, n_atoms_x3)
    dl_dfi_threads = zeros(T, n_atoms_x3, n_threads)
    dl_dfi_accum_1 = dl_dfc .* (1 .+ exp_mγdt)
    dl_dfi_accum_2 = dl_dfc .* (1 .+ exp_mγdt)
    dl_dfi_accum_3 = zeros(T, n_atoms_x3)
    dl_dfi_accum_4 = dl_dfc .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt)

    forces_flat = zeros(T, n_atoms_x3)
    forces_flat_threads = zeros(T, n_atoms_x3, n_threads)
    noise = zeros(SVector{3, T}, n_atoms)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    O_charge = params[end]
    sys.coords .= fwd_coords_logger[end]
    sys.velocities .= fwd_velocities_logger[end]

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
            @timeit to "neighbour finding" neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
            @timeit to "calc_F_and_grads!" calc_F_and_grads!(forces_flat, forces_flat_threads,
                    dl_dfi, dl_dfi_threads, dF_dp, dF_dp_threads, sys.coords, sys.boundary,
                    sys.atoms, sys.pairwise_inters, sys.specific_inter_lists, neighbors,
                    dl_dfi_accum_3, atom_masses_3N, O_charge, n_params, n_threads)
            @timeit to "loop part 2" begin
                @timeit to "2.b" sys.velocities .= sys.velocities .- reinterpret(SVector{3, T}, forces_flat) .* dt ./ masses(sys)
                @timeit to "2.c" dF_dp .= dl_dfi_accum_1 .* (dt2 ./ (2 .* atom_masses_3N)) .* dF_dp
                @timeit to "2.d" sum!(dF_dp_sum, dF_dp)
                @timeit to "2.e" dl_dp_accum .= dl_dp_accum .+ dropdims(dF_dp_sum; dims=1)
                @timeit to "2.f" dl_dfi_accum_2 .= exp_mγdt .* dl_dfi_accum_2 .+ dt2 .* dl_dfi
                @timeit to "2.g" dl_dfi_accum_1 .= dl_dfi_accum_1 .+ dl_dfi_accum_2
            end
        end

        @timeit to "loop part 3" begin
            if loss_count != n_blocks
                sys.coords .= fwd_coords_logger[end-loss_count]
                sys.velocities .= fwd_velocities_logger[end-loss_count]
                dl_dfc .= zero(T)
                dl_dfc_gas .= repeat(dl_dgc_snapshots[(end-(n_gas_molecules*(loss_count+1))+1):(end-n_gas_molecules*loss_count)]; inner=2) ./ 2
                dl_dfc[(end-(n_gas_molecules*3*2)+1):end] .= reinterpret(T, dl_dfc_gas)
                dl_dfi .= zero(T)
                dl_dfi_accum_1 .= dl_dfc .* (1 .+ exp_mγdt)
                dl_dfi_accum_2 .= dl_dfc .* (1 .+ exp_mγdt)
                dl_dfi_accum_3 .= zero(T)
                dl_dfi_accum_4 .= dl_dfc .* (1 .+ 2 .* exp_mγdt .+ exp_mγdt .^ 2) ./ (2 .* exp_mγdt)
            end
        end
    end
    println("rev n_threads ", n_threads)
    println("rev loss ", loss)
    println("rev gradient ", dl_dp_accum)

    return sys, loss, D, dl_dp_accum
end

inter_type = :lj
loss_type = :mse
σ, ϵ = T(0.315), T(0.636)
σ_gas, ϵ_gas = T(0.3297), T(0.438)
α, β = T(16.766), T(4.427)
A, B, C = T(133996.0), T(27.6), T(0.0168849)
A_gas, B_gas, C_gas = A, B, C
α_ljsc, λ = T(0.2), T(0.1)
δ, γ_buff = T(0.07), T(0.12)
O_charge = T(-0.834)
if inter_type == :lj
    params = [σ, ϵ, σ_gas, ϵ_gas, O_charge]
elseif inter_type == :dexp
    params = [σ, ϵ, σ_gas, ϵ_gas, α, β, O_charge]
elseif inter_type == :buck
    params = [A, B, C, A_gas, B_gas, C_gas, O_charge]
elseif inter_type == :ljsc
    params = [σ, ϵ, σ_gas, ϵ_gas, α_ljsc, λ, O_charge]
elseif inter_type == :buff
    params = [σ, ϵ, σ_gas, ϵ_gas, δ, γ_buff, O_charge]
end

γ = T(0.0)
n_steps_equil = 10
n_steps       = 10 # May also want to change below
n_steps_loss  = 2 # Should be a factor of n_steps
n_steps_trunc = 10
dt = T(0.001) # ps
n_threads = Threads.nthreads()
noise_seeds = [rand_seed() for i in 1:n_steps]

noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
sys_equil, _, _ = forward_sim(nothing, nothing, noise_equil_seeds, γ, n_steps_equil,
                    n_steps_equil, params, dt, n_threads, inter_type, loss_type, false, false)
coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

sys_fwd, loss_fwd, loss_accum_fwd, grads_fwd = forward_sim(coords_start, velocities_start,
                    noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads, inter_type,
                    loss_type)

grads_fd = map(eachindex(params)) do i
    central_fdm(7, 1; max_range=0.02)(params[i]) do param_fd
        params_fd = [j == i ? param_fd : params[j] for j in eachindex(params)]
        sys_fwd, loss_fwd, loss_accum_fwd, grads_fwd = forward_sim(coords_start, velocities_start,
                    noise_seeds, γ, n_steps, n_steps_loss, params_fd, dt, n_threads, inter_type,
                    loss_type, false, true)
        return loss_fwd
    end
end

sys_rev, loss_rev, D_rev, grads_rev = reverse_sim(sys_fwd.coords, sys_fwd.velocities,
                    noise_seeds, γ, n_steps, n_steps_loss, n_steps_trunc, params, dt,
                    n_threads, sys_fwd.loggers, loss_accum_fwd, inter_type, loss_type)

# This will only be true if n_steps_trunc >= n_steps_loss
@test maximum(norm.(coords_start .- sys_rev.coords)) < (T == Float64 ? 1e-9 : 1e-4)
@test abs(loss_rev - loss_fwd) < 1e-7

println("Grads forward:        ", grads_fwd)
println("Grads FD:             ", grads_fd)
println("Grads reverse:        ", grads_rev)
println("Diff forward reverse: ", abs.(grads_fwd .- grads_rev))
println("%err forward reverse: ", 100 .* abs.(grads_fwd .- grads_rev) ./ abs.(grads_fwd))

function train(inter_type, params_start)
    loss_type = :mse
    γ = T(1.0)
    dt = T(0.001) # ps
    n_steps = 50_000
    n_steps_equil = 10_000
    n_steps_loss = 200
    n_steps_trunc = 200
    n_threads = Threads.nthreads()
    n_epochs = 10_000
    grad_clamp_val = T(1e3)
    learning_rate = T(5e-4)
    scale_params = true
    params = copy(params_start)
    if scale_params
        params_scaled = ones(T, length(params))
        opt = Optimisers.setup(Optimisers.Adam(learning_rate), params_scaled)
    else
        opt = Optimisers.setup(Optimisers.Adam(learning_rate), params)
    end
    epochs_loss, epochs_D = T[], T[]

    for epoch_n in 1:n_epochs
        noise_seeds = [rand_seed() for i in 1:n_steps]
        noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]

        sys_equil, _, _ = forward_sim(nothing, nothing, noise_equil_seeds, γ,
                                    n_steps_equil, n_steps_equil, params,
                                    dt, n_threads, inter_type, loss_type, false, false)
        coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

        sys_fwd, _, loss_accum_fwd, _ = forward_sim(
                coords_start, velocities_start, noise_seeds, γ, n_steps, n_steps_loss,
                params, dt, n_threads, inter_type, loss_type, false, true)

        sys_rev, loss_rev, D_rev, grads_rev = reverse_sim(
                sys_fwd.coords, sys_fwd.velocities, noise_seeds, γ, n_steps,
                n_steps_loss, n_steps_trunc, params, dt,
                n_threads, sys_fwd.loggers, loss_accum_fwd, inter_type, loss_type)

        push!(epochs_loss, loss_rev)
        push!(epochs_D, D_rev)

        isdir(out_dir) || mkdir(out_dir)
        open("$out_dir/train_$run_n.log", "a") do of
            println(
                of,
                "$epoch_n $loss_rev $D_rev ",
                join(params, " "), " ",
                join(grads_rev, " "),
            )
        end

        f = Figure()
        ax = f[1, 1] = Axis(f, xlabel="Epoch number", ylabel="Loss")
        lines!(1:epoch_n, epochs_loss)
        xlims!(0, epoch_n)
        save("$out_dir/plot_loss.pdf", f)

        f = Figure()
        ax = f[1, 1] = Axis(f, xlabel="Epoch number", ylabel="D / m^2 * s^-1")
        lines!(1:epoch_n, epochs_D)
        xlims!(0, epoch_n)
        save("$out_dir/plot_D.pdf", f)

        if scale_params
            scaled_grads = grads_rev .* params_start
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
                clamp.(grads_rev, -grad_clamp_val, grad_clamp_val),
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
    σ_gas, ϵ_gas = T(0.3297), T(0.438)
    params = [σ, ϵ, σ_gas, ϵ_gas, O_charge]

    inter_type = :lj
    train(inter_type, params)
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
    σ_gas, ϵ_gas = T(0.3297), T(0.438)
    params = [σ, ϵ, σ_gas, ϵ_gas, α, β, O_charge]

    inter_type = :dexp
    train(inter_type, params)
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
    A_gas, B_gas, C_gas = A, B, C
    params = [A, B, C, A_gas, B_gas, C_gas, O_charge]

    inter_type = :buck
    train(inter_type, params)
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
    σ_gas, ϵ_gas = T(0.3297), T(0.438)
    params = [σ, ϵ, α, σ_gas, ϵ_gas, λ, O_charge]

    inter_type = :ljsc
    train(inter_type, params)
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
    σ_gas, ϵ_gas = T(0.3297), T(0.438)
    params = [σ, ϵ, σ_gas, ϵ_gas, δ, γ, O_charge]

    inter_type = :buff
    train(inter_type, params)
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
for rep in 1:3
    reset_timer!(to)
    forward_sim(coords_start, velocities_start, noise_seeds, γ, 100, 100, params, dt, n_threads, inter_type, loss_type, false, false)
    println()
    show(to)
    println()
end
for rep in 1:3
    reset_timer!(to)
    reverse_sim(sys_fwd.coords, sys_fwd.velocities, noise_seeds, γ, 100, 100, 100, params, dt, n_threads, sys_fwd.loggers, loss_accum_fwd, inter_type, loss_type)
    println()
    show(to)
    println()
end
=#
#=
# Run longer simulations for validation (Figure 4B)
for inter_type in (:lj, :dexp, :buck, :ljsc)
    for rep_n in 1:5
        if inter_type == :lj
            σ, ϵ = T(0.3409279162545354), T(0.6893488104331025)
            σ_gas, ϵ_gas = T(0.3521234084056665), T(0.4713977679122796)
            O_charge = T(-0.8205720550135371)
            params = [σ, ϵ, σ_gas, ϵ_gas, O_charge]
        elseif inter_type == :dexp
            σ, ϵ = T(0.335672186281716), T(0.6782721489843005)
            σ_gas, ϵ_gas = T(0.34845149656580865), T(0.46603183706809853)
            α, β = T(17.843187982960053), T(4.722324542394384)
            O_charge = T(-0.8174543110135074)
            params = [σ, ϵ, σ_gas, ϵ_gas, α, β, O_charge]
        elseif inter_type == :buck
            A, B, C = T(405331.60242937884), T(37.020398787211334), T(0.002979073317511142)
            A_gas, B_gas, C_gas = T(397767.94880717236), T(37.601492297129695), T(0.0030011990505467347)
            O_charge = T(-0.9454276172884645)
            params = [A, B, C, A_gas, B_gas, C_gas, O_charge]
        elseif inter_type == :ljsc
            σ, ϵ = T(0.362004623082507), T(0.7278963003389142)
            σ_gas, ϵ_gas = T(0.11418818652495372), T(0.3820873887436166)
            α_ljsc, λ = T(0.3786753896602146), T(0.08645556453318586)
            O_charge = T(-0.8337707326423408)
            params = [σ, ϵ, σ_gas, ϵ_gas, α_ljsc, λ, O_charge]
        end

        γ = T(1.0)
        loss_type = :mse
        n_steps_equil = 10_000 # 10 ps
        n_steps       = 100_000 # 100 ps
        n_steps_loss  = 200
        dt = T(0.001) # ps
        n_threads = Threads.nthreads()
        noise_seeds = [rand_seed() for i in 1:n_steps]

        noise_equil_seeds = [rand_seed() for i in 1:n_steps_equil]
        sys_equil, _, _ = forward_sim(nothing, nothing, noise_equil_seeds, γ, n_steps_equil,
                            n_steps_equil, params, dt, n_threads, inter_type, loss_type, false, false)
        coords_start, velocities_start = copy(sys_equil.coords), copy(sys_equil.velocities)

        sys_fwd, loss_fwd, loss_accum_fwd, grads_fwd = forward_sim(coords_start, velocities_start,
                            noise_seeds, γ, n_steps, n_steps_loss, params, dt, n_threads, inter_type,
                            loss_type, false, true)

        loss_frequency = n_steps_loss * dt
        D = calculate_D(loss_accum_fwd, loss_frequency)
        D_conv = D * T(1e-6) # Convert from nm^2 * ps^-1 to m^2 * s^-1
        println(inter_type, " ", D_conv)
    end
end
=#
