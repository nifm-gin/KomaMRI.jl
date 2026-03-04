# module KomaSimWithFieldMapDict
# @info "FieldMapDict VERSION LOADED"

# using Interpolations
# using KomaMRI
# import KomaMRI: get_spin_coords
export run_spin_precession!
#export BlochFieldmapDict, FieldmapArray, build_fieldmap_interpolant, 
# @info "FieldMapDict VERSION LOADED"

const γ = 2π * 42.57747892e6  # rad/s/T

# -----------------------------
# Fieldmap container & interpolation
# -----------------------------
struct FieldmapArray
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    arr::Array{Float64,3}
    interp::Any
end

# function build_fieldmap_interpolant(x, y, z, arr)
#     using Interpolations
#     itp = Interpolations.interpolate((x, y, z), arr, Interpolations.Gridded(Interpolations.Linear()))
#     FieldmapArray(x, y, z, arr, itp)
# end

# function build_fieldmap_interpolant(x, y, z, arr)
#     itp = Interpolations.interpolate((x, y, z), arr, Interpolations.Gridded(Interpolations.Linear()))

#     return FieldmapArray(x, y, z, arr, itp)
# end

function sample_fieldmap(fm::FieldmapArray, xs::Vector, ys::Vector, zs::Vector)
    fm.interp.(xs, ys, zs)
end

# -----------------------------
# SimulationMethod
# -----------------------------
struct BlochFieldmapDict <: SimulationMethod
    fieldmap::FieldmapArray
    # overwrite_deltaBz::Bool
end

# Optional: show method
# Base.show(io::IO, s::BlochFieldmapDict) = print(io, "BlochFieldmapDict(overwrite=$(s.overwrite_deltaBz))")

# -----------------------------
# Output dimension
# -----------------------------
function sim_output_dim(obj::Phantom{T}, seq::Sequence, sys::Scanner, sim_method::BlochFieldmapDict) where {T<:Real}
    # Only one output channel for Mxy
    return (sum(seq.ADC.N), length(obj), 1)
end

function run_spin_precession!(
    p::Phantom{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag{T},
    sim_method::BlochFieldmapDict,
    groupsize,
    backend::KA.CPU,
    prealloc::BlochCPUPrealloc
) where {T<:Real}
    #Simulation
    #Motion
    println("Using the new function")
    x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, seq.t[1])
    
    #Initialize arrays
    Bz_old = prealloc.Bz_old
    Bz_new = prealloc.Bz_new
    ϕ = prealloc.ϕ
    Mxy = prealloc.M.xy
    ΔBz = prealloc.ΔBz
    fill!(ϕ, zero(T))
    fm0_hz = sample_fieldmap(sim_method.fieldmap, x, y, z)  # sample initial fieldmap
    @. ΔBz = 2π * fm0_hz / γ                               # convert to Tesla (or rad/s)

    @. Bz_old = x[:,1] * seq.Gx[1] + y[:,1] * seq.Gy[1] + z[:,1] * seq.Gz[1] + ΔBz

    # Fill sig[1] if needed
    ADC_idx = 1
    if (seq.ADC[1])
        sig[1] = sum(M.xy)
        ADC_idx += 1
    end

    t_seq = zero(T) # Time
    for seq_idx=2:length(seq.t)
        x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, seq.t[seq_idx])

        fm_hz = sample_fieldmap(sim_method.fieldmap, x, y, z)
        @. ΔBz = 2π * fm_hz / γ  # overwrite with local B0

        t_seq += seq.Δt[seq_idx-1]

        #Effective Field
        @. Bz_new = x * seq.Gx[seq_idx] + y * seq.Gy[seq_idx] + z * seq.Gz[seq_idx] + ΔBz
        
        #Rotation
        @. ϕ += (Bz_old + Bz_new) * T(-π * γ) * seq.Δt[seq_idx-1]

        #Acquired Signal
        if seq_idx <= length(seq.ADC) && seq.ADC[seq_idx]
            @. Mxy = exp(-t_seq / p.T2) * M.xy * cis(ϕ)

            #Reset Spin-State (Magnetization). Only for FlowPath
            outflow_spin_reset!(Mxy, seq.t[seq_idx], p.motion)

            sig[ADC_idx] = sum(Mxy) 
            ADC_idx += 1
        end

        Bz_old, Bz_new = Bz_new, Bz_old
    end

    #Final Spin-State
    @. M.xy = M.xy * exp(-t_seq / p.T2) * cis(ϕ)
    @. M.z = M.z * exp(-t_seq / p.T1) + p.ρ * (T(1) - exp(-t_seq / p.T1))
    
    #Reset Spin-State (Magnetization). Only for FlowPath
    outflow_spin_reset!(M,  seq.t', p.motion; replace_by=p.ρ)

    return nothing
end

# end # module

