"""
In-place divergence operator that is negative-adjoint consistent with `gradient!`
under the same boundary convention.
"""
function divergence!(
    out::AbstractArray{T,N},
    p::NTuple{N,AT},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T,N,S<:Real,AT<:AbstractArray{T,N}}
    fill!(out, zero(T))

    @views @inbounds for d = 1:N
        pd = p[d]
        n_d = size(out, d)
        n_d < 2 && continue # Skip dimensions with size 1
        scale = T(inv_spacing[d])

        # 1. The "Interior" and "Forward" contribution:
        # For a backward difference (adjoint of forward diff),
        # add p[i] to out[i] for all i from 1 to n-1.
        view_out_start = selectdim(out, d, 1:(n_d-1))
        view_p_start = selectdim(pd, d, 1:(n_d-1))
        @. view_out_start = view_out_start + scale * view_p_start

        # 2. The "Backward" contribution:
        # subtract p[i] from out[i+1] for all i from 1 to n-1.
        view_out_end = selectdim(out, d, 2:n_d)
        @. view_out_end = view_out_end - scale * view_p_start
    end

    return out
end

"""
In-place negative adjoint of the periodic forward-difference gradient.
"""
function divergence!(
    out::AbstractArray{T,N},
    p::NTuple{N,AT},
    ::Periodic,
    inv_spacing::NTuple{N,S},
) where {T,N,S<:Real,AT<:AbstractArray{T,N}}
    fill!(out, zero(T))

    @views @inbounds for d = 1:N
        pd = p[d]
        n_d = size(out, d)
        n_d == 1 && continue
        scale = T(inv_spacing[d])

        out_first = selectdim(out, d, 1)
        p_first = selectdim(pd, d, 1)
        p_last = selectdim(pd, d, n_d)
        @. out_first = out_first + scale * (p_first - p_last)

        out_rest = selectdim(out, d, 2:n_d)
        p_rest = selectdim(pd, d, 2:n_d)
        p_previous = selectdim(pd, d, 1:(n_d-1))
        @. out_rest = out_rest + scale * (p_rest - p_previous)
    end

    return out
end

function divergence!(
    out::AbstractArray{T,N},
    p::NTuple{N,AT},
    boundary::Neumann,
) where {T,N,AT}
    return divergence!(out, p, boundary, ntuple(_ -> one(T), Val(N)))
end


function divergence!(
    out::AbstractArray{T,N},
    p::NTuple{N,AT},
    boundary::Periodic,
) where {T,N,AT}
    return divergence!(out, p, boundary, ntuple(_ -> one(T), Val(N)))
end
