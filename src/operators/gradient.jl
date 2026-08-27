"""
In-place forward finite-difference gradient with Neumann boundary treatment.

For each dimension `d`,
`g[d][i] = (u[i + e_d] - u[i]) / spacing[d]` on interior points and `0` on the
upper boundary.
"""
function gradient!(
    g::NTuple{N,AT},
    u::AbstractArray{T,N},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T,N,S<:Real,AT<:AbstractArray{T,N}}
    # @views ensures all slicing uses view instead of copy
    @views @inbounds for d = 1:N
        gd = g[d]
        n_d = size(u, d)
        scale = T(inv_spacing[d])

        # 1. interior differences
        target = selectdim(gd, d, 1:(n_d-1))
        u_front = selectdim(u, d, 1:(n_d-1))
        u_back = selectdim(u, d, 2:n_d)

        @. target = scale * (u_back - u_front)

        # 2. Neumann boundary to zero
        selectdim(gd, d, n_d) .= zero(T)
    end

    return g
end

"""
In-place forward finite-difference gradient with periodic wraparound.
"""
function gradient!(
    g::NTuple{N,AT},
    u::AbstractArray{T,N},
    ::Periodic,
    inv_spacing::NTuple{N,S},
) where {T,N,S<:Real,AT<:AbstractArray{T,N}}
    @views @inbounds for d = 1:N
        gd = g[d]
        n_d = size(u, d)
        scale = T(inv_spacing[d])

        if n_d == 1
            fill!(gd, zero(T))
            continue
        end

        target = selectdim(gd, d, 1:(n_d-1))
        u_front = selectdim(u, d, 1:(n_d-1))
        u_back = selectdim(u, d, 2:n_d)
        @. target = scale * (u_back - u_front)

        boundary_target = selectdim(gd, d, n_d)
        u_first = selectdim(u, d, 1)
        u_last = selectdim(u, d, n_d)
        @. boundary_target = scale * (u_first - u_last)
    end

    return g
end

function gradient!(
    g::NTuple{N,AT},
    u::AbstractArray{T,N},
    boundary::Neumann,
) where {T,N,AT<:AbstractArray{T,N}}
    return gradient!(g, u, boundary, ntuple(_ -> one(T), Val(N)))
end


function gradient!(
    g::NTuple{N,AT},
    u::AbstractArray{T,N},
    boundary::Periodic,
) where {T,N,AT<:AbstractArray{T,N}}
    return gradient!(g, u, boundary, ntuple(_ -> one(T), Val(N)))
end
