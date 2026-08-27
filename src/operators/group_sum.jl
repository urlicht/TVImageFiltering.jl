@inline function _group_offset_bounds(group_size::Int)
    return -fld(group_size - 1, 2), fld(group_size, 2)
end

function _group_sum_axis!(
    out::AbstractArray{T,N},
    input::AbstractArray{T,N},
    d::Int,
    first_offset::Int,
    last_offset::Int,
    ::Neumann,
) where {T,N}
    fill!(out, zero(T))
    n_d = size(input, d)

    @views @inbounds for offset = first_offset:last_offset
        if offset >= 0
            destination = selectdim(out, d, 1:(n_d-offset))
            source = selectdim(input, d, (1+offset):n_d)
        else
            shift = -offset
            destination = selectdim(out, d, (1+shift):n_d)
            source = selectdim(input, d, 1:(n_d-shift))
        end
        @. destination = destination + source
    end

    return out
end

function _group_sum_axis!(
    out::AbstractArray{T,N},
    input::AbstractArray{T,N},
    d::Int,
    first_offset::Int,
    last_offset::Int,
    ::Periodic,
) where {T,N}
    fill!(out, zero(T))
    n_d = size(input, d)

    @views @inbounds for offset = first_offset:last_offset
        if offset >= 0
            destination = selectdim(out, d, 1:(n_d-offset))
            source = selectdim(input, d, (1+offset):n_d)
            @. destination = destination + source

            if offset > 0
                wrapped_destination = selectdim(out, d, (n_d-offset+1):n_d)
                wrapped_source = selectdim(input, d, 1:offset)
                @. wrapped_destination = wrapped_destination + wrapped_source
            end
        else
            shift = -offset
            destination = selectdim(out, d, (1+shift):n_d)
            source = selectdim(input, d, 1:(n_d-shift))
            @. destination = destination + source

            wrapped_destination = selectdim(out, d, 1:shift)
            wrapped_source = selectdim(input, d, (n_d-shift+1):n_d)
            @. wrapped_destination = wrapped_destination + wrapped_source
        end
    end

    return out
end

function _validate_group_sum_buffers(
    out::AbstractArray{T,N},
    scratch::AbstractArray{T,N},
    input::AbstractArray{T,N},
    group_shape::NTuple{N,Int},
) where {T,N}
    size(out) == size(input) ||
        throw(ArgumentError("out and input must have matching sizes"))
    size(scratch) == size(input) ||
        throw(ArgumentError("scratch and input must have matching sizes"))
    out === scratch && throw(ArgumentError("out and scratch must be distinct arrays"))
    out === input && throw(ArgumentError("out and input must be distinct arrays"))
    scratch === input && throw(ArgumentError("scratch and input must be distinct arrays"))
    @inbounds for d = 1:N
        1 <= group_shape[d] <= size(input, d) || throw(
            ArgumentError(
                "group_shape[$d]=$(group_shape[d]) must be in 1:$(size(input, d))",
            ),
        )
    end
    return nothing
end

"""
Apply the separable overlapping-group sum to `input`.

At each output index, values from the centered group described by
`group_shape` are summed. `Neumann()` truncates groups at the domain boundary;
`Periodic()` wraps them. `out`, `scratch`, and `input` must be distinct arrays
with matching sizes.
"""
function group_sum!(
    out::AbstractArray{T,N},
    scratch::AbstractArray{T,N},
    input::AbstractArray{T,N},
    group_shape::NTuple{N,Int},
    boundary::AbstractBoundaryCondition,
) where {T,N}
    _validate_group_sum_buffers(out, scratch, input, group_shape)

    source = input
    @inbounds for d = 1:N
        target = isodd(d) ? out : scratch
        first_offset, last_offset = _group_offset_bounds(group_shape[d])
        _group_sum_axis!(target, source, d, first_offset, last_offset, boundary)
        source = target
    end
    source === out || copyto!(out, source)
    return out
end

"""
Apply the Euclidean adjoint of [`group_sum!`](@ref).
"""
function group_sum_adjoint!(
    out::AbstractArray{T,N},
    scratch::AbstractArray{T,N},
    input::AbstractArray{T,N},
    group_shape::NTuple{N,Int},
    boundary::AbstractBoundaryCondition,
) where {T,N}
    _validate_group_sum_buffers(out, scratch, input, group_shape)

    source = input
    step = 0
    @inbounds for d = N:-1:1
        step += 1
        target = isodd(step) ? out : scratch
        first_offset, last_offset = _group_offset_bounds(group_shape[d])
        _group_sum_axis!(target, source, d, -last_offset, -first_offset, boundary)
        source = target
    end
    source === out || copyto!(out, source)
    return out
end
