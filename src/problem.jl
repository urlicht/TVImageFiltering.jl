"""
TV-based reconstruction/denoising
`min_u data_fidelity(u, f) + lambda * TV(u)`

`spacing[d]` is the physical grid spacing along axis `d` used by differential operators.

`TVProblem` is immutable, but `f` is stored by reference.
For repeated solves on same shape/eltype data, mutate `f` in place (for example,
`copyto!(problem.f, new_f)`) and reuse the same `TVProblem`.
If you need a different `f` array object or different metadata (`lambda`,
`spacing`, `data_fidelity`, `tv_mode`, `boundary`, `constraint`), construct a
new `TVProblem`.
"""
struct TVProblem{
    T<:AbstractFloat,
    N,
    AF<:AbstractArray{T,N},
    DF<:AbstractDataFidelity,
    TV<:AbstractTVMode,
    BC<:AbstractBoundaryCondition,
    PC<:AbstractPrimalConstraint,
}
    f::AF
    lambda::T
    spacing::NTuple{N,T}
    data_fidelity::DF
    tv_mode::TV
    boundary::BC
    constraint::PC
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::Nothing,
) where {T<:AbstractFloat,N}
    return ntuple(_ -> one(T), Val(N))
end

function _normalize_spacing(::Type{T}, ::Val{N}, spacing::Real) where {T<:AbstractFloat,N}
    scale = T(spacing)
    isfinite(scale) || throw(ArgumentError("spacing must be finite"))
    scale > zero(T) || throw(ArgumentError("spacing must be positive, got $spacing"))
    return ntuple(_ -> scale, Val(N))
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::NTuple{N,<:Real},
) where {T<:AbstractFloat,N}
    scales = ntuple(d -> T(spacing[d]), Val(N))
    @inbounds for d = 1:N
        isfinite(scales[d]) || throw(ArgumentError("spacing[$d] must be finite"))
        scales[d] > zero(T) ||
            throw(ArgumentError("spacing[$d] must be positive, got $(spacing[d])"))
    end
    return scales
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::AbstractVector{<:Real},
) where {T<:AbstractFloat,N}
    length(spacing) == N || throw(
        ArgumentError("spacing length must match ndims(f)=$N, got $(length(spacing))"),
    )
    scales = ntuple(d -> T(spacing[d]), Val(N))
    @inbounds for d = 1:N
        isfinite(scales[d]) || throw(ArgumentError("spacing[$d] must be finite"))
        scales[d] > zero(T) ||
            throw(ArgumentError("spacing[$d] must be positive, got $(spacing[d])"))
    end
    return scales
end

function _normalize_spacing(::Type{T}, ::Val{N}, spacing) where {T<:AbstractFloat,N}
    throw(
        ArgumentError(
            "spacing must be `nothing`, a real scalar, an NTuple{$N,<:Real}, or an AbstractVector{<:Real}",
        ),
    )
end

function _normalize_constraint(::Type{T}, constraint::NoConstraint) where {T<:AbstractFloat}
    return constraint
end

function _normalize_tv_mode(
    ::Val{N},
    shape::NTuple{N,Int},
    tv_mode::AbstractTVMode,
) where {N}
    return tv_mode
end

function _validate_group_shape(group_shape::NTuple{N,Int}, shape::NTuple{N,Int}) where {N}
    @inbounds for d = 1:N
        group_shape[d] > 0 ||
            throw(ArgumentError("group_shape[$d] must be positive, got $(group_shape[d])"))
        group_shape[d] <= shape[d] || throw(
            ArgumentError(
                "group_shape[$d]=$(group_shape[d]) must not exceed size(f, $d)=$(shape[d])",
            ),
        )
    end
    return group_shape
end

function _normalize_tv_mode(
    ::Val{N},
    shape::NTuple{N,Int},
    tv_mode::GroupSparseTV{<:Integer},
) where {N}
    group_shape = ntuple(_ -> Int(tv_mode.group_shape), Val(N))
    _validate_group_shape(group_shape, shape)
    return GroupSparseTV(group_shape)
end

function _normalize_tv_mode(
    ::Val{N},
    shape::NTuple{N,Int},
    tv_mode::GroupSparseTV{<:Tuple},
) where {N}
    length(tv_mode.group_shape) == N || throw(
        ArgumentError(
            "group_shape length must match ndims(f)=$N, got $(length(tv_mode.group_shape))",
        ),
    )
    group_shape = ntuple(d -> Int(tv_mode.group_shape[d]), Val(N))
    _validate_group_shape(group_shape, shape)
    return GroupSparseTV(group_shape)
end

function _normalize_constraint(
    ::Type{T},
    constraint::NonnegativeConstraint,
) where {T<:AbstractFloat}
    return constraint
end

function _normalize_constraint(
    ::Type{T},
    constraint::BoxConstraint,
) where {T<:AbstractFloat}
    lower_t = T(constraint.lower)
    upper_t = T(constraint.upper)
    return BoxConstraint(lower_t, upper_t)
end

function _normalize_constraint(
    ::Type{T},
    constraint::AbstractPrimalConstraint,
) where {T<:AbstractFloat}
    throw(
        ArgumentError(
            "unsupported constraint type $(typeof(constraint)); supported: NoConstraint, NonnegativeConstraint, BoxConstraint",
        ),
    )
end

function TVProblem(
    f::AbstractArray{T,N};
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
) where {T<:AbstractFloat,N}
    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    spacing_t = _normalize_spacing(T, Val(N), spacing)
    constraint_t = _normalize_constraint(T, constraint)
    tv_mode_t = _normalize_tv_mode(Val(N), size(f), tv_mode)
    return TVProblem{
        T,
        N,
        typeof(f),
        typeof(data_fidelity),
        typeof(tv_mode_t),
        typeof(boundary),
        typeof(constraint_t),
    }(
        f,
        lambda_t,
        spacing_t,
        data_fidelity,
        tv_mode_t,
        boundary,
        constraint_t,
    )
end
