#
# Boundary condition
#

"""
Abstract supertype for boundary conventions used by differential operators.
"""
abstract type AbstractBoundaryCondition end

"""
Forward-difference gradient with homogeneous Neumann boundary treatment.
"""
struct Neumann <: AbstractBoundaryCondition end

"""
Forward-difference gradient with periodic wraparound at every boundary.
"""
struct Periodic <: AbstractBoundaryCondition end

#
# Data fidelity
#

"""
Abstract supertype for data-fidelity terms.
"""
abstract type AbstractDataFidelity end

"""
L2 data fidelity, `0.5 * ||u - f||^2`.
"""
struct L2Fidelity <: AbstractDataFidelity end

"""
Poisson negative log-likelihood fidelity, `sum(u - f * log(u))` up to constants.
"""
struct PoissonFidelity <: AbstractDataFidelity end

#
# Primal constraints
#

"""
Abstract supertype for pointwise convex constraints on the primal variable `u`.
"""
abstract type AbstractPrimalConstraint end

"""
No additional primal constraint.
"""
struct NoConstraint <: AbstractPrimalConstraint end

"""
Pointwise non-negativity constraint, `u[i] >= 0`.
"""
struct NonnegativeConstraint <: AbstractPrimalConstraint end

"""
Pointwise box constraint, `lower <= u[i] <= upper`.
"""
struct BoxConstraint{T<:AbstractFloat} <: AbstractPrimalConstraint
    lower::T
    upper::T
    function BoxConstraint{T}(lower::T, upper::T) where {T<:AbstractFloat}
        isnan(lower) && throw(ArgumentError("lower must not be NaN"))
        isnan(upper) && throw(ArgumentError("upper must not be NaN"))
        lower <= upper || throw(
            ArgumentError("lower must be <= upper, got lower=$(lower), upper=$(upper)"),
        )
        return new{T}(lower, upper)
    end
end

function BoxConstraint(lower::Real, upper::Real)
    T = promote_type(typeof(float(lower)), typeof(float(upper)))
    lower_t = T(lower)
    upper_t = T(upper)
    return BoxConstraint{T}(lower_t, upper_t)
end

#
# TVMode (pixel/voxel) isotropy
#

"""
Abstract supertype for TV penalty modes.
"""
abstract type AbstractTVMode end

"""
Isotropic TV, `sum(sqrt(sum_d (grad_d u)^2))`.
"""
struct IsotropicTV <: AbstractTVMode end

"""
Anisotropic TV, `sum(sum_d |grad_d u|)`.
"""
struct AnisotropicTV <: AbstractTVMode end

"""
Overlapping group-sparse anisotropic TV.

`group_shape` may be a positive integer, expanded across all spatial axes by
`TVProblem`, or a tuple with one positive entry per spatial axis. Groups are
applied separately to each directional derivative. A group shape of one along
every axis is equivalent to [`AnisotropicTV`](@ref).
"""
struct GroupSparseTV{S} <: AbstractTVMode
    group_shape::S
end

function GroupSparseTV(group_size::Integer = 3)
    group_size > 0 || throw(ArgumentError("group_size must be positive, got $group_size"))
    return GroupSparseTV{Int}(Int(group_size))
end

function GroupSparseTV(group_shape::Tuple{Vararg{Integer,N}}) where {N}
    N > 0 || throw(ArgumentError("group_shape must not be empty"))
    normalized = ntuple(d -> Int(group_shape[d]), Val(N))
    @inbounds for d = 1:N
        normalized[d] > 0 ||
            throw(ArgumentError("group_shape[$d] must be positive, got $(group_shape[d])"))
    end
    return GroupSparseTV{typeof(normalized)}(normalized)
end

#
# Solver
#
"""
Abstract supertype for all total-variation solver configurations.
"""
abstract type AbstractTVSolver end

"""
Convergence summary for one solver run.
"""
struct SolverStats{T<:AbstractFloat}
    iterations::Int
    converged::Bool
    rel_change::T
end

function _validate_common_config(maxiter::Int, check_every::Int)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive, got $maxiter"))
    check_every > 0 ||
        throw(ArgumentError("check_every must be positive, got $check_every"))
    return nothing
end
