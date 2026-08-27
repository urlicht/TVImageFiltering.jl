"""
Configuration for the PDHG / Chambolle-Pock primal-dual solver.

This solver targets TV-regularized models of the form

    minimize_u D(u, f) + lambda * TV(u) + I_C(u)

where:
- `D` is either `L2Fidelity` or `PoissonFidelity`,
- `C` is a pointwise convex set from `NoConstraint()`,
  `NonnegativeConstraint()`, or `BoxConstraint(lower, upper)`.

The primal-dual step sizes must satisfy:

    tau * sigma * ||grad||^2 < 1

This implementation uses the conservative bound

    ||grad||^2 <= 4 * sum_{d: n_d > 1} h_d^(-2)

with `h_d = spacing[d]` and `n_d = size(f, d)`.

Fields:
- `maxiter`: maximum number of iterations.
- `tau`: primal step size.
- `sigma`: dual step size.
- `theta`: over-relaxation in `[0, 1]`.
- `tol`: stopping tolerance on `max(relative_primal_change, primal_dual_residual)`.
- `check_every`: evaluate convergence every `check_every` iterations.

References:
- A. Chambolle and T. Pock, "A First-Order Primal-Dual Algorithm for Convex
  Problems with Applications to Imaging," *JMIV* 40:120-145, 2011.
  DOI: 10.1007/s10851-010-0251-1
"""
struct PDHGConfig{T<:AbstractFloat} <: AbstractTVSolver
    maxiter::Int
    tau::T
    sigma::T
    theta::T
    tol::T
    check_every::Int
end

function PDHGConfig(;
    maxiter::Int = 500,
    tau::Real = 0.25,
    sigma::Real = 0.25,
    theta::Real = 1.0,
    tol::Real = 1e-4,
    check_every::Int = 10,
)
    T = promote_type(
        typeof(float(tau)),
        typeof(float(sigma)),
        typeof(float(theta)),
        typeof(float(tol)),
    )
    return PDHGConfig{T}(maxiter, T(tau), T(sigma), T(theta), T(tol), check_every)
end

function _validate(config::PDHGConfig)
    _validate_common_config(config.maxiter, config.check_every)
    config.tau > zero(config.tau) || throw(ArgumentError("tau must be positive"))
    config.sigma > zero(config.sigma) || throw(ArgumentError("sigma must be positive"))
    config.theta >= zero(config.theta) ||
        throw(ArgumentError("theta must be in [0, 1], got $(config.theta)"))
    config.theta <= one(config.theta) ||
        throw(ArgumentError("theta must be in [0, 1], got $(config.theta)"))
    config.tol >= zero(config.tol) || throw(ArgumentError("tol must be non-negative"))
    return nothing
end

"""
Reusable workspace for the PDHG / Chambolle-Pock solver.

`PDHGState` serves two roles:
- scratch buffers (`u`, `u_prev`, `u_bar`, `divp`, `primal_tmp`, `grad_u_bar`)
  reused every call,
- warm-start storage for the dual variable `p`.

When the same `state` object is passed to repeated `solve!` calls, `p` is not
reset between calls.
"""
struct PDHGState{T<:AbstractFloat,N,A<:AbstractArray{T,N},G<:NTuple{N,A}}
    u::A
    u_prev::A
    u_bar::A
    divp::A
    primal_tmp::A
    p::G
    p_prev::G
    grad_u_bar::G
end

function PDHGState(reference::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    u = similar(reference)
    u_prev = similar(reference)
    u_bar = similar(reference)
    divp = similar(reference)
    primal_tmp = similar(reference)
    p = allocate_dual(reference)
    p_prev = allocate_dual(reference)
    grad_u_bar = allocate_dual(reference)

    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(u_bar, zero(T))
    fill!(divp, zero(T))
    fill!(primal_tmp, zero(T))
    @inbounds for d = 1:N
        fill!(p[d], zero(T))
        fill!(p_prev[d], zero(T))
        fill!(grad_u_bar[d], zero(T))
    end

    return PDHGState{T,N,typeof(u),typeof(p)}(
        u,
        u_prev,
        u_bar,
        divp,
        primal_tmp,
        p,
        p_prev,
        grad_u_bar,
    )
end

function _pdhg_operator_norm_sq_upper_bound(
    inv_spacing::NTuple{N,T},
    shape::NTuple{N,Int},
) where {N,T<:AbstractFloat}
    lipschitz_term = zero(T)
    @inbounds for d = 1:N
        if shape[d] > 1
            lipschitz_term += abs2(inv_spacing[d])
        end
    end
    return T(4) * lipschitz_term
end

function _validate_poisson_data(f::AbstractArray{T}) where {T<:AbstractFloat}
    any(.!isfinite.(f)) && throw(ArgumentError("PoissonFidelity requires finite observation data f"))
    any(f .< zero(T)) &&
        throw(ArgumentError("PoissonFidelity requires non-negative observation data f"))
    return nothing
end

function _constraint_bounds(::Type{T}, ::NoConstraint) where {T<:AbstractFloat}
    return -T(Inf), T(Inf)
end

function _constraint_bounds(::Type{T}, ::NonnegativeConstraint) where {T<:AbstractFloat}
    return zero(T), T(Inf)
end

function _constraint_bounds(::Type{T}, constraint::BoxConstraint) where {T<:AbstractFloat}
    return T(constraint.lower), T(constraint.upper)
end

function _constraint_bounds(
    ::Type{T},
    constraint::AbstractPrimalConstraint,
) where {T<:AbstractFloat}
    throw(
        ArgumentError(
            "unsupported constraint type $(typeof(constraint)); supported: NoConstraint, NonnegativeConstraint, BoxConstraint",
        ),
    )
end

function _project_interval!(
    u::AbstractArray{T},
    lower::T,
    upper::T,
) where {T<:AbstractFloat}
    neg_inf = -T(Inf)
    pos_inf = T(Inf)

    if (lower == neg_inf) && (upper == pos_inf)
        return nothing
    elseif upper == pos_inf
        @. u = ifelse(u < lower, lower, u)
    elseif lower == neg_inf
        @. u = ifelse(u > upper, upper, u)
    else
        @. u = clamp(u, lower, upper)
    end

    return nothing
end

function _pdhg_primal_bounds(problem::TVProblem{T,N}) where {T<:AbstractFloat,N}
    lower, upper = _constraint_bounds(T, problem.constraint)
    if problem.data_fidelity isa PoissonFidelity
        lower = max(lower, zero(T))
    end
    return lower, upper
end

function _validate_pdhg_constraint(problem::TVProblem{T,N}) where {T<:AbstractFloat,N}
    lower, upper = _constraint_bounds(T, problem.constraint)
    lower <= upper || throw(
        ArgumentError(
            "constraint lower bound must be <= upper bound, got [$lower, $upper]",
        ),
    )

    if problem.data_fidelity isa PoissonFidelity
        upper >= zero(T) || throw(
            ArgumentError(
                "PoissonFidelity with constraint [$lower, $upper] is infeasible; upper bound must be >= 0",
            ),
        )
        if (upper == zero(T)) && any(problem.f .> zero(T))
            throw(
                ArgumentError(
                    "PoissonFidelity with upper bound 0 is infeasible when observation data contains positive values",
                ),
            )
        end
    end

    return nothing
end

function _prox_data!(
    out::AbstractArray{T},
    v::AbstractArray{T},
    f::AbstractArray{T},
    tau::T,
    ::L2Fidelity,
) where {T<:AbstractFloat}
    denom = one(T) + tau
    @. out = (v + tau * f) / denom
    return nothing
end

function _prox_data!(
    out::AbstractArray{T},
    v::AbstractArray{T},
    f::AbstractArray{T},
    tau::T,
    ::PoissonFidelity,
) where {T<:AbstractFloat}
    four_tau = T(4) * tau
    two = T(2)
    @. out = (v - tau + sqrt((v - tau) * (v - tau) + four_tau * f)) / two
    @. out = ifelse(out < zero(T), zero(T), out)
    return nothing
end

function _prox_data!(
    out::AbstractArray,
    v::AbstractArray,
    f::AbstractArray,
    tau,
    data_fidelity::AbstractDataFidelity,
)
    throw(
        ArgumentError(
            "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
        ),
    )
end

function _pdhg_data_minimizer!(
    u::AbstractArray{T},
    f::AbstractArray{T},
    data_fidelity::AbstractDataFidelity,
    lower::T,
    upper::T,
) where {T<:AbstractFloat}
    if data_fidelity isa L2Fidelity
        copyto!(u, f)
    elseif data_fidelity isa PoissonFidelity
        copyto!(u, f)
    else
        throw(
            ArgumentError(
                "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
            ),
        )
    end

    _project_interval!(u, lower, upper)
    return nothing
end

function _validate_pdhg_data_fidelity(problem::TVProblem)
    if problem.data_fidelity isa PoissonFidelity
        _validate_poisson_data(problem.f)
    elseif !(problem.data_fidelity isa L2Fidelity)
        throw(
            ArgumentError(
                "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(problem.data_fidelity))",
            ),
        )
    end
    return nothing
end

function _validate_state_shape(
    state::PDHGState{T,N},
    shape::NTuple{N,Int},
) where {T<:AbstractFloat,N}
    size(state.u) == shape ||
        throw(ArgumentError("state.u size must match solve buffer size $shape"))
    size(state.u_prev) == shape ||
        throw(ArgumentError("state.u_prev size must match solve buffer size $shape"))
    size(state.u_bar) == shape ||
        throw(ArgumentError("state.u_bar size must match solve buffer size $shape"))
    size(state.divp) == shape ||
        throw(ArgumentError("state.divp size must match solve buffer size $shape"))
    size(state.primal_tmp) == shape ||
        throw(ArgumentError("state.primal_tmp size must match solve buffer size $shape"))

    @inbounds for d = 1:N
        size(state.p[d]) == shape ||
            throw(ArgumentError("state.p[$d] size must match solve buffer size $shape"))
        size(state.p_prev[d]) == shape || throw(
            ArgumentError("state.p_prev[$d] size must match solve buffer size $shape"),
        )
        size(state.grad_u_bar[d]) == shape || throw(
            ArgumentError("state.grad_u_bar[$d] size must match solve buffer size $shape"),
        )
    end
    return nothing
end

function _pdhg_relative_residual!(
    state::PDHGState{T,N},
    problem::TVProblem{T,N},
    tau::T,
    sigma::T,
    inv_spacing::NTuple{N,T},
) where {T<:AbstractFloat,N}
    @inbounds for d = 1:N
        @. state.grad_u_bar[d] = state.p_prev[d] - state.p[d]
    end

    divergence!(state.primal_tmp, state.grad_u_bar, problem.boundary, inv_spacing)

    @. state.divp = state.u_prev - state.u

    @. state.primal_tmp = state.divp / tau + state.primal_tmp
    primal_res_norm = T(sqrt(sum(abs2, state.primal_tmp)))

    gradient!(state.grad_u_bar, state.divp, problem.boundary, inv_spacing)
    dual_res_norm2 = zero(T)
    @inbounds for d = 1:N
        ad = state.grad_u_bar[d]
        @. ad = (state.p_prev[d] - state.p[d]) / sigma - ad
        dual_res_norm2 += sum(abs2, ad)
    end

    dual_res_norm = T(sqrt(dual_res_norm2))
    nscale = sqrt(T(length(state.u)))
    return max(primal_res_norm, dual_res_norm) / max(nscale, one(T))
end

"""
Run TV denoising/reconstruction in place using PDHG / Chambolle-Pock.

`u` is both the initial guess and output buffer.

State and buffer reuse:
- Reuse `u` and `state` across calls to avoid allocations.
- `u` is copied into solver state and acts as primal warm start.
- `state.p` is reused as dual warm start and is not reset by `solve!`.
- For new image data with the same array storage, update in place with
  `copyto!(problem.f, new_f)` and call `solve!` again.
- If image array object, shape/eltype, or problem metadata changes, create a
  new `TVProblem` (and a new `PDHGState` only when shape/eltype changes).

Stopping criterion:
- `solve!` checks both relative primal change and a PDHG primal-dual residual.
- The primal-dual residual is normalized by `sqrt(length(u))`.
- Convergence uses `max(relative_primal_change, primal_dual_residual) <= tol`.

Constraints:
- If `problem.constraint` is not `NoConstraint()`, the primal update applies the
  exact pointwise constrained proximal map for the supported fidelities.
"""
function solve!(
    u::AbstractArray{T,N},
    problem::TVProblem{T,N},
    config::PDHGConfig;
    state::Union{Nothing,PDHGState{T,N}} = nothing,
) where {T<:AbstractFloat,N}
    _validate(config)
    problem.tv_mode isa GroupSparseTV &&
        throw(ArgumentError("PDHG does not support GroupSparseTV; use GSTVConfig"))
    (problem.boundary isa Neumann || problem.boundary isa Periodic) ||
        throw(ArgumentError("PDHG supports only Neumann and Periodic boundaries"))
    _validate_pdhg_data_fidelity(problem)
    _validate_pdhg_constraint(problem)
    size(u) == size(problem.f) ||
        throw(ArgumentError("You must have the same size as problem.f"))
    problem.lambda >= zero(T) ||
        throw(ArgumentError("lambda must be non-negative, got $(problem.lambda)"))

    primal_lower, primal_upper = _pdhg_primal_bounds(problem)

    local_state = state === nothing ? nothing : state
    local_state === nothing || _validate_state_shape(local_state, size(u))

    if problem.lambda == zero(T)
        _pdhg_data_minimizer!(
            u,
            problem.f,
            problem.data_fidelity,
            primal_lower,
            primal_upper,
        )
        return SolverStats{T}(0, true, zero(T))
    end

    local_state = local_state === nothing ? PDHGState(problem.f) : local_state

    tau_t = T(config.tau)
    sigma_t = T(config.sigma)
    theta_t = T(config.theta)

    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(N))
    op_norm_sq = _pdhg_operator_norm_sq_upper_bound(inv_spacing, size(problem.f))
    if op_norm_sq > zero(T)
        tau_t * sigma_t * op_norm_sq < one(T) || throw(
            ArgumentError(
                "tau * sigma * ||grad||^2 must be < 1; with this spacing/shape, use tau*sigma < $(inv(op_norm_sq)), got $(tau_t * sigma_t)",
            ),
        )
    end

    copyto!(local_state.u, u)
    copyto!(local_state.u_bar, u)

    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        @inbounds for d = 1:N
            copyto!(local_state.p_prev[d], local_state.p[d])
        end

        gradient!(local_state.grad_u_bar, local_state.u_bar, problem.boundary, inv_spacing)
        @inbounds for d = 1:N
            @. local_state.p[d] = local_state.p[d] + sigma_t * local_state.grad_u_bar[d]
        end
        project_dual_ball!(local_state.p, problem.lambda, problem.tv_mode)

        divergence!(local_state.divp, local_state.p, problem.boundary, inv_spacing)
        @. local_state.primal_tmp = local_state.u + tau_t * local_state.divp
        _prox_data!(
            local_state.u,
            local_state.primal_tmp,
            problem.f,
            tau_t,
            problem.data_fidelity,
        )
        _project_interval!(local_state.u, primal_lower, primal_upper)
        @. local_state.u_bar =
            local_state.u + theta_t * (local_state.u - local_state.u_prev)

        if (k % config.check_every == 0) || (k == config.maxiter)
            primal_rel_change = _relative_change(local_state.u_prev, local_state.u)
            pdhg_residual =
                _pdhg_relative_residual!(local_state, problem, tau_t, sigma_t, inv_spacing)
            rel_change = max(primal_rel_change, pdhg_residual)
            if rel_change <= T(config.tol)
                converged = true
                iterations = k
                break
            end
        end

        iterations = k
    end

    copyto!(u, local_state.u)
    return SolverStats{T}(iterations, converged, rel_change)
end
