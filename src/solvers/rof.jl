"""
Configuration for the ROF dual-projection solver.

This solver targets the unconstrained Rudin-Osher-Fatemi denoising model with
`L2Fidelity`:

    minimize_u 0.5 * ||u - f||_2^2 + lambda * TV(u)

`tau` is the dual ascent step and must satisfy the stability bound used by this
implementation:

    tau < 1 / (2 * sum_{d: n_d > 1} h_d^(-2))

where `h_d = spacing[d]` and `n_d = size(f, d)`.

Fields:
- `maxiter`: maximum number of iterations.
- `tau`: dual step size.
- `tol`: stopping tolerance on relative primal change.
- `check_every`: evaluate convergence every `check_every` iterations.

References:
- L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise
  removal algorithms," *Physica D* 60(1-4):259-268, 1992.
  DOI: 10.1016/0167-2789(92)90242-F
- A. Chambolle, "An algorithm for total variation minimization and
  applications," *Journal of Mathematical Imaging and Vision* 20:89-97, 2004.
  DOI: 10.1023/B:JMIV.0000011325.36760.1E
"""
struct ROFConfig{T<:AbstractFloat} <: AbstractTVSolver
    maxiter::Int
    tau::T
    tol::T
    check_every::Int
end

function ROFConfig(;
    maxiter::Int = 300,
    tau::Real = 0.0625, # Conservative default; for 2D unit spacing, Chambolle (2004) gives tau < 0.25.
    tol::Real = 1e-4,
    check_every::Int = 10,
)
    T = promote_type(typeof(float(tau)), typeof(float(tol)))
    return ROFConfig{T}(maxiter, T(tau), T(tol), check_every)
end

function _validate(config::ROFConfig)
    _validate_common_config(config.maxiter, config.check_every)
    config.tau > zero(config.tau) || throw(ArgumentError("tau must be positive"))
    config.tol >= zero(config.tol) || throw(ArgumentError("tol must be non-negative"))
    return nothing
end

"""
Reusable workspace for the ROF dual projection solver.

`ROFState` serves two roles:
- scratch buffers (`u`, `u_prev`, `divp`, `g`, `grad_g`) reused every call,
- warm-start storage for the dual variable `p`.

When the same `state` object is passed to repeated `solve!` calls, `p` is not
reset between calls.
"""
struct ROFState{T<:AbstractFloat,N,A<:AbstractArray{T,N},G<:NTuple{N,A}}
    u::A
    u_prev::A
    divp::A
    g::A
    p::G
    grad_g::G
end

function ROFState(reference::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    u = similar(reference)
    u_prev = similar(reference)
    divp = similar(reference)
    g = similar(reference)
    p = allocate_dual(reference)
    grad_g = allocate_dual(reference)
    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(divp, zero(T))
    fill!(g, zero(T))
    for d = 1:N
        fill!(p[d], zero(T))
        fill!(grad_g[d], zero(T))
    end

    return ROFState{T,N,typeof(u),typeof(p)}(u, u_prev, divp, g, p, grad_g)
end

function _relative_change(
    u_prev::AbstractArray{T},
    u::AbstractArray{T},
) where {T<:AbstractFloat}
    return T(sqrt(sum(abs2, u .- u_prev)) / max(sqrt(sum(abs2, u_prev)), eps(T)))
end

function _validate_state_shape(
    state::ROFState{T,N},
    shape::NTuple{N,Int},
) where {T<:AbstractFloat,N}
    size(state.u) == shape ||
        throw(ArgumentError("state.u size must match solve buffer size $shape"))
    size(state.u_prev) == shape ||
        throw(ArgumentError("state.u_prev size must match solve buffer size $shape"))
    size(state.divp) == shape ||
        throw(ArgumentError("state.divp size must match solve buffer size $shape"))
    size(state.g) == shape ||
        throw(ArgumentError("state.g size must match solve buffer size $shape"))

    @inbounds for d = 1:N
        size(state.p[d]) == shape ||
            throw(ArgumentError("state.p[$d] size must match solve buffer size $shape"))
        size(state.grad_g[d]) == shape || throw(
            ArgumentError("state.grad_g[$d] size must match solve buffer size $shape"),
        )
    end
    return nothing
end

function _rof_dual_step!(
    state::ROFState{T,N},
    problem::TVProblem{T,N},
    tau::T,
    inv_spacing::NTuple{N,T},
) where {T<:AbstractFloat,N}
    lambda = problem.lambda
    inv_lambda = inv(lambda)

    divergence!(state.divp, state.p, problem.boundary, inv_spacing)
    @. state.g = state.divp - inv_lambda * problem.f

    gradient!(state.grad_g, state.g, problem.boundary, inv_spacing)
    @inbounds for d = 1:N
        @. state.p[d] = state.p[d] + tau * state.grad_g[d]
    end

    project_dual_ball!(state.p, one(T), problem.tv_mode)

    divergence!(state.divp, state.p, problem.boundary, inv_spacing)
    @. state.u = problem.f - lambda * state.divp

    return nothing
end

function _tau_upper_bound(
    inv_spacing::NTuple{N,T},
    shape::NTuple{N,Int},
) where {N,T<:AbstractFloat}
    lipschitz_term = zero(T)
    @inbounds for d = 1:N
        if shape[d] > 1
            lipschitz_term += abs2(inv_spacing[d])
        end
    end

    lipschitz_term == zero(T) && return T(Inf)
    return inv(T(2) * lipschitz_term)
end

"""
Run ROF denoising in place.

`u` is both the initial guess and output buffer.

Constraint support:
- `ROFConfig` supports only `problem.constraint = NoConstraint()`.
- For constrained problems, use `PDHGConfig`.

State and buffer reuse:
- Reuse `u` and `state` across calls to avoid allocations.
- `u` is copied into solver state as the initial output buffer.
- For ROF iterations, the effective warm start is the retained dual variable `state.p`.
- `state.p` is reused as dual warm start and is not reset by `solve!`.
- For new image data with the same array storage, update in place with
  `copyto!(problem.f, new_f)` and call `solve!` again.
- If image array object, shape/eltype, or problem metadata changes, create a
  new `TVProblem` (and a new `ROFState` only when shape/eltype changes).

The implementation uses a Chambolle-style dual projected-gradient method for the
ROF model.
With dual variable `p` and image `f`, each iteration computes:

    g^k = div(p^k) - f / lambda

    p^(k+1) = Proj_B(p^k + tau * grad(g^k))

    u^(k+1) = f - lambda * div(p^(k+1))

where `Proj_B` is projection onto the TV dual unit ball:
- isotropic TV: `||p[i]||_2 <= 1` per pixel/voxel,
- anisotropic TV: `|p[d][i]| <= 1` per component.

`gradient!` and `divergence!` use forward/backward finite differences with the
selected `Neumann` or `Periodic` boundary handling and account for
`problem.spacing`.

References:
- L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise
  removal algorithms," *Physica D* 60(1-4):259-268, 1992.
  DOI: 10.1016/0167-2789(92)90242-F
- A. Chambolle, "An algorithm for total variation minimization and
  applications," *Journal of Mathematical Imaging and Vision* 20:89-97, 2004.
  DOI: 10.1023/B:JMIV.0000011325.36760.1E
"""
function solve!(
    u::AbstractArray{T,N},
    problem::TVProblem{T,N},
    config::ROFConfig;
    state::Union{Nothing,ROFState{T,N}} = nothing,
) where {T<:AbstractFloat,N}
    _validate(config)
    problem.data_fidelity isa L2Fidelity ||
        throw(ArgumentError("ROF currently supports only L2Fidelity"))
    problem.constraint isa NoConstraint || throw(
        ArgumentError(
            "ROF currently supports only unconstrained problems; set constraint = NoConstraint() or use PDHGConfig",
        ),
    )
    problem.tv_mode isa GroupSparseTV &&
        throw(ArgumentError("ROF does not support GroupSparseTV; use GSTVConfig"))
    (problem.boundary isa Neumann || problem.boundary isa Periodic) ||
        throw(ArgumentError("ROF supports only Neumann and Periodic boundaries"))
    size(u) == size(problem.f) ||
        throw(ArgumentError("You must have the same size as problem.f"))
    problem.lambda >= zero(T) ||
        throw(ArgumentError("lambda must be non-negative, got $(problem.lambda)"))

    local_state = state === nothing ? nothing : state
    local_state === nothing || _validate_state_shape(local_state, size(u))

    if problem.lambda == zero(T)
        copyto!(u, problem.f)
        return SolverStats{T}(0, true, zero(T))
    end

    local_state = local_state === nothing ? ROFState(problem.f) : local_state

    copyto!(local_state.u, u)
    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter
    tau_t = T(config.tau)
    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(N))
    tau_upper = _tau_upper_bound(inv_spacing, size(problem.f))
    tau_t < tau_upper || throw(
        ArgumentError(
            "tau must be < $(tau_upper) for this grid spacing and shape; got $(tau_t)",
        ),
    )

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)

        _rof_dual_step!(local_state, problem, tau_t, inv_spacing)

        if (k % config.check_every == 0) || (k == config.maxiter)
            rel_change = _relative_change(local_state.u_prev, local_state.u)
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
