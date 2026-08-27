"""
Configuration for low-memory overlapping group-sparse TV denoising.

The outer problem is solved with ADMM. Directional overlapping-group proximal
maps use majorization-minimization (MM), and the quadratic primal update uses a
matrix-free conjugate-gradient (CG) solve.
"""
struct GSTVConfig{T<:AbstractFloat} <: AbstractTVSolver
    maxiter::Int
    rho::T
    tol::T
    check_every::Int
    mm_maxiter::Int
    mm_tol::T
    cg_maxiter::Int
    cg_tol::T
end

function GSTVConfig(;
    maxiter::Int = 300,
    rho::Real = 1.0,
    tol::Real = 1e-4,
    check_every::Int = 5,
    mm_maxiter::Int = 20,
    mm_tol::Real = 1e-5,
    cg_maxiter::Int = 50,
    cg_tol::Real = 1e-5,
)
    T = promote_type(
        typeof(float(rho)),
        typeof(float(tol)),
        typeof(float(mm_tol)),
        typeof(float(cg_tol)),
    )
    return GSTVConfig{T}(
        maxiter,
        T(rho),
        T(tol),
        check_every,
        mm_maxiter,
        T(mm_tol),
        cg_maxiter,
        T(cg_tol),
    )
end

function _validate(config::GSTVConfig)
    _validate_common_config(config.maxiter, config.check_every)
    config.rho > zero(config.rho) || throw(ArgumentError("rho must be positive"))
    config.tol >= zero(config.tol) || throw(ArgumentError("tol must be non-negative"))
    config.mm_maxiter > 0 || throw(ArgumentError("mm_maxiter must be positive"))
    config.mm_tol >= zero(config.mm_tol) ||
        throw(ArgumentError("mm_tol must be non-negative"))
    config.cg_maxiter > 0 || throw(ArgumentError("cg_maxiter must be positive"))
    config.cg_tol >= zero(config.cg_tol) ||
        throw(ArgumentError("cg_tol must be non-negative"))
    return nothing
end

function _normalized_group_shape(
    reference::AbstractArray{T,N},
    mode::GroupSparseTV,
) where {T,N}
    return _normalize_tv_mode(Val(N), size(reference), mode).group_shape
end

"""
Reusable workspace and warm-start state for [`GSTVConfig`](@ref).

`v` and `b` retain the ADMM split and scaled-dual variables between calls.
Construct the state with the same spatial shape, eltype, and group mode used by
the corresponding `TVProblem`.
"""
struct GSTVState{T<:AbstractFloat,N,A<:AbstractArray{T,N},G<:NTuple{N,A}}
    group_shape::NTuple{N,Int}
    u::A
    u_prev::A
    rhs::A
    div_tmp::A
    cg_r::A
    cg_p::A
    cg_Ap::A
    grad_u::G
    v::G
    v_prev::G
    b::G
end

function GSTVState(
    reference::AbstractArray{T,N},
    mode::GroupSparseTV,
) where {T<:AbstractFloat,N}
    group_shape = _normalized_group_shape(reference, mode)
    u = similar(reference)
    u_prev = similar(reference)
    rhs = similar(reference)
    div_tmp = similar(reference)
    cg_r = similar(reference)
    cg_p = similar(reference)
    cg_Ap = similar(reference)
    grad_u = allocate_dual(reference)
    v = allocate_dual(reference)
    v_prev = allocate_dual(reference)
    b = allocate_dual(reference)

    for buffer in (u, u_prev, rhs, div_tmp, cg_r, cg_p, cg_Ap)
        fill!(buffer, zero(T))
    end
    @inbounds for d = 1:N
        fill!(grad_u[d], zero(T))
        fill!(v[d], zero(T))
        fill!(v_prev[d], zero(T))
        fill!(b[d], zero(T))
    end

    return GSTVState{T,N,typeof(u),typeof(grad_u)}(
        group_shape,
        u,
        u_prev,
        rhs,
        div_tmp,
        cg_r,
        cg_p,
        cg_Ap,
        grad_u,
        v,
        v_prev,
        b,
    )
end

"""
One-slice reusable workspace for low-memory sequential GSTV batch solves.

The final axis of `reference_batch` is the batch axis. The state contains one
contiguous spatial data buffer and one [`GSTVState`](@ref), so peak workspace
memory is independent of batch size.
"""
struct GSTVBatchState{T<:AbstractFloat,A<:AbstractArray{T},S<:GSTVState}
    f_work::A
    u_work::A
    solver_state::S
end

function GSTVBatchState(
    reference_batch::AbstractArray{T,N},
    mode::GroupSparseTV,
) where {T<:AbstractFloat,N}
    N >= 2 || throw(ArgumentError("reference_batch must include a batch axis"))
    spatial_shape = ntuple(d -> size(reference_batch, d), Val(N - 1))
    f_work = similar(reference_batch, T, spatial_shape)
    u_work = similar(reference_batch, T, spatial_shape)
    fill!(f_work, zero(T))
    fill!(u_work, zero(T))
    solver_state = GSTVState(f_work, mode)
    return GSTVBatchState{T,typeof(f_work),typeof(solver_state)}(
        f_work,
        u_work,
        solver_state,
    )
end

function _reset_gstv_duals!(state::GSTVState{T,N}) where {T,N}
    @inbounds for d = 1:N
        fill!(state.v[d], zero(T))
        fill!(state.v_prev[d], zero(T))
        fill!(state.b[d], zero(T))
    end
    return nothing
end

function _validate_gstv_state(
    state::GSTVState{T,N},
    shape::NTuple{N,Int},
    group_shape::NTuple{N,Int},
) where {T,N}
    state.group_shape == group_shape || throw(
        ArgumentError(
            "state group shape $(state.group_shape) must match problem group shape $group_shape",
        ),
    )
    for (name, buffer) in (
        (:u, state.u),
        (:u_prev, state.u_prev),
        (:rhs, state.rhs),
        (:div_tmp, state.div_tmp),
        (:cg_r, state.cg_r),
        (:cg_p, state.cg_p),
        (:cg_Ap, state.cg_Ap),
    )
        size(buffer) == shape ||
            throw(ArgumentError("state.$name size must match solve buffer size $shape"))
    end
    @inbounds for d = 1:N
        for (name, buffers) in
            ((:grad_u, state.grad_u), (:v, state.v), (:v_prev, state.v_prev), (:b, state.b))
            size(buffers[d]) == shape || throw(
                ArgumentError("state.$name[$d] size must match solve buffer size $shape"),
            )
        end
    end
    return nothing
end

function _gstv_apply_primal_operator!(
    out::AbstractArray{T,N},
    input::AbstractArray{T,N},
    state::GSTVState{T,N},
    rho::T,
    boundary::AbstractBoundaryCondition,
    inv_spacing::NTuple{N,T},
) where {T<:AbstractFloat,N}
    gradient!(state.grad_u, input, boundary, inv_spacing)
    divergence!(state.div_tmp, state.grad_u, boundary, inv_spacing)
    @. out = input - rho * state.div_tmp
    return out
end

function _gstv_cg!(
    state::GSTVState{T,N},
    rho::T,
    boundary::AbstractBoundaryCondition,
    inv_spacing::NTuple{N,T},
    maxiter::Int,
    tol::T,
) where {T<:AbstractFloat,N}
    _gstv_apply_primal_operator!(state.cg_Ap, state.u, state, rho, boundary, inv_spacing)
    @. state.cg_r = state.rhs - state.cg_Ap
    copyto!(state.cg_p, state.cg_r)

    residual2 = T(sum(abs2, state.cg_r))
    rhs_norm = T(sqrt(sum(abs2, state.rhs)))
    scale = max(rhs_norm, one(T))
    rel_residual = T(sqrt(residual2)) / scale
    isfinite(rel_residual) || throw(ErrorException("CG produced a nonfinite residual"))
    rel_residual <= tol && return rel_residual, true

    for _ = 1:maxiter
        _gstv_apply_primal_operator!(
            state.cg_Ap,
            state.cg_p,
            state,
            rho,
            boundary,
            inv_spacing,
        )
        denominator = T(sum(state.cg_p .* state.cg_Ap))
        (isfinite(denominator) && denominator > zero(T)) ||
            throw(ErrorException("CG encountered a non-positive or nonfinite curvature"))

        alpha = residual2 / denominator
        @. state.u = state.u + alpha * state.cg_p
        @. state.cg_r = state.cg_r - alpha * state.cg_Ap

        next_residual2 = T(sum(abs2, state.cg_r))
        rel_residual = T(sqrt(next_residual2)) / scale
        isfinite(rel_residual) || throw(ErrorException("CG produced a nonfinite residual"))
        rel_residual <= tol && return rel_residual, true

        beta = next_residual2 / residual2
        @. state.cg_p = state.cg_r + beta * state.cg_p
        residual2 = next_residual2
    end

    return rel_residual, false
end

function _gstv_group_prox!(
    z::AbstractArray{T,N},
    w::AbstractArray{T,N},
    state::GSTVState{T,N},
    alpha::T,
    group_shape::NTuple{N,Int},
    boundary::AbstractBoundaryCondition,
    maxiter::Int,
    tol::T,
) where {T<:AbstractFloat,N}
    copyto!(z, w)
    alpha == zero(T) && return zero(T), true

    if all(==(1), group_shape)
        @. z = sign(w) * max(abs(w) - alpha, zero(T))
        return zero(T), true
    end

    rel_change = T(Inf)
    for _ = 1:maxiter
        @. state.rhs = abs2(z)
        group_sum!(state.cg_r, state.cg_p, state.rhs, group_shape, boundary)
        @. state.cg_r = ifelse(state.cg_r == zero(T), T(Inf), inv(sqrt(state.cg_r)))
        group_sum_adjoint!(state.rhs, state.cg_p, state.cg_r, group_shape, boundary)
        @. state.cg_r = w / (one(T) + alpha * state.rhs)
        @. state.cg_p = state.cg_r - z

        delta_norm = T(sqrt(sum(abs2, state.cg_p)))
        z_norm = T(sqrt(sum(abs2, z)))
        rel_change = delta_norm / max(z_norm, one(T))
        isfinite(rel_change) ||
            throw(ErrorException("MM group proximal map produced a nonfinite residual"))
        copyto!(z, state.cg_r)
        rel_change <= tol && return rel_change, true
    end

    return rel_change, false
end

function _validate_gstv_problem(problem::TVProblem)
    problem.tv_mode isa GroupSparseTV ||
        throw(ArgumentError("GSTVConfig requires problem.tv_mode = GroupSparseTV(...)"))
    problem.data_fidelity isa L2Fidelity ||
        throw(ArgumentError("GSTVConfig currently supports only L2Fidelity"))
    problem.constraint isa NoConstraint ||
        throw(ArgumentError("GSTVConfig currently supports only NoConstraint"))
    (problem.boundary isa Neumann || problem.boundary isa Periodic) ||
        throw(ArgumentError("GSTVConfig supports only Neumann and Periodic boundaries"))
    any(.!isfinite.(problem.f)) && throw(ArgumentError("GSTVConfig requires finite observation data f"))
    return nothing
end

"""
Run overlapping group-sparse anisotropic TV denoising in place.

This solver supports finite floating-point observations, `L2Fidelity`,
`NoConstraint`, and either `Neumann` or `Periodic` boundaries.
"""
function solve!(
    u::AbstractArray{T,N},
    problem::TVProblem{T,N},
    config::GSTVConfig;
    state::Union{Nothing,GSTVState{T,N}} = nothing,
) where {T<:AbstractFloat,N}
    _validate(config)
    _validate_gstv_problem(problem)
    size(u) == size(problem.f) ||
        throw(ArgumentError("u must have the same size as problem.f"))
    any(.!isfinite.(u)) && throw(ArgumentError("GSTVConfig requires a finite initial u"))

    if problem.lambda == zero(T)
        copyto!(u, problem.f)
        return SolverStats{T}(0, true, zero(T))
    end

    group_shape = problem.tv_mode.group_shape
    local_state = state === nothing ? GSTVState(problem.f, problem.tv_mode) : state
    _validate_gstv_state(local_state, size(u), group_shape)

    rho = T(config.rho)
    alpha = problem.lambda / rho
    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(N))
    copyto!(local_state.u, u)

    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        @inbounds for d = 1:N
            copyto!(local_state.v_prev[d], local_state.v[d])
            @. local_state.grad_u[d] = local_state.v[d] - local_state.b[d]
        end

        divergence!(local_state.div_tmp, local_state.grad_u, problem.boundary, inv_spacing)
        @. local_state.rhs = problem.f - rho * local_state.div_tmp
        cg_residual, cg_converged = _gstv_cg!(
            local_state,
            rho,
            problem.boundary,
            inv_spacing,
            config.cg_maxiter,
            T(config.cg_tol),
        )

        gradient!(local_state.grad_u, local_state.u, problem.boundary, inv_spacing)
        grad_norm2 = zero(T)
        v_norm2 = zero(T)
        primal_norm2 = zero(T)
        mm_converged = true
        @inbounds for d = 1:N
            grad_norm2 += sum(abs2, local_state.grad_u[d])
            @. local_state.cg_Ap = local_state.grad_u[d] + local_state.b[d]
            _, direction_converged = _gstv_group_prox!(
                local_state.v[d],
                local_state.cg_Ap,
                local_state,
                alpha,
                group_shape,
                problem.boundary,
                config.mm_maxiter,
                T(config.mm_tol),
            )
            mm_converged &= direction_converged
            v_norm2 += sum(abs2, local_state.v[d])
            @. local_state.grad_u[d] = local_state.grad_u[d] - local_state.v[d]
            @. local_state.b[d] = local_state.b[d] + local_state.grad_u[d]
            primal_norm2 += sum(abs2, local_state.grad_u[d])
            @. local_state.v_prev[d] = local_state.v[d] - local_state.v_prev[d]
        end

        divergence!(local_state.div_tmp, local_state.v_prev, problem.boundary, inv_spacing)
        dual_norm = rho * T(sqrt(sum(abs2, local_state.div_tmp)))
        primal_norm = T(sqrt(primal_norm2))
        primal_scale = max(
            T(sqrt(grad_norm2)),
            T(sqrt(v_norm2)),
            sqrt(T(N * length(local_state.u))),
            one(T),
        )
        dual_scale = max(sqrt(T(length(local_state.u))), one(T))
        primal_residual = primal_norm / primal_scale
        dual_residual = dual_norm / dual_scale
        primal_change = _relative_change(local_state.u_prev, local_state.u)
        rel_change = max(primal_change, primal_residual, dual_residual)
        all(isfinite, (rel_change, cg_residual)) ||
            throw(ErrorException("GSTV iteration produced a nonfinite residual"))

        if ((k % config.check_every == 0) || (k == config.maxiter)) &&
           rel_change <= T(config.tol) &&
           cg_converged &&
           mm_converged
            converged = true
            iterations = k
            break
        end
        iterations = k
    end

    copyto!(u, local_state.u)
    return SolverStats{T}(iterations, converged, rel_change)
end
