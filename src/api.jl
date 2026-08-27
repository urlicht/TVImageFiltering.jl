"""
Solve a TV problem and return `(u, stats)`.

`config` selects the solver backend. Implemented: `ROFConfig`, `PDHGConfig`,
and `GSTVConfig`.
"""
function solve(
    problem::TVProblem,
    config::AbstractTVSolver = ROFConfig();
    init::Union{Nothing,AbstractArray} = nothing,
    kwargs...,
)
    u = init === nothing ? copy(problem.f) : copy(init)
    size(u) == size(problem.f) || throw(ArgumentError("init must match problem.f size"))
    stats = solve!(u, problem, config; kwargs...)
    return u, stats
end

function solve!(u::AbstractArray, problem::TVProblem, config::AbstractTVSolver; kwargs...)
    throw(MethodError(solve!, (u, problem, config)))
end

"""
Solve a batch of same-size images stored in one array.

The last axis is interpreted as batch index, and TV operators act only on the
leading spatial axes.

Return values:
- Default (`return_per_item_stats = false`): `(u_batch, summary_stats)`.
- With `return_per_item_stats = true`: `(u_batch, summary_stats, per_item_stats)`,
  where `per_item_stats` is a `Vector{SolverStats}` for each batch item.
"""
function solve_batch(
    f_batch::AbstractArray{T,N},
    config::AbstractTVSolver = ROFConfig();
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    init::Union{Nothing,AbstractArray} = nothing,
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))

    u_batch = init === nothing ? copy(f_batch) : copy(init)
    size(u_batch) == size(f_batch) || throw(ArgumentError("init must match f_batch size"))

    stats = solve_batch!(
        u_batch,
        f_batch,
        config;
        lambda = lambda,
        spacing = spacing,
        data_fidelity = data_fidelity,
        tv_mode = tv_mode,
        boundary = boundary,
        constraint = constraint,
        state = state,
        return_per_item_stats = return_per_item_stats,
    )
    if return_per_item_stats
        summary_stats, per_item_stats = stats
        return u_batch, summary_stats, per_item_stats
    end
    return u_batch, stats
end

function solve_batch!(
    u_batch::AbstractArray,
    f_batch::AbstractArray,
    config::AbstractTVSolver;
    kwargs...,
)
    throw(MethodError(solve_batch!, (u_batch, f_batch, config)))
end

function _zero_batch_result(
    ::Type{T},
    batch_count::Int,
    return_per_item_stats::Bool,
) where {T<:AbstractFloat}
    summary = SolverStats{T}(0, true, zero(T))
    if return_per_item_stats
        return summary, fill(summary, batch_count)
    end
    return summary
end

function solve_batch!(
    u_batch::AbstractArray{T,N},
    f_batch::AbstractArray{T,N},
    config::GSTVConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))
    tv_mode isa GroupSparseTV ||
        throw(ArgumentError("GSTVConfig requires tv_mode = GroupSparseTV(...)"))

    batch_count = size(f_batch, N)
    batch_count == 0 && return _zero_batch_result(T, batch_count, return_per_item_stats)

    local_state = if state === nothing
        GSTVBatchState(f_batch, tv_mode)
    elseif state isa GSTVBatchState
        state
    else
        throw(ArgumentError("state must be `nothing` or a compatible GSTVBatchState"))
    end

    spatial_shape = ntuple(d -> size(f_batch, d), Val(N - 1))
    size(local_state.f_work) == spatial_shape || throw(
        ArgumentError(
            "state spatial shape must match f_batch spatial shape $spatial_shape",
        ),
    )
    normalized_mode = _normalize_tv_mode(Val(N - 1), spatial_shape, tv_mode)
    local_state.solver_state.group_shape == normalized_mode.group_shape ||
        throw(ArgumentError("state group shape must match tv_mode group shape"))

    max_iterations = 0
    all_converged = true
    max_rel_change = zero(T)
    per_item_stats =
        return_per_item_stats ? Vector{SolverStats{T}}(undef, batch_count) : nothing

    @views for b = 1:batch_count
        copyto!(local_state.f_work, selectdim(f_batch, N, b))
        copyto!(local_state.u_work, selectdim(u_batch, N, b))
        _reset_gstv_duals!(local_state.solver_state)
        problem = TVProblem(
            local_state.f_work;
            lambda = lambda,
            spacing = spacing,
            data_fidelity = data_fidelity,
            tv_mode = normalized_mode,
            boundary = boundary,
            constraint = constraint,
        )
        item_stats =
            solve!(local_state.u_work, problem, config; state = local_state.solver_state)
        copyto!(selectdim(u_batch, N, b), local_state.u_work)

        max_iterations = max(max_iterations, item_stats.iterations)
        all_converged &= item_stats.converged
        max_rel_change = max(max_rel_change, item_stats.rel_change)
        return_per_item_stats && (per_item_stats[b] = item_stats)
    end

    summary = SolverStats{T}(max_iterations, all_converged, max_rel_change)
    return return_per_item_stats ? (summary, per_item_stats) : summary
end

function solve_batch!(
    u_batch::AbstractArray{T,N},
    f_batch::AbstractArray{T,N},
    config::ROFConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))
    constraint isa NoConstraint || throw(
        ArgumentError(
            "ROF currently supports only unconstrained problems; set constraint = NoConstraint() or use PDHGConfig",
        ),
    )

    batch_count = size(f_batch, N)
    local_states = if state === nothing
        nothing
    elseif state isa AbstractVector
        length(state) == batch_count || throw(
            ArgumentError("state vector length must equal batch size $batch_count"),
        )
        state
    else
        throw(ArgumentError("state must be `nothing` or a vector of per-image state objects"))
    end

    max_iterations = 0
    converged = true
    max_rel_change = zero(T)
    per_item_stats =
        return_per_item_stats ? Vector{SolverStats{T}}(undef, batch_count) : nothing

    @views for b = 1:batch_count
        f_view = selectdim(f_batch, N, b)
        u_view = selectdim(u_batch, N, b)
        problem = TVProblem(
            f_view;
            lambda = lambda,
            spacing = spacing,
            data_fidelity = data_fidelity,
            tv_mode = tv_mode,
            boundary = boundary,
            constraint = constraint,
        )

        stats =
            local_states === nothing ? solve!(u_view, problem, config) :
            solve!(u_view, problem, config; state = local_states[b])

        max_iterations = max(max_iterations, stats.iterations)
        converged &= stats.converged
        max_rel_change = max(max_rel_change, stats.rel_change)
        if return_per_item_stats
            per_item_stats[b] = stats
        end
    end

    summary_stats = SolverStats{T}(max_iterations, converged, max_rel_change)
    if return_per_item_stats
        return summary_stats, per_item_stats
    end
    return summary_stats
end

function solve_batch!(
    u_batch::AbstractArray{T,N},
    f_batch::AbstractArray{T,N},
    config::PDHGConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))

    batch_count = size(f_batch, N)
    local_states = if state === nothing
        nothing
    elseif state isa AbstractVector
        length(state) == batch_count || throw(
            ArgumentError("state vector length must equal batch size $batch_count"),
        )
        state
    else
        throw(ArgumentError("state must be `nothing` or a vector of per-image state objects"))
    end

    max_iterations = 0
    converged = true
    max_rel_change = zero(T)
    per_item_stats =
        return_per_item_stats ? Vector{SolverStats{T}}(undef, batch_count) : nothing

    @views for b = 1:batch_count
        f_view = selectdim(f_batch, N, b)
        u_view = selectdim(u_batch, N, b)
        problem = TVProblem(
            f_view;
            lambda = lambda,
            spacing = spacing,
            data_fidelity = data_fidelity,
            tv_mode = tv_mode,
            boundary = boundary,
            constraint = constraint,
        )

        stats =
            local_states === nothing ? solve!(u_view, problem, config) :
            solve!(u_view, problem, config; state = local_states[b])

        max_iterations = max(max_iterations, stats.iterations)
        converged &= stats.converged
        max_rel_change = max(max_rel_change, stats.rel_change)
        if return_per_item_stats
            per_item_stats[b] = stats
        end
    end

    summary_stats = SolverStats{T}(max_iterations, converged, max_rel_change)
    if return_per_item_stats
        return summary_stats, per_item_stats
    end
    return summary_stats
end
