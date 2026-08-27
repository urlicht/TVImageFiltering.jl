using Random

"""
Summary returned by [`select_lambda_discrepancy`](@ref).

Fields:
- `lambda`: selected TV weight.
- `u`: denoised image/volume at `lambda`.
- `solve_stats`: solver statistics for `u`.
- `residual_norm2`: `||u - f||_2^2`.
- `target_norm2`: discrepancy target (`target_scale * length(f) * sigma^2`).
- `relative_mismatch`: relative residual-target mismatch.
- `evaluations`: number of ROF solves used for parameter search.
- `bracket`: last `(lambda_lo, lambda_hi)` search interval.
"""
struct DiscrepancySelection{T<:AbstractFloat,A<:AbstractArray{T}}
    lambda::T
    u::A
    solve_stats::SolverStats{T}
    residual_norm2::T
    target_norm2::T
    relative_mismatch::T
    evaluations::Int
    bracket::Tuple{T,T}
end

"""
Summary returned by [`select_lambda_sure`](@ref).

Fields:
- `lambda`: selected TV weight.
- `u`: denoised image/volume at `lambda`.
- `solve_stats`: solver statistics for `u`.
- `sure`: selected SURE value.
- `residual_norm2`: `||u - f||_2^2`.
- `divergence`: MC estimate of divergence at selected `lambda`.
- `epsilon`: finite-difference perturbation scale used for MC-SURE.
- `evaluations`: number of ROF solves used for parameter search.
- `lambda_grid`: tested lambda grid (sorted ascending).
- `sure_values`: SURE values aligned with `lambda_grid`.
- `residual_values`: residual values aligned with `lambda_grid`.
- `divergence_values`: divergence estimates aligned with `lambda_grid`.
"""
struct SURESelection{T<:AbstractFloat,A<:AbstractArray{T}}
    lambda::T
    u::A
    solve_stats::SolverStats{T}
    sure::T
    residual_norm2::T
    divergence::T
    epsilon::T
    evaluations::Int
    lambda_grid::Vector{T}
    sure_values::Vector{T}
    residual_values::Vector{T}
    divergence_values::Vector{T}
end

function _residual_norm2(u::AbstractArray{T}, f::AbstractArray{T}) where {T<:AbstractFloat}
    return T(sum(abs2, u .- f))
end

function _solve_with_lambda(
    f::AbstractArray{T,N},
    lambda::T,
    config::ROFConfig;
    spacing,
    tv_mode::AbstractTVMode,
    boundary::AbstractBoundaryCondition,
    init::Union{Nothing,AbstractArray},
    state::Union{Nothing,ROFState{T,N}},
) where {T<:AbstractFloat,N}
    problem = TVProblem(
        f;
        lambda = lambda,
        spacing = spacing,
        tv_mode = tv_mode,
        boundary = boundary,
    )
    if state === nothing
        return solve(problem, config; init = init)
    end
    return solve(problem, config; init = init, state = state)
end

function _randn_like!(rng::AbstractRNG, arr::AbstractArray)
    try
        randn!(rng, arr)
    catch err
        err isa MethodError || rethrow()
        randn!(arr)
    end
    return arr
end

"""
Select `lambda` for ROF denoising using Morozov's discrepancy principle.

This function finds `lambda >= 0` such that the ROF residual satisfies

`||u_lambda - f||_2^2 ≈ target_scale * length(f) * sigma^2`,

with `u_lambda` solving

`min_u 0.5 * ||u - f||_2^2 + lambda * TV(u)`.

The search strategy is:
1. bracket the target residual by expanding `lambda_max`,
2. run bisection in `lambda` up to `max_bisect` iterations.

Works for any dimensionality (`N`-D arrays), including 2D images and 3D volumes.

References:
- V. A. Morozov, *Methods for Solving Incorrectly Posed Problems*, 1984.
  DOI: 10.1007/978-1-4612-5280-1
- Y. Wen and R. H. Chan, "Parameter selection for total-variation based image
  restoration using discrepancy principle," *IEEE TIP* 21(4):1770-1781, 2012.
  DOI: 10.1109/TIP.2011.2181401
"""
function select_lambda_discrepancy(
    f::AbstractArray{T,N},
    config::ROFConfig = ROFConfig();
    sigma::Real,
    spacing = nothing,
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    lambda_min::Real = 0.0,
    lambda_max::Real = 1.0,
    target_scale::Real = 1.0,
    rtol::Real = 0.02,
    max_bisect::Int = 30,
    max_expand::Int = 20,
    init::Union{Nothing,AbstractArray} = nothing,
    warm_start::Bool = true,
) where {T<:AbstractFloat,N}
    init === nothing ||
        size(init) == size(f) ||
        throw(ArgumentError("init must match f size"))

    sigma_t = T(sigma)
    sigma_t >= zero(T) || throw(ArgumentError("sigma must be non-negative, got $sigma"))
    isfinite(sigma_t) || throw(ArgumentError("sigma must be finite"))

    lambda_lo = T(lambda_min)
    lambda_hi = T(lambda_max)
    isfinite(lambda_lo) || throw(ArgumentError("lambda_min must be finite"))
    isfinite(lambda_hi) || throw(ArgumentError("lambda_max must be finite"))
    lambda_lo >= zero(T) || throw(ArgumentError("lambda_min must be non-negative"))
    lambda_hi > lambda_lo || throw(ArgumentError("lambda_max must be > lambda_min"))

    target_scale_t = T(target_scale)
    target_scale_t > zero(T) || throw(ArgumentError("target_scale must be positive"))
    isfinite(target_scale_t) || throw(ArgumentError("target_scale must be finite"))

    rtol_t = T(rtol)
    rtol_t >= zero(T) || throw(ArgumentError("rtol must be non-negative"))
    isfinite(rtol_t) || throw(ArgumentError("rtol must be finite"))
    max_bisect > 0 || throw(ArgumentError("max_bisect must be positive"))
    max_expand >= 0 || throw(ArgumentError("max_expand must be non-negative"))

    n = T(length(f))
    target_norm2 = target_scale_t * n * sigma_t^2
    mismatch_scale = max(target_norm2, eps(T))

    state = warm_start ? ROFState(f) : nothing
    init_guess = init
    evaluations = 0

    function evaluate(lambda::T, local_init)
        u, st = _solve_with_lambda(
            f,
            lambda,
            config;
            spacing = spacing,
            tv_mode = tv_mode,
            boundary = boundary,
            init = local_init,
            state = state,
        )
        res = _residual_norm2(u, f)
        return u, st, res
    end

    u_lo, st_lo, res_lo = evaluate(lambda_lo, init_guess)
    evaluations += 1
    init_guess = warm_start ? u_lo : init
    best_lambda = lambda_lo
    best_u = u_lo
    best_stats = st_lo
    best_residual = res_lo
    best_rel = abs(res_lo - target_norm2) / mismatch_scale

    if best_rel <= rtol_t
        return DiscrepancySelection(
            best_lambda,
            best_u,
            best_stats,
            best_residual,
            target_norm2,
            best_rel,
            evaluations,
            (lambda_lo, lambda_lo),
        )
    end

    u_hi, st_hi, res_hi = evaluate(lambda_hi, init_guess)
    evaluations += 1
    init_guess = warm_start ? u_hi : init
    rel_hi = abs(res_hi - target_norm2) / mismatch_scale
    if rel_hi < best_rel
        best_lambda = lambda_hi
        best_u = u_hi
        best_stats = st_hi
        best_residual = res_hi
        best_rel = rel_hi
    end

    expansions = 0
    while (res_hi < target_norm2) && (expansions < max_expand)
        lambda_lo = lambda_hi
        u_lo, st_lo, res_lo = u_hi, st_hi, res_hi

        next_hi = lambda_hi == zero(T) ? one(T) : T(2) * lambda_hi
        (next_hi > lambda_hi && isfinite(next_hi)) || break
        lambda_hi = next_hi

        u_hi, st_hi, res_hi = evaluate(lambda_hi, init_guess)
        evaluations += 1
        init_guess = warm_start ? u_hi : init

        rel_hi = abs(res_hi - target_norm2) / mismatch_scale
        if rel_hi < best_rel
            best_lambda = lambda_hi
            best_u = u_hi
            best_stats = st_hi
            best_residual = res_hi
            best_rel = rel_hi
        end
        expansions += 1
    end

    if res_hi < target_norm2
        return DiscrepancySelection(
            best_lambda,
            best_u,
            best_stats,
            best_residual,
            target_norm2,
            best_rel,
            evaluations,
            (lambda_lo, lambda_hi),
        )
    end

    for _ = 1:max_bisect
        lambda_mid = (lambda_lo + lambda_hi) / T(2)
        if lambda_mid == lambda_lo || lambda_mid == lambda_hi
            break
        end

        u_mid, st_mid, res_mid = evaluate(lambda_mid, init_guess)
        evaluations += 1
        init_guess = warm_start ? u_mid : init

        rel_mid = abs(res_mid - target_norm2) / mismatch_scale
        if rel_mid < best_rel
            best_lambda = lambda_mid
            best_u = u_mid
            best_stats = st_mid
            best_residual = res_mid
            best_rel = rel_mid
        end

        if res_mid < target_norm2
            lambda_lo = lambda_mid
        else
            lambda_hi = lambda_mid
        end

        if best_rel <= rtol_t
            break
        end
    end

    return DiscrepancySelection(
        best_lambda,
        best_u,
        best_stats,
        best_residual,
        target_norm2,
        best_rel,
        evaluations,
        (lambda_lo, lambda_hi),
    )
end

"""
Select `lambda` for ROF denoising by minimizing Monte-Carlo SURE on a grid.

For each `lambda` in `lambda_grid`, this evaluates

`SURE(lambda) = -N*sigma^2 + ||u_lambda - f||_2^2 + 2*sigma^2*div(u_lambda(f))`,

where divergence is approximated with Monte-Carlo finite differences:

`div(u_lambda(f)) ≈ (1/epsilon) * b' * (u_lambda(f + epsilon*b) - u_lambda(f))`,
with `b ~ N(0, I)` and averaged over `mc_samples`.

Works for any dimensionality (`N`-D arrays), including 2D images and 3D volumes.

References:
- C.-A. Deledalle et al., "Stein Unbiased GrAdient estimator of the Risk
  (SUGAR) for multiple parameter selection," 2014.
  HAL: hal-00987295
- S. Ramani, T. Blu, M. Unser, "Monte-Carlo SURE: A black-box optimization of
  regularization parameters for general denoising algorithms," *IEEE TIP*
  17(9):1540-1554, 2008.
  DOI: 10.1109/TIP.2008.2001404
- Y. Lin, B. Wohlberg, H. Guo, "UPRE method for total variation parameter
  selection," *Signal Processing* 90(8):2546-2551, 2010.
  DOI: 10.1016/j.sigpro.2010.02.025
"""
function select_lambda_sure(
    f::AbstractArray{T,N},
    config::ROFConfig = ROFConfig();
    sigma::Real,
    lambda_grid::AbstractVector{<:Real},
    spacing = nothing,
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    epsilon::Union{Nothing,Real} = nothing,
    mc_samples::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
    init::Union{Nothing,AbstractArray} = nothing,
    warm_start::Bool = true,
) where {T<:AbstractFloat,N}
    init === nothing ||
        size(init) == size(f) ||
        throw(ArgumentError("init must match f size"))

    isempty(lambda_grid) && throw(ArgumentError("lambda_grid must be non-empty"))
    lambdas = T[]
    for lambda in lambda_grid
        lambda_t = T(lambda)
        isfinite(lambda_t) || throw(ArgumentError("lambda_grid values must be finite"))
        lambda_t >= zero(T) ||
            throw(ArgumentError("lambda_grid values must be non-negative"))
        push!(lambdas, lambda_t)
    end
    sort!(lambdas)

    sigma_t = T(sigma)
    sigma_t >= zero(T) || throw(ArgumentError("sigma must be non-negative, got $sigma"))
    isfinite(sigma_t) || throw(ArgumentError("sigma must be finite"))
    mc_samples > 0 || throw(ArgumentError("mc_samples must be positive"))

    epsilon_t = if epsilon === nothing
        n_scale = max(T(length(f)^0.3), one(T))
        if sigma_t == zero(T)
            sqrt(eps(T))
        else
            (T(2) * sigma_t) / n_scale
        end
    else
        T(epsilon)
    end
    epsilon_t > zero(T) || throw(ArgumentError("epsilon must be positive"))
    isfinite(epsilon_t) || throw(ArgumentError("epsilon must be finite"))

    n_lambda = length(lambdas)
    sure_values = Vector{T}(undef, n_lambda)
    residual_values = Vector{T}(undef, n_lambda)
    divergence_values = Vector{T}(undef, n_lambda)

    base_state = warm_start ? ROFState(f) : nothing
    pert_state = warm_start ? ROFState(f) : nothing
    direction = similar(f)
    f_perturbed = similar(f)

    base_init = init
    n = T(length(f))
    noise_energy = n * sigma_t^2

    evaluations = 0
    best_idx = 1
    best_sure = T(Inf)
    best_u = similar(f)
    best_stats = SolverStats{T}(0, false, T(Inf))

    for (idx, lambda_t) in pairs(lambdas)
        u, st = _solve_with_lambda(
            f,
            lambda_t,
            config;
            spacing = spacing,
            tv_mode = tv_mode,
            boundary = boundary,
            init = base_init,
            state = base_state,
        )
        evaluations += 1
        base_init = warm_start ? u : init

        residual = _residual_norm2(u, f)
        divergence_acc = zero(T)

        for _ = 1:mc_samples
            _randn_like!(rng, direction)
            @. f_perturbed = f + epsilon_t * direction

            u_pert, _ = _solve_with_lambda(
                f_perturbed,
                lambda_t,
                config;
                spacing = spacing,
                tv_mode = tv_mode,
                boundary = boundary,
                init = (warm_start ? u : init),
                state = pert_state,
            )
            evaluations += 1
            divergence_acc += T(sum(direction .* (u_pert .- u))) / epsilon_t
        end

        divergence = divergence_acc / T(mc_samples)
        sure = -noise_energy + residual + T(2) * sigma_t^2 * divergence

        sure_values[idx] = sure
        residual_values[idx] = residual
        divergence_values[idx] = divergence

        if sure < best_sure
            best_idx = idx
            best_sure = sure
            best_u = u
            best_stats = st
        end
    end

    return SURESelection(
        lambdas[best_idx],
        best_u,
        best_stats,
        best_sure,
        residual_values[best_idx],
        divergence_values[best_idx],
        epsilon_t,
        evaluations,
        lambdas,
        sure_values,
        residual_values,
        divergence_values,
    )
end
