using Test
using Random
using Statistics
using TotalVariationImageFiltering

struct DummyFidelityForPDHG <: TotalVariationImageFiltering.AbstractDataFidelity end

function tv_seminorm(
    u::AbstractArray{T,N},
    spacing::NTuple{N,T},
    tv_mode::TotalVariationImageFiltering.AbstractTVMode,
) where {T<:AbstractFloat,N}
    g = TotalVariationImageFiltering.allocate_dual(u)
    inv_spacing = ntuple(d -> inv(spacing[d]), Val(N))
    TotalVariationImageFiltering.gradient!(
        g,
        u,
        TotalVariationImageFiltering.Neumann(),
        inv_spacing,
    )

    total = zero(T)
    if tv_mode isa TotalVariationImageFiltering.IsotropicTV
        @inbounds for i in eachindex(g[1])
            nrm2 = zero(T)
            for d = 1:N
                nrm2 = muladd(g[d][i], g[d][i], nrm2)
            end
            total += sqrt(nrm2)
        end
    else
        @inbounds for d = 1:N
            total += sum(abs, g[d])
        end
    end

    return total
end

function poisson_data_term(
    u::AbstractArray{T},
    f::AbstractArray{T},
) where {T<:AbstractFloat}
    total = zero(T)
    @inbounds for i in eachindex(u, f)
        ui = u[i]
        fi = f[i]
        if fi == zero(T)
            total += ui
        else
            ui > zero(T) || return T(Inf)
            total += ui - fi * log(ui)
        end
    end
    return total
end

function rand_poisson_knuth(rng::AbstractRNG, λ::T) where {T<:AbstractFloat}
    λ <= zero(T) && return 0
    limit = exp(-λ)
    product = one(T)
    k = 0
    while product > limit
        k += 1
        product *= rand(rng, T)
    end
    return k - 1
end

function sample_poisson_normalized(
    rng::AbstractRNG,
    clean::AbstractArray{T},
    scale::T,
) where {T<:AbstractFloat}
    scale > zero(T) || throw(ArgumentError("scale must be positive"))
    noisy = similar(clean)
    @inbounds for i in eachindex(clean)
        λ = scale * clean[i]
        noisy[i] = T(rand_poisson_knuth(rng, λ)) / scale
    end
    return noisy
end

@testset "PDHG Internal Helpers" begin
    @test TotalVariationImageFiltering._pdhg_operator_norm_sq_upper_bound(
        (1.0, 2.0, 3.0),
        (8, 1, 5),
    ) == 40.0
    @test TotalVariationImageFiltering._pdhg_operator_norm_sq_upper_bound(
        (1.0, 2.0),
        (1, 1),
    ) == 0.0
end

@testset "PDHG Primal Residual Sign Convention" begin
    # This test guards the sign in the primal residual:
    # r_p = (u_prev - u)/tau - K*(p_prev - p), with K* = -div => +div term.
    f = zeros(Float64, 6)
    problem = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (0.5,),
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
    )
    state = TotalVariationImageFiltering.PDHGState(f)

    state.u_prev .= [0.5, -1.0, 1.5, -2.0, 2.5, -3.0]
    state.u .= [0.0, 0.25, -0.5, 0.75, -1.0, 1.25]
    state.p_prev[1] .= [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
    state.p[1] .= [-1.5, 2.5, -3.5, 4.5, -5.5, 6.5]

    tau = 0.3
    sigma = 0.7
    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(1))

    delta_u = state.u_prev .- state.u
    delta_p = (state.p_prev[1] .- state.p[1],)
    div_delta_p = similar(f)
    TotalVariationImageFiltering.divergence!(
        div_delta_p,
        delta_p,
        problem.boundary,
        inv_spacing,
    )
    expected_primal = delta_u ./ tau .+ div_delta_p

    _ = TotalVariationImageFiltering._pdhg_relative_residual!(
        state,
        problem,
        tau,
        sigma,
        inv_spacing,
    )
    @test isapprox(state.primal_tmp, expected_primal; atol = 1e-12, rtol = 0.0)
end

@testset "PDHG L2 Matches ROF" begin
    Random.seed!(71)
    f = randn(32, 24)
    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.18,
        spacing = (1.0, 0.75),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )

    rof_cfg = TotalVariationImageFiltering.ROFConfig(
        maxiter = 5000,
        tau = 0.05,
        tol = 1e-8,
        check_every = 20,
    )
    pdhg_cfg = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 8000,
        tau = 0.12,
        sigma = 0.08,
        theta = 1.0,
        tol = 1e-8,
        check_every = 20,
    )

    u_rof, stats_rof = TotalVariationImageFiltering.solve(prob, rof_cfg)
    u_pdhg, stats_pdhg = TotalVariationImageFiltering.solve(prob, pdhg_cfg)

    @test stats_rof.converged
    @test stats_pdhg.converged
    @test maximum(abs.(u_pdhg .- u_rof)) <= 1e-2
end

@testset "PDHG Periodic L2 Matches ROF" begin
    Random.seed!(73)
    f = randn(20, 16)
    problem = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.12,
        boundary = TotalVariationImageFiltering.Periodic(),
    )
    u_rof, stats_rof = TotalVariationImageFiltering.solve(
        problem,
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 5000,
            tau = 0.05,
            tol = 1e-8,
            check_every = 20,
        ),
    )
    u_pdhg, stats_pdhg = TotalVariationImageFiltering.solve(
        problem,
        TotalVariationImageFiltering.PDHGConfig(
            maxiter = 5000,
            tau = 0.2,
            sigma = 0.2,
            tol = 1e-7,
            check_every = 20,
        ),
    )
    @test stats_rof.converged
    @test stats_pdhg.converged
    @test isapprox(u_pdhg, u_rof; atol = 2e-5, rtol = 0)
end

@testset "PDHG Poisson Fidelity" begin
    Random.seed!(73)
    f = rand(Float64, 28, 20) .* 2 .+ 0.2
    mode = TotalVariationImageFiltering.AnisotropicTV()

    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.12,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        tv_mode = mode,
    )
    cfg = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 4000,
        tau = 0.1,
        sigma = 0.1,
        theta = 1.0,
        tol = 1e-7,
        check_every = 20,
    )

    u, stats = TotalVariationImageFiltering.solve(prob, cfg)
    obj_initial = poisson_data_term(f, f) + prob.lambda * tv_seminorm(f, prob.spacing, mode)
    obj_final = poisson_data_term(u, f) + prob.lambda * tv_seminorm(u, prob.spacing, mode)

    @test stats.converged
    @test minimum(u) >= -1e-10
    @test obj_final <= obj_initial + 1e-4

    prob_lambda0 = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.0,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
    )
    u0, stats0 =
        TotalVariationImageFiltering.solve(prob_lambda0, cfg; init = fill(7.0, size(f)))
    @test u0 == f
    @test stats0.iterations == 0
    @test stats0.converged
    @test stats0.rel_change == 0.0
end

@testset "PDHG Primal Constraints" begin
    cfg = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 5000,
        tau = 0.15,
        sigma = 0.15,
        theta = 1.0,
        tol = 1e-7,
        check_every = 20,
    )

    Random.seed!(75)
    f_l2 = 1.5 .* randn(Float64, 26, 18) .- 0.4

    prob_nonneg = TotalVariationImageFiltering.TVProblem(
        f_l2;
        lambda = 0.08,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
    u_nonneg, st_nonneg = TotalVariationImageFiltering.solve(prob_nonneg, cfg)
    @test st_nonneg.converged
    @test minimum(u_nonneg) >= -1e-10

    box = TotalVariationImageFiltering.BoxConstraint(-0.2, 0.65)
    prob_box = TotalVariationImageFiltering.TVProblem(
        f_l2;
        lambda = 0.08,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        constraint = box,
    )
    u_box, st_box = TotalVariationImageFiltering.solve(prob_box, cfg)
    @test st_box.converged
    @test minimum(u_box) >= box.lower - 1e-10
    @test maximum(u_box) <= box.upper + 1e-10

    prob_box_lambda0 = TotalVariationImageFiltering.TVProblem(
        f_l2;
        lambda = 0.0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        constraint = box,
    )
    u_box0, st_box0 = TotalVariationImageFiltering.solve(
        prob_box_lambda0,
        cfg;
        init = fill(4.0, size(f_l2)),
    )
    @test st_box0.iterations == 0
    @test st_box0.converged
    @test st_box0.rel_change == 0.0
    @test u_box0 == clamp.(f_l2, box.lower, box.upper)

    f_poisson = rand(Float64, 24, 17) .* 1.8 .+ 0.05
    pbox = TotalVariationImageFiltering.BoxConstraint(0.15, 0.95)
    prob_poisson_box = TotalVariationImageFiltering.TVProblem(
        f_poisson;
        lambda = 0.07,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        constraint = pbox,
    )
    u_poisson_box, st_poisson_box =
        TotalVariationImageFiltering.solve(prob_poisson_box, cfg)
    @test st_poisson_box.converged
    @test minimum(u_poisson_box) >= pbox.lower - 1e-10
    @test maximum(u_poisson_box) <= pbox.upper + 1e-10

    prob_poisson_box_lambda0 = TotalVariationImageFiltering.TVProblem(
        f_poisson;
        lambda = 0.0,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        constraint = pbox,
    )
    u_poisson_box0, st_poisson_box0 = TotalVariationImageFiltering.solve(
        prob_poisson_box_lambda0,
        cfg;
        init = fill(2.0, size(f_poisson)),
    )
    @test st_poisson_box0.iterations == 0
    @test st_poisson_box0.converged
    @test st_poisson_box0.rel_change == 0.0
    @test u_poisson_box0 == clamp.(f_poisson, pbox.lower, pbox.upper)

    infeasible_negative = TotalVariationImageFiltering.TVProblem(
        f_poisson;
        lambda = 0.1,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        constraint = TotalVariationImageFiltering.BoxConstraint(-1.0, -0.1),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(infeasible_negative, cfg)

    infeasible_zero = TotalVariationImageFiltering.TVProblem(
        f_poisson;
        lambda = 0.1,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        constraint = TotalVariationImageFiltering.BoxConstraint(-1.0, 0.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(infeasible_zero, cfg)

    zero_data = zeros(Float64, 6, 6)
    feasible_zero = TotalVariationImageFiltering.TVProblem(
        zero_data;
        lambda = 0.0,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        constraint = TotalVariationImageFiltering.BoxConstraint(-1.0, 0.0),
    )
    u_zero, st_zero = TotalVariationImageFiltering.solve(
        feasible_zero,
        cfg;
        init = fill(9.0, size(zero_data)),
    )
    @test st_zero.iterations == 0
    @test st_zero.converged
    @test u_zero == zero_data
end

@testset "PDHG State Reuse and Error Paths" begin
    f = randn(24)
    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.PDHGConfig(
            maxiter = 20,
            tau = 0.6,
            sigma = 0.6,
            tol = 0.0,
            check_every = 1,
        ),
    )

    cfg = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 3000,
        tau = 0.2,
        sigma = 0.2,
        theta = 1.0,
        tol = 1e-8,
        check_every = 20,
    )
    state = TotalVariationImageFiltering.PDHGState(f)
    u1, s1 = TotalVariationImageFiltering.solve(prob, cfg; state = state)
    u2, s2 = TotalVariationImageFiltering.solve(
        prob,
        cfg;
        init = fill(3.0, size(f)),
        state = state,
    )
    @test s1.converged
    @test s2.converged
    @test maximum(abs.(u1 .- u2)) <= 1e-5

    bad_poisson_prob = TotalVariationImageFiltering.TVProblem(
        [-0.1, 0.2];
        lambda = 0.1,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(bad_poisson_prob, cfg)

    bad_fidelity_prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        data_fidelity = DummyFidelityForPDHG(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(bad_fidelity_prob, cfg)

    bad_state = TotalVariationImageFiltering.PDHGState(randn(12))
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        prob,
        cfg;
        state = bad_state,
    )

    # Bypass keyword constructor checks by calling the positional struct constructor directly.
    negative_lambda_prob = TotalVariationImageFiltering.TVProblem(
        randn(24),
        -0.2,
        (1.0,),
        TotalVariationImageFiltering.L2Fidelity(),
        TotalVariationImageFiltering.IsotropicTV(),
        TotalVariationImageFiltering.Neumann(),
        TotalVariationImageFiltering.NoConstraint(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(negative_lambda_prob, cfg)
end

@testset "PDHG Batch API" begin
    Random.seed!(79)
    f_batch = randn(Float64, 12, 10, 4)
    config = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 2500,
        tau = 0.2,
        sigma = 0.2,
        theta = 1.0,
        tol = 1e-7,
        check_every = 10,
    )

    u_batch, stats_batch = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )

    expected = similar(f_batch)
    per_stats = TotalVariationImageFiltering.SolverStats{Float64}[]
    @views for b = 1:size(f_batch, 3)
        fb = selectdim(f_batch, 3, b)
        prob = TotalVariationImageFiltering.TVProblem(
            fb;
            lambda = 0.15,
            tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        )
        ub, st = TotalVariationImageFiltering.solve(prob, config)
        copyto!(selectdim(expected, 3, b), ub)
        push!(per_stats, st)
    end

    @test isapprox(u_batch, expected; rtol = 0.0, atol = 1e-8)
    @test stats_batch.iterations == maximum(st.iterations for st in per_stats)
    @test stats_batch.converged == all(st.converged for st in per_stats)
    @test stats_batch.rel_change == maximum(st.rel_change for st in per_stats)

    u_batch_full, summary_stats, per_item_stats = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        return_per_item_stats = true,
    )
    @test isapprox(u_batch_full, expected; rtol = 0.0, atol = 1e-8)
    @test summary_stats.iterations == stats_batch.iterations
    @test summary_stats.converged == stats_batch.converged
    @test summary_stats.rel_change == stats_batch.rel_change
    @test length(per_item_stats) == length(per_stats)
    @test summary_stats.iterations == maximum(st.iterations for st in per_item_stats)
    @test summary_stats.converged == all(st.converged for st in per_item_stats)
    @test summary_stats.rel_change == maximum(st.rel_change for st in per_item_stats)
    @inbounds for b = 1:length(per_stats)
        @test per_item_stats[b].iterations == per_stats[b].iterations
        @test per_item_stats[b].converged == per_stats[b].converged
        @test per_item_stats[b].rel_change == per_stats[b].rel_change
    end

    constrained_box = TotalVariationImageFiltering.BoxConstraint(-0.1, 0.3)
    u_batch_constrained, stats_batch_constrained = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        constraint = constrained_box,
    )
    @test stats_batch_constrained.iterations <= config.maxiter
    @test minimum(u_batch_constrained) >= constrained_box.lower - 1e-10
    @test maximum(u_batch_constrained) <= constrained_box.upper + 1e-10

    states = [
        TotalVariationImageFiltering.PDHGState(selectdim(f_batch, 3, b)) for
        b = 1:size(f_batch, 3)
    ]
    u1, st1 = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
        state = states,
    )
    u2, st2 = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
        state = states,
        init = fill(3.0, size(f_batch)),
    )
    @test st1.converged
    @test st2.converged
    @test maximum(abs.(u1 .- u2)) <= 1e-5

    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        randn(8),
        config;
        lambda = 0.1,
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        init = zeros(12, 10, 3),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        state = states[1:2],
    )
end

@testset "PDHG Poisson-vs-L2 on Simulated Poisson Noise" begin
    rng = MersenneTwister(97)
    clean = fill(0.15, 40, 40)
    clean[9:32, 9:32] .= 0.8
    clean[17:24, 17:24] .= 1.8

    # Low-photon regime where Poisson modeling should be beneficial.
    scale = 8.0
    noisy = sample_poisson_normalized(rng, clean, scale)

    lambda = 0.12
    mode = TotalVariationImageFiltering.IsotropicTV()
    cfg = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 6000,
        tau = 0.15,
        sigma = 0.15,
        theta = 1.0,
        tol = 1e-7,
        check_every = 20,
    )

    prob_poisson = TotalVariationImageFiltering.TVProblem(
        noisy;
        lambda = lambda,
        data_fidelity = TotalVariationImageFiltering.PoissonFidelity(),
        tv_mode = mode,
    )
    prob_l2 = TotalVariationImageFiltering.TVProblem(
        noisy;
        lambda = lambda,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = mode,
    )

    u_poisson, st_poisson = TotalVariationImageFiltering.solve(prob_poisson, cfg)
    u_l2, st_l2 = TotalVariationImageFiltering.solve(prob_l2, cfg)

    obj_poisson =
        poisson_data_term(u_poisson, noisy) +
        lambda * tv_seminorm(u_poisson, prob_poisson.spacing, mode)
    obj_l2_in_poisson_metric =
        poisson_data_term(u_l2, noisy) + lambda * tv_seminorm(u_l2, prob_l2.spacing, mode)
    mse_poisson = mean(abs2, u_poisson .- clean)
    mse_l2 = mean(abs2, u_l2 .- clean)

    @test st_poisson.converged
    @test st_l2.iterations <= cfg.maxiter
    @test minimum(u_poisson) >= -1e-10
    @test obj_poisson <= obj_l2_in_poisson_metric + 1e-4
    @test isfinite(mse_poisson)
    @test isfinite(mse_l2)
    @test abs(mse_poisson - mse_l2) >= 1e-6
end
