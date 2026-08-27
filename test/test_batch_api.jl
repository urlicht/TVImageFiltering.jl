using Test
using Random
using TotalVariationImageFiltering

@testset "solve_batch CPU API" begin
    Random.seed!(503)
    f_batch = randn(Float64, 16, 12, 4)
    config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 1500,
        tau = 0.0625,
        tol = 1e-8,
        check_every = 10,
    )

    u_batch, stats_batch = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.2,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )

    expected = similar(f_batch)
    per_stats = TotalVariationImageFiltering.SolverStats{Float64}[]
    @views for b = 1:size(f_batch, 3)
        fb = selectdim(f_batch, 3, b)
        prob = TotalVariationImageFiltering.TVProblem(
            fb;
            lambda = 0.2,
            tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        )
        ub, st = TotalVariationImageFiltering.solve(prob, config)
        copyto!(selectdim(expected, 3, b), ub)
        push!(per_stats, st)
    end

    @test isapprox(u_batch, expected; rtol = 0.0, atol = 1e-10)
    @test stats_batch.iterations == maximum(st.iterations for st in per_stats)
    @test stats_batch.converged == all(st.converged for st in per_stats)
    @test stats_batch.rel_change == maximum(st.rel_change for st in per_stats)

    u_batch_full, summary_stats, per_item_stats = TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.2,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        return_per_item_stats = true,
    )
    @test isapprox(u_batch_full, expected; rtol = 0.0, atol = 1e-10)
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
end

@testset "solve_batch state reuse and error paths" begin
    Random.seed!(509)
    f_batch = randn(Float64, 8, 8, 3)
    config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 3000,
        tau = 0.0625,
        tol = 1e-9,
        check_every = 10,
    )

    states = [
        TotalVariationImageFiltering.ROFState(selectdim(f_batch, 3, b)) for
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
        init = zeros(8, 8, 2),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        state = states[1:2],
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
end
