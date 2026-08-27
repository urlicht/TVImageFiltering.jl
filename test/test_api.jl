using Test
using Random
using TotalVariationImageFiltering

struct DummySolverForAPI <: TotalVariationImageFiltering.AbstractTVSolver end

@testset "solve API Default Path and Init Semantics" begin
    Random.seed!(101)
    f = randn(10)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 0.0)
    init = fill(9.0, size(f))
    init_before = copy(init)

    u, stats = TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.ROFConfig();
        init = init,
    )
    @test u == f
    @test stats.converged
    @test u !== init
    @test u !== f
    @test init == init_before

    u_default, stats_default = TotalVariationImageFiltering.solve(prob)
    @test u_default == f
    @test stats_default isa TotalVariationImageFiltering.SolverStats{Float64}
end

@testset "solve API Init Validation" begin
    f = randn(5, 4)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 0.1)
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.ROFConfig();
        init = zeros(5, 5),
    )
end

@testset "solve API Keyword Forwarding to solve!" begin
    Random.seed!(103)
    f = randn(20)
    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
    )
    cfg = TotalVariationImageFiltering.ROFConfig(
        maxiter = 2000,
        tau = 0.0625,
        tol = 1e-8,
        check_every = 20,
    )
    state = TotalVariationImageFiltering.ROFState(f)

    u1, s1 = TotalVariationImageFiltering.solve(prob, cfg; state = state)
    u2, s2 = TotalVariationImageFiltering.solve(
        prob,
        cfg;
        init = fill(3.0, size(f)),
        state = state,
    )
    @test s1.converged
    @test s2.converged
    @test isapprox(u1, u2; atol = 1e-6, rtol = 0.0)
end

@testset "solve!/solve Fallback for Unknown Solver" begin
    f = randn(6)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 0.1)
    u = zeros(6)

    @test_throws MethodError TotalVariationImageFiltering.solve!(
        u,
        prob,
        DummySolverForAPI(),
    )
    @test_throws MethodError TotalVariationImageFiltering.solve(prob, DummySolverForAPI())
end
