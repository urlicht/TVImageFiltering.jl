using Test
using Random
using TotalVariationImageFiltering

struct DummyFidelityForROF <: TotalVariationImageFiltering.AbstractDataFidelity end
struct DummyBoundaryForROF <: TotalVariationImageFiltering.AbstractBoundaryCondition end

function exact_rof_two_pixel(f1::Real, f2::Real, lambda::Real, spacing::Real)
    mean_value = (f1 + f2) / 2
    delta = f2 - f1
    shrunk = sign(delta) * max(abs(delta) - 2 * lambda / spacing, 0)
    return [mean_value - shrunk / 2, mean_value + shrunk / 2]
end

@testset "ROF Internal Helpers" begin
    @test TotalVariationImageFiltering._tau_upper_bound((1.0, 2.0, 3.0), (8, 1, 5)) ==
          1 / 20
    @test TotalVariationImageFiltering._tau_upper_bound((1.0, 2.0), (1, 1)) == Inf

    u_prev = [1.0, 2.0, 3.0]
    u_same = [1.0, 2.0, 3.0]
    u_other = [2.0, 2.0, 2.0]
    @test TotalVariationImageFiltering._relative_change(u_prev, u_same) == 0.0
    @test TotalVariationImageFiltering._relative_change(zeros(3), u_other) > 0.0
end

@testset "ROF Lambda Zero Shortcut" begin
    Random.seed!(31)
    f = randn(6, 5)
    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.0,
        tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
    )
    u = fill(99.0, size(f))
    stats = TotalVariationImageFiltering.solve!(
        u,
        prob,
        TotalVariationImageFiltering.ROFConfig(maxiter = 50, tau = 0.1),
    )

    @test u == f
    @test stats.iterations == 0
    @test stats.converged
    @test stats.rel_change == 0.0
end

@testset "ROF 2-Pixel Analytic Solutions (Both TV Modes)" begin
    config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 20_000,
        tau = 0.0625,
        tol = 1e-10,
        check_every = 20,
    )

    for mode in (
        TotalVariationImageFiltering.AnisotropicTV(),
        TotalVariationImageFiltering.IsotropicTV(),
    )
        for (f1, f2, lambda, spacing) in
            ((-1.3, 2.4, 0.5, 1.0), (0.2, 0.8, 1.0, 2.0), (3.0, -2.0, 0.1, 0.5))
            f = [f1, f2]
            prob = TotalVariationImageFiltering.TVProblem(
                f;
                lambda = lambda,
                spacing = spacing,
                tv_mode = mode,
            )
            u, stats = TotalVariationImageFiltering.solve(prob, config)
            exact = exact_rof_two_pixel(f1, f2, lambda, spacing)

            @test stats.converged
            @test maximum(abs.(u .- exact)) <= 1e-8
        end
    end
end

@testset "ROF Constant Input Is Fixed Point" begin
    f = fill(2.25, 20)
    config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 2000,
        tau = 0.01,
        tol = 1e-10,
        check_every = 10,
    )

    for mode in (
        TotalVariationImageFiltering.AnisotropicTV(),
        TotalVariationImageFiltering.IsotropicTV(),
    )
        prob = TotalVariationImageFiltering.TVProblem(
            f;
            lambda = 1.0,
            spacing = 0.3,
            tv_mode = mode,
        )
        u, stats = TotalVariationImageFiltering.solve(prob, config)
        @test stats.converged
        @test maximum(abs.(u .- f)) <= 1e-10
    end
end

@testset "ROF Tau Bound and Spacing Effects" begin
    Random.seed!(37)
    f = randn(8, 8)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 0.2)
    fine_prob =
        TotalVariationImageFiltering.TVProblem(f; lambda = 0.2, spacing = (0.5, 1.0))

    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 10,
            tau = 0.25,
            tol = 0.0,
            check_every = 1,
        ),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        fine_prob,
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 10,
            tau = 0.1,
            tol = 0.0,
            check_every = 1,
        ),
    )

    u, stats = TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 500,
            tau = 0.2,
            tol = 1e-6,
            check_every = 10,
        ),
    )
    @test size(u) == size(f)
    @test stats.iterations <= 500

    u_fine, _ = TotalVariationImageFiltering.solve(
        TotalVariationImageFiltering.TVProblem(
            [0.0, 2.0];
            lambda = 0.5,
            spacing = 0.5,
            tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
        ),
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 10_000,
            tau = 0.0625,
            tol = 1e-10,
            check_every = 20,
        ),
    )
    u_coarse, _ = TotalVariationImageFiltering.solve(
        TotalVariationImageFiltering.TVProblem(
            [0.0, 2.0];
            lambda = 0.5,
            spacing = 2.0,
            tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
        ),
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 10_000,
            tau = 0.0625,
            tol = 1e-10,
            check_every = 20,
        ),
    )
    @test abs(u_fine[2] - u_fine[1]) < abs(u_coarse[2] - u_coarse[1])
end

@testset "ROF Size-One Dimensions and State Reuse" begin
    f = reshape([3.0], 1, 1)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 0.4, spacing = (0.5, 2.0))
    u, stats = TotalVariationImageFiltering.solve(
        prob,
        TotalVariationImageFiltering.ROFConfig(
            maxiter = 10,
            tau = 100.0,
            tol = 0.0,
            check_every = 1,
        ),
    )
    @test u == f
    @test stats.converged

    Random.seed!(41)
    f2 = randn(16)
    prob2 = TotalVariationImageFiltering.TVProblem(
        f2;
        lambda = 0.2,
        tv_mode = TotalVariationImageFiltering.AnisotropicTV(),
    )
    cfg = TotalVariationImageFiltering.ROFConfig(
        maxiter = 2000,
        tau = 0.0625,
        tol = 1e-8,
        check_every = 20,
    )
    state = TotalVariationImageFiltering.ROFState(f2)

    u1, stats1 = TotalVariationImageFiltering.solve(prob2, cfg; state = state)
    u2, stats2 = TotalVariationImageFiltering.solve(
        prob2,
        cfg;
        init = fill(5.0, size(f2)),
        state = state,
    )
    @test stats1.converged
    @test stats2.converged
    @test maximum(abs.(u1 .- u2)) <= 1e-7
end

@testset "ROF Error Paths" begin
    f = randn(8)
    bad_fidelity_prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        data_fidelity = DummyFidelityForROF(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(bad_fidelity_prob)

    constrained_prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(constrained_prob)

    bad_boundary_prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        boundary = DummyBoundaryForROF(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(bad_boundary_prob)

    mismatch_prob = TotalVariationImageFiltering.TVProblem(randn(6); lambda = 0.2)
    @test_throws ArgumentError TotalVariationImageFiltering.solve!(
        zeros(5),
        mismatch_prob,
        TotalVariationImageFiltering.ROFConfig(),
    )

    bad_state = TotalVariationImageFiltering.ROFState(randn(6))
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        TotalVariationImageFiltering.TVProblem(randn(8); lambda = 0.2),
        TotalVariationImageFiltering.ROFConfig();
        state = bad_state,
    )

    # Bypass keyword constructor checks by calling the positional struct constructor directly.
    negative_lambda_prob = TotalVariationImageFiltering.TVProblem(
        randn(8),
        -0.2,
        (1.0,),
        TotalVariationImageFiltering.L2Fidelity(),
        TotalVariationImageFiltering.IsotropicTV(),
        TotalVariationImageFiltering.Neumann(),
        TotalVariationImageFiltering.NoConstraint(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        negative_lambda_prob,
        TotalVariationImageFiltering.ROFConfig(),
    )
end
