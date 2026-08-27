using Test
using Random
using TotalVariationImageFiltering

const TVIF_GSTV = TotalVariationImageFiltering

function gstv_seminorm(u, group_shape, boundary, spacing)
    gradient = TVIF_GSTV.allocate_dual(u)
    inv_spacing = ntuple(d -> inv(spacing[d]), ndims(u))
    TVIF_GSTV.gradient!(gradient, u, boundary, inv_spacing)
    energy = similar(u)
    sums = similar(u)
    scratch = similar(u)
    total = zero(eltype(u))
    for d = 1:ndims(u)
        @. energy = abs2(gradient[d])
        TVIF_GSTV.group_sum!(sums, scratch, energy, group_shape, boundary)
        total += sum(sqrt, sums)
    end
    return total
end

@testset "GSTV MM proximal optimality" begin
    Random.seed!(601)
    w = 1.0 .+ rand(7, 6)
    mode = TVIF_GSTV.GroupSparseTV((2, 3))
    state = TVIF_GSTV.GSTVState(w, mode)
    z = similar(w)
    alpha = 0.05
    rel, converged = TVIF_GSTV._gstv_group_prox!(
        z,
        w,
        state,
        alpha,
        state.group_shape,
        TVIF_GSTV.Neumann(),
        1000,
        1e-11,
    )
    @test converged
    @test rel <= 1e-11

    @. state.rhs = abs2(z)
    TVIF_GSTV.group_sum!(
        state.cg_r,
        state.cg_p,
        state.rhs,
        state.group_shape,
        TVIF_GSTV.Neumann(),
    )
    @. state.cg_r = inv(sqrt(state.cg_r))
    TVIF_GSTV.group_sum_adjoint!(
        state.rhs,
        state.cg_p,
        state.cg_r,
        state.group_shape,
        TVIF_GSTV.Neumann(),
    )
    optimality = z .- w .+ alpha .* state.rhs .* z
    @test maximum(abs, optimality) <= 1e-8
end

@testset "GSTV group size one matches anisotropic ROF" begin
    Random.seed!(607)
    f = randn(18, 14)
    gstv_cfg =
        TVIF_GSTV.GSTVConfig(maxiter = 2000, tol = 1e-8, cg_tol = 1e-10, check_every = 5)
    rof_cfg =
        TVIF_GSTV.ROFConfig(maxiter = 20_000, tau = 0.04, tol = 1e-11, check_every = 10)

    for boundary in (TVIF_GSTV.Neumann(), TVIF_GSTV.Periodic())
        gstv_problem = TVIF_GSTV.TVProblem(
            f;
            lambda = 0.12,
            tv_mode = TVIF_GSTV.GroupSparseTV(1),
            boundary = boundary,
        )
        rof_problem = TVIF_GSTV.TVProblem(
            f;
            lambda = 0.12,
            tv_mode = TVIF_GSTV.AnisotropicTV(),
            boundary = boundary,
        )
        u_gstv, stats_gstv = TVIF_GSTV.solve(gstv_problem, gstv_cfg)
        u_rof, stats_rof = TVIF_GSTV.solve(rof_problem, rof_cfg)
        @test stats_gstv.converged
        @test stats_rof.converged
        @test isapprox(u_gstv, u_rof; atol = 1e-6, rtol = 0)
    end
end

@testset "GSTV 2D and 3D solve behavior" begin
    Random.seed!(613)
    config = TVIF_GSTV.GSTVConfig(
        maxiter = 500,
        tol = 1e-4,
        mm_maxiter = 100,
        mm_tol = 1e-5,
        cg_maxiter = 100,
        cg_tol = 1e-6,
    )

    cases = (
        (randn(16, 12), (2, 3), TVIF_GSTV.Neumann(), (0.7, 1.2)),
        (randn(8, 7, 6), (3, 2, 1), TVIF_GSTV.Periodic(), (0.8, 1.1, 1.4)),
    )
    for (f, group_shape, boundary, spacing) in cases
        problem = TVIF_GSTV.TVProblem(
            f;
            lambda = 0.08,
            spacing = spacing,
            tv_mode = TVIF_GSTV.GroupSparseTV(group_shape),
            boundary = boundary,
        )
        u, stats = TVIF_GSTV.solve(problem, config)
        initial_objective =
            problem.lambda * gstv_seminorm(f, group_shape, boundary, spacing)
        final_objective =
            sum(abs2, u .- f) / 2 +
            problem.lambda * gstv_seminorm(u, group_shape, boundary, spacing)
        @test stats.converged
        @test all(isfinite, u)
        @test final_objective <= initial_objective + 1e-7
    end

    constant_data = fill(2.5, 8, 7, 6)
    constant_problem = TVIF_GSTV.TVProblem(
        constant_data;
        lambda = 0.2,
        tv_mode = TVIF_GSTV.GroupSparseTV(3),
    )
    constant_result, constant_stats = TVIF_GSTV.solve(constant_problem, config)
    @test constant_stats.converged
    @test isapprox(constant_result, constant_data; atol = 1e-12, rtol = 0)

    zero_problem =
        TVIF_GSTV.TVProblem(randn(8, 7); lambda = 0, tv_mode = TVIF_GSTV.GroupSparseTV(3))
    zero_result, zero_stats = TVIF_GSTV.solve(zero_problem, config; init = zeros(8, 7))
    @test zero_result == zero_problem.f
    @test zero_stats == TVIF_GSTV.SolverStats{Float64}(0, true, 0.0)
end

@testset "GSTV state reuse and batch API" begin
    Random.seed!(617)
    f = randn(12, 10)
    mode = TVIF_GSTV.GroupSparseTV((3, 2))
    problem = TVIF_GSTV.TVProblem(f; lambda = 0.07, tv_mode = mode)
    config = TVIF_GSTV.GSTVConfig(
        maxiter = 1000,
        tol = 1e-5,
        mm_maxiter = 100,
        cg_maxiter = 100,
        cg_tol = 1e-7,
    )
    state = TVIF_GSTV.GSTVState(f, mode)
    u1, stats1 = TVIF_GSTV.solve(problem, config; state = state)
    u2, stats2 = TVIF_GSTV.solve(problem, config; init = fill(4.0, size(f)), state = state)
    @test stats1.converged
    @test stats2.converged
    @test isapprox(u1, u2; atol = 3e-5, rtol = 0)

    f_batch = randn(10, 8, 3)
    batch_state = TVIF_GSTV.GSTVBatchState(f_batch, TVIF_GSTV.GroupSparseTV(3))
    u_batch, summary, per_item = TVIF_GSTV.solve_batch(
        f_batch,
        config;
        lambda = 0.06,
        tv_mode = TVIF_GSTV.GroupSparseTV(3),
        state = batch_state,
        return_per_item_stats = true,
    )
    expected = similar(f_batch)
    expected_stats = Vector{TVIF_GSTV.SolverStats{Float64}}(undef, size(f_batch, 3))
    @views for b = 1:size(f_batch, 3)
        item_problem = TVIF_GSTV.TVProblem(
            selectdim(f_batch, 3, b);
            lambda = 0.06,
            tv_mode = TVIF_GSTV.GroupSparseTV(3),
        )
        item_result, item_stats = TVIF_GSTV.solve(item_problem, config)
        copyto!(selectdim(expected, 3, b), item_result)
        expected_stats[b] = item_stats
    end
    @test isapprox(u_batch, expected; atol = 1e-10, rtol = 0)
    @test per_item == expected_stats
    @test summary.converged == all(item.converged for item in per_item)
    @test summary.iterations == maximum(item.iterations for item in per_item)
end

@testset "GSTV validation and solver routing" begin
    f = randn(8, 7)
    group_problem =
        TVIF_GSTV.TVProblem(f; lambda = 0.1, tv_mode = TVIF_GSTV.GroupSparseTV(3))
    @test_throws ArgumentError TVIF_GSTV.solve(group_problem, TVIF_GSTV.ROFConfig())
    @test_throws ArgumentError TVIF_GSTV.solve(group_problem, TVIF_GSTV.PDHGConfig())
    @test_throws ArgumentError TVIF_GSTV.solve(
        TVIF_GSTV.TVProblem(f; lambda = 0.1),
        TVIF_GSTV.GSTVConfig(),
    )
    @test_throws ArgumentError TVIF_GSTV.solve(
        TVIF_GSTV.TVProblem(
            abs.(f);
            lambda = 0.1,
            tv_mode = TVIF_GSTV.GroupSparseTV(3),
            data_fidelity = TVIF_GSTV.PoissonFidelity(),
        ),
        TVIF_GSTV.GSTVConfig(),
    )
    @test_throws ArgumentError TVIF_GSTV.solve(
        TVIF_GSTV.TVProblem(
            f;
            lambda = 0.1,
            tv_mode = TVIF_GSTV.GroupSparseTV(3),
            constraint = TVIF_GSTV.NonnegativeConstraint(),
        ),
        TVIF_GSTV.GSTVConfig(),
    )
    bad_data = copy(f)
    bad_data[1] = Inf
    @test_throws ArgumentError TVIF_GSTV.solve(
        TVIF_GSTV.TVProblem(bad_data; lambda = 0.1, tv_mode = TVIF_GSTV.GroupSparseTV(3)),
        TVIF_GSTV.GSTVConfig(),
    )
end
