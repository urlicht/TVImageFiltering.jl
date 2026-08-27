using Test
using Random
using TotalVariationImageFiltering
using CUDA

@testset "CUDA Constraint Handling" begin
    Random.seed!(241)
    f_cpu = rand(Float32, 64, 64)
    f_gpu = CUDA.cu(f_cpu)

    pdhg_config = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 1500,
        tau = 0.2f0,
        sigma = 0.2f0,
        theta = 1.0f0,
        tol = 1.0f-5,
        check_every = 10,
    )

    box_constraint = TotalVariationImageFiltering.BoxConstraint(0.05f0, 0.8f0)
    prob_cpu_pdhg_box = TotalVariationImageFiltering.TVProblem(
        f_cpu;
        lambda = 0.12f0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        constraint = box_constraint,
    )
    u_cpu_pdhg_box, stats_cpu_pdhg_box =
        TotalVariationImageFiltering.solve(prob_cpu_pdhg_box, pdhg_config)
    prob_gpu_pdhg_box = TotalVariationImageFiltering.TVProblem(
        f_gpu;
        lambda = 0.12f0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        constraint = box_constraint,
    )
    u_gpu_pdhg_box, stats_gpu_pdhg_box =
        TotalVariationImageFiltering.solve(prob_gpu_pdhg_box, pdhg_config)

    @test stats_cpu_pdhg_box.iterations <= pdhg_config.maxiter
    @test stats_gpu_pdhg_box.iterations <= pdhg_config.maxiter
    @test minimum(Array(u_gpu_pdhg_box)) >= box_constraint.lower - 1.0f-5
    @test maximum(Array(u_gpu_pdhg_box)) <= box_constraint.upper + 1.0f-5
    @test isapprox(Array(u_gpu_pdhg_box), u_cpu_pdhg_box; rtol = 8.0f-4, atol = 8.0f-4)

    prob_gpu_pdhg_box_lambda0 = TotalVariationImageFiltering.TVProblem(
        f_gpu;
        lambda = 0.0f0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        constraint = box_constraint,
    )
    u_gpu_pdhg_box0, stats_gpu_pdhg_box0 = TotalVariationImageFiltering.solve(
        prob_gpu_pdhg_box_lambda0,
        pdhg_config;
        init = CUDA.fill(5.0f0, size(f_gpu)),
    )
    @test stats_gpu_pdhg_box0.iterations == 0
    @test stats_gpu_pdhg_box0.converged
    @test Array(u_gpu_pdhg_box0) ==
          clamp.(f_cpu, box_constraint.lower, box_constraint.upper)

    rof_config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 400,
        tau = 0.0625f0,
        tol = 1.0f-5,
        check_every = 10,
    )
    prob_gpu_rof_constrained = TotalVariationImageFiltering.TVProblem(
        f_gpu;
        lambda = 0.15f0,
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve(
        prob_gpu_rof_constrained,
        rof_config,
    )
end
