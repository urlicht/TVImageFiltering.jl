using Test
using Random
using TotalVariationImageFiltering
using CUDA

@testset "CUDA PDHG Batch API" begin
    Random.seed!(251)
    f_batch_pdhg_cpu = rand(Float32, 32, 32, 3)
    f_batch_pdhg_gpu = CUDA.cu(f_batch_pdhg_cpu)

    pdhg_config = TotalVariationImageFiltering.PDHGConfig(
        maxiter = 1500,
        tau = 0.2f0,
        sigma = 0.2f0,
        theta = 1.0f0,
        tol = 1.0f-5,
        check_every = 10,
    )

    u_batch_pdhg_cpu, stats_batch_pdhg_cpu = TotalVariationImageFiltering.solve_batch(
        f_batch_pdhg_cpu,
        pdhg_config;
        lambda = 0.1f0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )
    u_batch_pdhg_gpu, stats_batch_pdhg_gpu = TotalVariationImageFiltering.solve_batch(
        f_batch_pdhg_gpu,
        pdhg_config;
        lambda = 0.1f0,
        data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
        tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    )

    @test stats_batch_pdhg_cpu.iterations <= pdhg_config.maxiter
    @test stats_batch_pdhg_gpu.iterations <= pdhg_config.maxiter
    @test isapprox(Array(u_batch_pdhg_gpu), u_batch_pdhg_cpu; rtol = 1.0f-3, atol = 1.0f-3)

    u_batch_pdhg_gpu_box, stats_batch_pdhg_gpu_box =
        TotalVariationImageFiltering.solve_batch(
            f_batch_pdhg_gpu,
            pdhg_config;
            lambda = 0.1f0,
            data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
            tv_mode = TotalVariationImageFiltering.IsotropicTV(),
            constraint = TotalVariationImageFiltering.BoxConstraint(-0.1f0, 0.2f0),
        )
    @test stats_batch_pdhg_gpu_box.iterations <= pdhg_config.maxiter
    @test minimum(Array(u_batch_pdhg_gpu_box)) >= -0.10001f0
    @test maximum(Array(u_batch_pdhg_gpu_box)) <= 0.20001f0

    rof_config = TotalVariationImageFiltering.ROFConfig(
        maxiter = 400,
        tau = 0.0625f0,
        tol = 1.0f-5,
        check_every = 10,
    )
    @test_throws ArgumentError TotalVariationImageFiltering.solve_batch(
        f_batch_pdhg_gpu,
        rof_config;
        lambda = 0.15f0,
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
end
