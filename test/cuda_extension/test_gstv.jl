using Test
using Random
using TotalVariationImageFiltering
using CUDA

const TVIF_CUDA_GSTV = TotalVariationImageFiltering

@testset "CUDA periodic and group operators" begin
    CUDA.allowscalar(false)
    Random.seed!(701)
    for (shape, group_shape) in (((9, 7), (2, 3)), ((7, 6, 5), (3, 2, 1)))
        x_cpu = randn(Float32, shape...)
        x_gpu = CUDA.cu(x_cpu)
        for boundary in (TVIF_CUDA_GSTV.Neumann(), TVIF_CUDA_GSTV.Periodic())
            out_cpu = similar(x_cpu)
            scratch_cpu = similar(x_cpu)
            out_gpu = similar(x_gpu)
            scratch_gpu = similar(x_gpu)
            TVIF_CUDA_GSTV.group_sum!(out_cpu, scratch_cpu, x_cpu, group_shape, boundary)
            TVIF_CUDA_GSTV.group_sum!(out_gpu, scratch_gpu, x_gpu, group_shape, boundary)
            @test isapprox(Array(out_gpu), out_cpu; atol = 1.0f-5, rtol = 1.0f-6)

            gradient_cpu = TVIF_CUDA_GSTV.allocate_dual(x_cpu)
            gradient_gpu = TVIF_CUDA_GSTV.allocate_dual(x_gpu)
            TVIF_CUDA_GSTV.gradient!(gradient_cpu, x_cpu, boundary)
            TVIF_CUDA_GSTV.gradient!(gradient_gpu, x_gpu, boundary)
            @test all(
                isapprox(Array(gradient_gpu[d]), gradient_cpu[d]; atol = 1.0f-6, rtol = 0)
                for d = 1:ndims(x_cpu)
            )
        end
    end
end

@testset "CUDA GSTV single 2D and 3D" begin
    CUDA.allowscalar(false)
    Random.seed!(709)
    config = TVIF_CUDA_GSTV.GSTVConfig(
        maxiter = 500,
        tol = 2.0f-4,
        mm_maxiter = 100,
        mm_tol = 2.0f-5,
        cg_maxiter = 100,
        cg_tol = 2.0f-5,
    )

    cases = (
        (randn(Float32, 18, 14), (2, 3), TVIF_CUDA_GSTV.Neumann()),
        (randn(Float32, 9, 8, 7), (3, 2, 1), TVIF_CUDA_GSTV.Periodic()),
    )
    for (f_cpu, group_shape, boundary) in cases
        cpu_problem = TVIF_CUDA_GSTV.TVProblem(
            f_cpu;
            lambda = 0.07f0,
            tv_mode = TVIF_CUDA_GSTV.GroupSparseTV(group_shape),
            boundary = boundary,
        )
        gpu_problem = TVIF_CUDA_GSTV.TVProblem(
            CUDA.cu(f_cpu);
            lambda = 0.07f0,
            tv_mode = TVIF_CUDA_GSTV.GroupSparseTV(group_shape),
            boundary = boundary,
        )
        u_cpu, stats_cpu = TVIF_CUDA_GSTV.solve(cpu_problem, config)
        u_gpu, stats_gpu = TVIF_CUDA_GSTV.solve(gpu_problem, config)
        @test stats_cpu.converged
        @test stats_gpu.converged
        @test isapprox(Array(u_gpu), u_cpu; atol = 2.0f-3, rtol = 2.0f-3)

        state_gpu = TVIF_CUDA_GSTV.GSTVState(gpu_problem.f, gpu_problem.tv_mode)
        reused, reused_stats = TVIF_CUDA_GSTV.solve(
            gpu_problem,
            config;
            init = CUDA.fill(3.0f0, size(f_cpu)),
            state = state_gpu,
        )
        @test reused_stats.converged
        @test isapprox(Array(reused), u_cpu; atol = 3.0f-3, rtol = 3.0f-3)
    end
end

@testset "CUDA GSTV sequential batch" begin
    CUDA.allowscalar(false)
    Random.seed!(719)
    f_cpu = randn(Float32, 14, 11, 3)
    f_gpu = CUDA.cu(f_cpu)
    config = TVIF_CUDA_GSTV.GSTVConfig(
        maxiter = 500,
        tol = 2.0f-4,
        mm_maxiter = 100,
        mm_tol = 2.0f-5,
        cg_maxiter = 100,
        cg_tol = 2.0f-5,
    )
    u_cpu, cpu_summary, cpu_items = TVIF_CUDA_GSTV.solve_batch(
        f_cpu,
        config;
        lambda = 0.06f0,
        tv_mode = TVIF_CUDA_GSTV.GroupSparseTV((3, 2)),
        boundary = TVIF_CUDA_GSTV.Periodic(),
        return_per_item_stats = true,
    )
    state_gpu = TVIF_CUDA_GSTV.GSTVBatchState(f_gpu, TVIF_CUDA_GSTV.GroupSparseTV((3, 2)))
    u_gpu, gpu_summary, gpu_items = TVIF_CUDA_GSTV.solve_batch(
        f_gpu,
        config;
        lambda = 0.06f0,
        tv_mode = TVIF_CUDA_GSTV.GroupSparseTV((3, 2)),
        boundary = TVIF_CUDA_GSTV.Periodic(),
        state = state_gpu,
        return_per_item_stats = true,
    )
    @test cpu_summary.converged
    @test gpu_summary.converged
    @test length(cpu_items) == length(gpu_items) == 3
    @test isapprox(Array(u_gpu), u_cpu; atol = 2.0f-3, rtol = 2.0f-3)
end

@testset "CUDA periodic standard TV" begin
    CUDA.allowscalar(false)
    Random.seed!(727)
    f_cpu = randn(Float32, 16, 13)
    f_gpu = CUDA.cu(f_cpu)
    for config in (
        TVIF_CUDA_GSTV.ROFConfig(maxiter = 1500, tol = 1.0f-5),
        TVIF_CUDA_GSTV.PDHGConfig(maxiter = 1500, tau = 0.2f0, sigma = 0.2f0, tol = 1.0f-5),
    )
        cpu_problem = TVIF_CUDA_GSTV.TVProblem(
            f_cpu;
            lambda = 0.1f0,
            boundary = TVIF_CUDA_GSTV.Periodic(),
        )
        gpu_problem = TVIF_CUDA_GSTV.TVProblem(
            f_gpu;
            lambda = 0.1f0,
            boundary = TVIF_CUDA_GSTV.Periodic(),
        )
        u_cpu, _ = TVIF_CUDA_GSTV.solve(cpu_problem, config)
        u_gpu, _ = TVIF_CUDA_GSTV.solve(gpu_problem, config)
        @test isapprox(Array(u_gpu), u_cpu; atol = 2.0f-3, rtol = 2.0f-3)
    end
end
