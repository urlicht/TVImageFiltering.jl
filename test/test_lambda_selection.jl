using Test
using Random
using TotalVariationImageFiltering

@testset "Discrepancy Lambda Selection (2D and 3D)" begin
    Random.seed!(701)

    clean2d = zeros(Float64, 32, 32)
    clean2d[9:24, 9:24] .= 1.0
    sigma2d = 0.1
    noisy2d = clean2d .+ sigma2d * randn(Float64, size(clean2d)...)

    cfg2d = TotalVariationImageFiltering.ROFConfig(
        maxiter = 1000,
        tau = 0.0625,
        tol = 1e-6,
        check_every = 10,
    )
    sel2d = TotalVariationImageFiltering.select_lambda_discrepancy(
        noisy2d,
        cfg2d;
        sigma = sigma2d,
        lambda_min = 0.0,
        lambda_max = 0.2,
        rtol = 0.05,
        max_bisect = 25,
    )

    @test sel2d.lambda >= 0.0
    @test sel2d.evaluations >= 2
    @test isfinite(sel2d.residual_norm2)
    @test sel2d.relative_mismatch <= 0.25
    @test size(sel2d.u) == size(noisy2d)

    noisy_mse2d = sum(abs2, noisy2d .- clean2d) / length(clean2d)
    denoised_mse2d = sum(abs2, sel2d.u .- clean2d) / length(clean2d)
    @test denoised_mse2d < noisy_mse2d

    clean3d = zeros(Float32, 16, 16, 8)
    clean3d[5:12, 5:12, 3:6] .= 1.0f0
    sigma3d = 0.08f0
    noisy3d = clean3d .+ sigma3d * randn(Float32, size(clean3d)...)
    cfg3d = TotalVariationImageFiltering.ROFConfig(
        maxiter = 1200,
        tau = 0.04,
        tol = 1.0f-5,
        check_every = 20,
    )
    sel3d = TotalVariationImageFiltering.select_lambda_discrepancy(
        noisy3d,
        cfg3d;
        sigma = sigma3d,
        lambda_min = 0.0f0,
        lambda_max = 0.3f0,
        rtol = 0.08f0,
    )
    @test size(sel3d.u) == size(noisy3d)
    @test sel3d.lambda >= 0.0f0
    @test isfinite(sel3d.relative_mismatch)
end

@testset "MC-SURE Lambda Selection" begin
    Random.seed!(709)

    clean = zeros(Float64, 32, 32)
    clean[7:26, 7:26] .= 1.0
    clean[14:19, 14:19] .= 0.2
    sigma = 0.15
    noisy = clean .+ sigma * randn(Float64, size(clean)...)

    cfg = TotalVariationImageFiltering.ROFConfig(
        maxiter = 900,
        tau = 0.0625,
        tol = 1e-5,
        check_every = 10,
    )
    grid = [0.0, 0.04, 0.08, 0.12, 0.18]
    sel = TotalVariationImageFiltering.select_lambda_sure(
        noisy,
        cfg;
        sigma = sigma,
        lambda_grid = grid,
        mc_samples = 2,
        rng = MersenneTwister(5),
    )

    @test sel.lambda in sel.lambda_grid
    @test length(sel.sure_values) == length(grid)
    @test length(sel.residual_values) == length(grid)
    @test length(sel.divergence_values) == length(grid)
    @test sel.evaluations == length(grid) * 3
    @test all(isfinite, sel.sure_values)
    @test isfinite(sel.sure)
    @test isfinite(sel.epsilon)

    noisy_mse = sum(abs2, noisy .- clean) / length(clean)
    denoised_mse = sum(abs2, sel.u .- clean) / length(clean)
    @test denoised_mse < noisy_mse
end

@testset "Lambda Selection Validation" begin
    f = randn(Float64, 8, 8)
    @test_throws ArgumentError TotalVariationImageFiltering.select_lambda_discrepancy(
        f;
        sigma = -1.0,
    )
    @test_throws ArgumentError TotalVariationImageFiltering.select_lambda_sure(
        f;
        sigma = 0.1,
        lambda_grid = Float64[],
    )
    @test_throws ArgumentError TotalVariationImageFiltering.select_lambda_sure(
        f;
        sigma = 0.1,
        lambda_grid = [0.1],
        mc_samples = 0,
    )
end
