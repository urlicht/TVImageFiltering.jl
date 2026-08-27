using Test
using Random
using TotalVariationImageFiltering

@testset "allocate_dual" begin
    u = randn(Float32, 2, 3, 4)
    p = TotalVariationImageFiltering.allocate_dual(u)

    @test length(p) == 3
    @test all(size(pd) == size(u) for pd in p)
    @test all(eltype(pd) == Float32 for pd in p)
    @test p[1] !== p[2]
    @test p[2] !== p[3]

    fill!(p[1], 1.0f0)
    @test !all(p[2] .== 1.0f0)
end

@testset "gradient! exact values" begin
    u1 = [1.0, 4.0, 10.0]
    g1 = TotalVariationImageFiltering.allocate_dual(u1)

    TotalVariationImageFiltering.gradient!(g1, u1, TotalVariationImageFiltering.Neumann())
    @test g1[1] == [3.0, 6.0, 0.0]

    TotalVariationImageFiltering.gradient!(
        g1,
        u1,
        TotalVariationImageFiltering.Neumann(),
        (2.0,),
    )
    @test g1[1] == [6.0, 12.0, 0.0]

    u2 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    g2 = TotalVariationImageFiltering.allocate_dual(u2)
    TotalVariationImageFiltering.gradient!(g2, u2, TotalVariationImageFiltering.Neumann())
    @test g2[1] == [3.0 3.0 3.0; 0.0 0.0 0.0]
    @test g2[2] == [1.0 1.0 0.0; 1.0 1.0 0.0]

    TotalVariationImageFiltering.gradient!(
        g2,
        u2,
        TotalVariationImageFiltering.Neumann(),
        (0.5, 2.0),
    )
    @test g2[1] == [1.5 1.5 1.5; 0.0 0.0 0.0]
    @test g2[2] == [2.0 2.0 0.0; 2.0 2.0 0.0]
end

@testset "divergence! exact values" begin
    out = zeros(3)
    p = ([2.0, -1.0, 7.5],)

    TotalVariationImageFiltering.divergence!(out, p, TotalVariationImageFiltering.Neumann())
    @test out == [2.0, -3.0, 1.0]

    TotalVariationImageFiltering.divergence!(
        out,
        p,
        TotalVariationImageFiltering.Neumann(),
        (2.0,),
    )
    @test out == [4.0, -6.0, 2.0]

    out_singleton = fill(42.0, 1)
    p_singleton = ([3.0],)
    TotalVariationImageFiltering.divergence!(
        out_singleton,
        p_singleton,
        TotalVariationImageFiltering.Neumann(),
    )
    @test out_singleton == [0.0]
end

@testset "Scaled adjointness of gradient/divergence" begin
    Random.seed!(23)
    for sz in ((7,), (3, 4), (2, 3, 4), (1, 5, 1))
        u = randn(Float64, sz...)
        n_dims = ndims(u)
        spacing = ntuple(d -> 0.5 + d, n_dims)
        inv_spacing = ntuple(d -> inv(spacing[d]), n_dims)
        p = ntuple(_ -> randn(Float64, sz...), n_dims)
        g = TotalVariationImageFiltering.allocate_dual(u)
        divp = similar(u)

        TotalVariationImageFiltering.gradient!(
            g,
            u,
            TotalVariationImageFiltering.Neumann(),
            inv_spacing,
        )
        TotalVariationImageFiltering.divergence!(
            divp,
            p,
            TotalVariationImageFiltering.Neumann(),
            inv_spacing,
        )

        lhs = sum(sum(g[d] .* p[d]) for d = 1:n_dims)
        rhs = sum(u .* divp)
        @test isapprox(lhs + rhs, 0.0; atol = 1e-10, rtol = 0.0)
    end
end

@testset "Periodic gradient/divergence" begin
    u = [1.0, 4.0, 10.0]
    g = TotalVariationImageFiltering.allocate_dual(u)
    TotalVariationImageFiltering.gradient!(g, u, TotalVariationImageFiltering.Periodic())
    @test g[1] == [3.0, 6.0, -9.0]

    p = ([2.0, -1.0, 7.5],)
    divp = similar(u)
    TotalVariationImageFiltering.divergence!(
        divp,
        p,
        TotalVariationImageFiltering.Periodic(),
    )
    @test divp == [-5.5, -3.0, 8.5]

    Random.seed!(29)
    for sz in ((7,), (4, 5), (3, 4, 5), (1, 5, 1))
        x = randn(sz...)
        q = ntuple(_ -> randn(sz...), ndims(x))
        gx = TotalVariationImageFiltering.allocate_dual(x)
        dq = similar(x)
        spacing = ntuple(d -> 0.4 + d, ndims(x))
        TotalVariationImageFiltering.gradient!(
            gx,
            x,
            TotalVariationImageFiltering.Periodic(),
            spacing,
        )
        TotalVariationImageFiltering.divergence!(
            dq,
            q,
            TotalVariationImageFiltering.Periodic(),
            spacing,
        )
        @test isapprox(
            sum(sum(gx[d] .* q[d]) for d = 1:ndims(x)),
            -sum(x .* dq);
            atol = 1e-10,
            rtol = 0,
        )
    end
end

function brute_group_sum(input, group_shape, boundary; adjoint = false)
    output = zeros(eltype(input), size(input))
    offsets = ntuple(
        d -> begin
            first, last = TotalVariationImageFiltering._group_offset_bounds(group_shape[d])
            values = collect(first:last)
            adjoint ? (-reverse(values)) : values
        end,
        ndims(input),
    )

    for center in CartesianIndices(input)
        total = zero(eltype(input))
        for offset in Iterators.product(offsets...)
            source = ntuple(d -> center[d] + offset[d], ndims(input))
            if boundary isa TotalVariationImageFiltering.Periodic
                wrapped = ntuple(d -> mod1(source[d], size(input, d)), ndims(input))
                total += input[wrapped...]
            elseif all(d -> 1 <= source[d] <= size(input, d), 1:ndims(input))
                total += input[source...]
            end
        end
        output[center] = total
    end
    return output
end

@testset "Overlapping group sums in 2D and 3D" begin
    Random.seed!(31)
    for shape_and_group in
        (((5, 4), (3, 3)), ((5, 4), (2, 3)), ((4, 5, 3), (2, 2, 2)), ((4, 5, 3), (3, 2, 1)))
        shape, group_shape = shape_and_group
        x = randn(shape...)
        y = randn(shape...)
        out = similar(x)
        adj = similar(x)
        scratch = similar(x)
        for boundary in (
            TotalVariationImageFiltering.Neumann(),
            TotalVariationImageFiltering.Periodic(),
        )
            TotalVariationImageFiltering.group_sum!(out, scratch, x, group_shape, boundary)
            @test isapprox(
                out,
                brute_group_sum(x, group_shape, boundary);
                atol = 1e-12,
                rtol = 0,
            )

            TotalVariationImageFiltering.group_sum_adjoint!(
                adj,
                scratch,
                y,
                group_shape,
                boundary,
            )
            @test isapprox(sum(out .* y), sum(x .* adj); atol = 1e-10, rtol = 0)
        end
    end

    x = randn(4, 4)
    @test_throws ArgumentError TotalVariationImageFiltering.group_sum!(
        x,
        similar(x),
        x,
        (3, 3),
        TotalVariationImageFiltering.Neumann(),
    )
end

@testset "project_dual_ball! isotropic and anisotropic" begin
    p_iso = ([3.0, 0.6, -0.3], [4.0, 0.8, 0.4])
    TotalVariationImageFiltering.project_dual_ball!(
        p_iso,
        1.0,
        TotalVariationImageFiltering.IsotropicTV(),
    )
    @test isapprox(p_iso[1][1], 0.6; atol = 1e-12)
    @test isapprox(p_iso[2][1], 0.8; atol = 1e-12)
    @test p_iso[1][2] == 0.6
    @test p_iso[2][2] == 0.8
    @test p_iso[1][3] == -0.3
    @test p_iso[2][3] == 0.4

    p_aniso = ([-2.0, -0.2, 0.3, 1.7], [0.5, -3.1, 2.2, -0.9])
    TotalVariationImageFiltering.project_dual_ball!(
        p_aniso,
        1.0,
        TotalVariationImageFiltering.AnisotropicTV(),
    )
    @test p_aniso[1] == [-1.0, -0.2, 0.3, 1.0]
    @test p_aniso[2] == [0.5, -1.0, 1.0, -0.9]
end
