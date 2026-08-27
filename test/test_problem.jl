using Test
using Random
using TotalVariationImageFiltering

struct DummyBoundaryForProblem <: TotalVariationImageFiltering.AbstractBoundaryCondition end
struct DummyDataFidelityForProblem <: TotalVariationImageFiltering.AbstractDataFidelity end
struct DummyTVModeForProblem <: TotalVariationImageFiltering.AbstractTVMode end
struct DummyConstraintForProblem <: TotalVariationImageFiltering.AbstractPrimalConstraint end

@testset "TVProblem Construction and Defaults" begin
    Random.seed!(11)
    f = randn(Float32, 3, 4)
    prob = TotalVariationImageFiltering.TVProblem(f; lambda = 2)

    @test prob.f === f
    @test prob.lambda isa Float32
    @test prob.lambda == 2.0f0
    @test prob.spacing == (1.0f0, 1.0f0)
    @test prob.data_fidelity isa TotalVariationImageFiltering.L2Fidelity
    @test prob.tv_mode isa TotalVariationImageFiltering.IsotropicTV
    @test prob.boundary isa TotalVariationImageFiltering.Neumann
    @test prob.constraint isa TotalVariationImageFiltering.NoConstraint
end

@testset "TVProblem Custom Components" begin
    f = randn(Float64, 4, 2)
    df = DummyDataFidelityForProblem()
    tv = DummyTVModeForProblem()
    bc = DummyBoundaryForProblem()
    constraint = TotalVariationImageFiltering.BoxConstraint(-2.0, 1.0)

    prob = TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.3,
        spacing = (0.5, 3.0),
        data_fidelity = df,
        tv_mode = tv,
        boundary = bc,
        constraint = constraint,
    )

    @test prob.data_fidelity === df
    @test prob.tv_mode === tv
    @test prob.boundary === bc
    @test prob.constraint isa TotalVariationImageFiltering.BoxConstraint{Float64}
    @test prob.constraint.lower == -2.0
    @test prob.constraint.upper == 1.0
    @test prob.spacing == (0.5, 3.0)
end

@testset "TVProblem Constraint Normalization and Validation" begin
    f32 = randn(Float32, 5, 4)
    prob_box32 = TotalVariationImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = TotalVariationImageFiltering.BoxConstraint(-1.0, 2.0),
    )
    @test prob_box32.constraint isa TotalVariationImageFiltering.BoxConstraint{Float32}
    @test prob_box32.constraint.lower == -1.0f0
    @test prob_box32.constraint.upper == 2.0f0

    prob_nn = TotalVariationImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = TotalVariationImageFiltering.NonnegativeConstraint(),
    )
    @test prob_nn.constraint isa TotalVariationImageFiltering.NonnegativeConstraint

    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = DummyConstraintForProblem(),
    )
end

@testset "TVProblem Spacing Normalization" begin
    f = randn(Float64, 3, 4, 2)

    prob_default = TotalVariationImageFiltering.TVProblem(f; lambda = 0.1)
    @test prob_default.spacing == (1.0, 1.0, 1.0)

    prob_scalar = TotalVariationImageFiltering.TVProblem(f; lambda = 0.1, spacing = 2)
    @test prob_scalar.spacing == (2.0, 2.0, 2.0)

    prob_tuple =
        TotalVariationImageFiltering.TVProblem(f; lambda = 0.1, spacing = (0.5, 2.0, 4.0))
    @test prob_tuple.spacing == (0.5, 2.0, 4.0)

    prob_vector =
        TotalVariationImageFiltering.TVProblem(f; lambda = 0.1, spacing = [0.5, 2.0, 4.0])
    @test prob_vector.spacing == (0.5, 2.0, 4.0)

    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = [1.0, 2.0],
    )
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, 0.0, 2.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, -1.0, 2.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, Inf, 2.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = "invalid",
    )
end

@testset "TVProblem Lambda Validation" begin
    f = randn(Float64, 3, 4)
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(f; lambda = -1.0)
end


@testset "TVProblem GroupSparseTV Normalization" begin
    f2 = zeros(Float32, 8, 6)
    scalar = TotalVariationImageFiltering.TVProblem(
        f2;
        lambda = 0.1,
        tv_mode = TotalVariationImageFiltering.GroupSparseTV(3),
    )
    @test scalar.tv_mode.group_shape == (3, 3)

    tuple_mode = TotalVariationImageFiltering.TVProblem(
        zeros(7, 6, 5);
        lambda = 0.1,
        tv_mode = TotalVariationImageFiltering.GroupSparseTV((2, 3, 4)),
        boundary = TotalVariationImageFiltering.Periodic(),
    )
    @test tuple_mode.tv_mode.group_shape == (2, 3, 4)
    @test tuple_mode.boundary isa TotalVariationImageFiltering.Periodic

    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f2;
        lambda = 0.1,
        tv_mode = TotalVariationImageFiltering.GroupSparseTV((3, 3, 3)),
    )
    @test_throws ArgumentError TotalVariationImageFiltering.TVProblem(
        f2;
        lambda = 0.1,
        tv_mode = TotalVariationImageFiltering.GroupSparseTV((9, 3)),
    )
end
