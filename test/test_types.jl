using Test
using TotalVariationImageFiltering

@testset "Type Hierarchy" begin
    @test TotalVariationImageFiltering.Neumann <:
          TotalVariationImageFiltering.AbstractBoundaryCondition
    @test TotalVariationImageFiltering.Periodic <:
          TotalVariationImageFiltering.AbstractBoundaryCondition
    @test TotalVariationImageFiltering.L2Fidelity <:
          TotalVariationImageFiltering.AbstractDataFidelity
    @test TotalVariationImageFiltering.PoissonFidelity <:
          TotalVariationImageFiltering.AbstractDataFidelity
    @test TotalVariationImageFiltering.NoConstraint <:
          TotalVariationImageFiltering.AbstractPrimalConstraint
    @test TotalVariationImageFiltering.NonnegativeConstraint <:
          TotalVariationImageFiltering.AbstractPrimalConstraint
    @test TotalVariationImageFiltering.BoxConstraint <:
          TotalVariationImageFiltering.AbstractPrimalConstraint
    @test TotalVariationImageFiltering.IsotropicTV <:
          TotalVariationImageFiltering.AbstractTVMode
    @test TotalVariationImageFiltering.AnisotropicTV <:
          TotalVariationImageFiltering.AbstractTVMode
    @test TotalVariationImageFiltering.GroupSparseTV <:
          TotalVariationImageFiltering.AbstractTVMode
    @test TotalVariationImageFiltering.ROFConfig <:
          TotalVariationImageFiltering.AbstractTVSolver
    @test TotalVariationImageFiltering.PDHGConfig <:
          TotalVariationImageFiltering.AbstractTVSolver
    @test TotalVariationImageFiltering.GSTVConfig <:
          TotalVariationImageFiltering.AbstractTVSolver
end

@testset "Constraint Constructors" begin
    no_constraint = TotalVariationImageFiltering.NoConstraint()
    @test no_constraint isa TotalVariationImageFiltering.NoConstraint

    nn_constraint = TotalVariationImageFiltering.NonnegativeConstraint()
    @test nn_constraint isa TotalVariationImageFiltering.NonnegativeConstraint

    box = TotalVariationImageFiltering.BoxConstraint(-1.0, 2.5f0)
    @test box isa TotalVariationImageFiltering.BoxConstraint{Float64}
    @test box.lower == -1.0
    @test box.upper == 2.5

    @test_throws ArgumentError TotalVariationImageFiltering.BoxConstraint(1.0, -1.0)
    @test_throws ArgumentError TotalVariationImageFiltering.BoxConstraint(NaN, 1.0)
    @test_throws ArgumentError TotalVariationImageFiltering.BoxConstraint(-1.0, NaN)
end

@testset "GroupSparseTV Constructors" begin
    @test TotalVariationImageFiltering.GroupSparseTV().group_shape == 3
    @test TotalVariationImageFiltering.GroupSparseTV(5).group_shape == 5
    @test TotalVariationImageFiltering.GroupSparseTV((2, 3)).group_shape == (2, 3)
    @test_throws ArgumentError TotalVariationImageFiltering.GroupSparseTV(0)
    @test_throws ArgumentError TotalVariationImageFiltering.GroupSparseTV(())
    @test_throws ArgumentError TotalVariationImageFiltering.GroupSparseTV((3, -1))
end

@testset "Common Config Validation" begin
    @test TotalVariationImageFiltering._validate_common_config(1, 1) === nothing
    @test_throws ArgumentError TotalVariationImageFiltering._validate_common_config(0, 1)
    @test_throws ArgumentError TotalVariationImageFiltering._validate_common_config(1, 0)
end

@testset "PDHGConfig Constructor and Validation" begin
    cfg_default = TotalVariationImageFiltering.PDHGConfig()
    @test cfg_default.maxiter == 500
    @test cfg_default.tau == 0.25
    @test cfg_default.sigma == 0.25
    @test cfg_default.theta == 1.0
    @test cfg_default.tol == 1e-4
    @test cfg_default.check_every == 10
    @test TotalVariationImageFiltering._validate(cfg_default) === nothing

    cfg_promoted =
        TotalVariationImageFiltering.PDHGConfig(tau = Float32(0.2), sigma = 0.1, tol = 1e-6)
    @test cfg_promoted isa TotalVariationImageFiltering.PDHGConfig{Float64}

    cfg_float32 = TotalVariationImageFiltering.PDHGConfig(
        tau = Float32(0.2),
        sigma = Float32(0.1),
        theta = Float32(1),
        tol = Float32(1e-3),
    )
    @test cfg_float32 isa TotalVariationImageFiltering.PDHGConfig{Float32}

    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(maxiter = 0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(check_every = 0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(tau = 0.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(sigma = 0.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(theta = -0.1),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(theta = 1.1),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.PDHGConfig(tol = -1e-6),
    )
end

@testset "SolverStats Shape" begin
    stats = TotalVariationImageFiltering.SolverStats{Float64}(17, true, 1e-6)
    @test stats.iterations == 17
    @test stats.converged
    @test stats.rel_change == 1e-6
end

@testset "ROFConfig Constructor and Validation" begin
    cfg_default = TotalVariationImageFiltering.ROFConfig()
    @test cfg_default.maxiter == 300
    @test cfg_default.tau == 0.0625
    @test cfg_default.tol == 1e-4
    @test cfg_default.check_every == 10
    @test TotalVariationImageFiltering._validate(cfg_default) === nothing

    cfg_promoted = TotalVariationImageFiltering.ROFConfig(tau = Float32(0.2), tol = 1e-6)
    @test cfg_promoted isa TotalVariationImageFiltering.ROFConfig{Float64}

    cfg_float32 =
        TotalVariationImageFiltering.ROFConfig(tau = Float32(0.2), tol = Float32(1e-3))
    @test cfg_float32 isa TotalVariationImageFiltering.ROFConfig{Float32}

    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.ROFConfig(maxiter = 0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.ROFConfig(check_every = 0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.ROFConfig(tau = 0.0),
    )
    @test_throws ArgumentError TotalVariationImageFiltering._validate(
        TotalVariationImageFiltering.ROFConfig(tol = -1e-6),
    )
end


@testset "GSTVConfig Constructor and Validation" begin
    cfg = TotalVariationImageFiltering.GSTVConfig()
    @test cfg.maxiter == 300
    @test cfg.rho == 1.0
    @test cfg.tol == 1e-4
    @test cfg.check_every == 5
    @test cfg.mm_maxiter == 20
    @test cfg.mm_tol == 1e-5
    @test cfg.cg_maxiter == 50
    @test cfg.cg_tol == 1e-5
    @test TotalVariationImageFiltering._validate(cfg) === nothing

    cfg32 = TotalVariationImageFiltering.GSTVConfig(
        rho = 1.0f0,
        tol = 1.0f-4,
        mm_tol = 1.0f-5,
        cg_tol = 1.0f-5,
    )
    @test cfg32 isa TotalVariationImageFiltering.GSTVConfig{Float32}

    for bad in (
        TotalVariationImageFiltering.GSTVConfig(maxiter = 0),
        TotalVariationImageFiltering.GSTVConfig(rho = 0),
        TotalVariationImageFiltering.GSTVConfig(tol = -1),
        TotalVariationImageFiltering.GSTVConfig(check_every = 0),
        TotalVariationImageFiltering.GSTVConfig(mm_maxiter = 0),
        TotalVariationImageFiltering.GSTVConfig(mm_tol = -1),
        TotalVariationImageFiltering.GSTVConfig(cg_maxiter = 0),
        TotalVariationImageFiltering.GSTVConfig(cg_tol = -1),
    )
        @test_throws ArgumentError TotalVariationImageFiltering._validate(bad)
    end
end
