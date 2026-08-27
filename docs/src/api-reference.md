# API Reference

```@meta
CurrentModule = TotalVariationImageFiltering
```

`TotalVariationImageFiltering.jl` does not currently export symbols, so call APIs with the
`TotalVariationImageFiltering.` prefix in user code.

## Module

```@docs
TotalVariationImageFiltering
```

## Types

```@docs
AbstractBoundaryCondition
Neumann
Periodic
AbstractDataFidelity
L2Fidelity
PoissonFidelity
AbstractPrimalConstraint
NoConstraint
NonnegativeConstraint
BoxConstraint
AbstractTVMode
IsotropicTV
AnisotropicTV
GroupSparseTV
AbstractTVSolver
TVProblem
SolverStats
ROFConfig
ROFState
PDHGConfig
PDHGState
GSTVConfig
GSTVState
GSTVBatchState
DiscrepancySelection
SURESelection
```

## Solvers

```@docs
solve
solve!
solve_batch
```

## Lambda Selection

```@docs
select_lambda_discrepancy
select_lambda_sure
```

## Operator Utilities

```@docs
allocate_dual
gradient!
divergence!
project_dual_ball!
group_sum!
group_sum_adjoint!
```
