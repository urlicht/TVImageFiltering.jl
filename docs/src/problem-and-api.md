# Problem & API

## Problem Definition

Use `TVProblem` to define a denoising/reconstruction task:

```julia
problem = TotalVariationImageFiltering.TVProblem(
    f;
    lambda,
    spacing = nothing,
    data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
    tv_mode = TotalVariationImageFiltering.IsotropicTV(),
    boundary = TotalVariationImageFiltering.Neumann(),
    constraint = TotalVariationImageFiltering.NoConstraint(),
)
```

Key inputs:

- `f`: `AbstractArray{<:AbstractFloat,N}` data.
- `lambda`: TV weight (`>= 0`).
- `spacing`: `nothing`, scalar, `NTuple{N}`, or length-`N` vector.
- `data_fidelity`: `L2Fidelity()` or `PoissonFidelity()`.
- `tv_mode`: `IsotropicTV()`, `AnisotropicTV()`, or `GroupSparseTV(...)`.
- `boundary`: `Neumann()` or `Periodic()`.
- `constraint`: `NoConstraint()`, `NonnegativeConstraint()`, or `BoxConstraint(lower, upper)`.

`TVProblem` is immutable, but it stores `f` by reference. For repeated solves
with the same shape/eltype, update `f` in place (for example `copyto!`) and
reuse the same `TVProblem`.

## Solving a Single Input

Allocate output internally:

```julia
u, stats = TotalVariationImageFiltering.solve(problem, TotalVariationImageFiltering.ROFConfig())
```

Or in place:

```julia
u = copy(problem.f)
stats = TotalVariationImageFiltering.solve!(u, problem, TotalVariationImageFiltering.ROFConfig())
```

`stats` is:

```julia
TotalVariationImageFiltering.SolverStats(iterations, converged, rel_change)
```

Supported solvers:

- `ROFConfig` (ROF model with `L2Fidelity`).
- `PDHGConfig` (`L2Fidelity` and `PoissonFidelity`, with optional primal constraints).
- `GSTVConfig` (`L2Fidelity`, `GroupSparseTV`, and no primal constraint).

`ROFConfig` currently supports only `constraint = NoConstraint()`.

## Warm Starts and State Reuse

For repeated solves, pass reusable workspace:

```julia
state = TotalVariationImageFiltering.ROFState(problem.f)   # or PDHGState(problem.f)
u = copy(problem.f)
stats = TotalVariationImageFiltering.solve!(u, problem, TotalVariationImageFiltering.ROFConfig(); state = state)
```

Behavior:

- For `PDHGConfig`, `u` provides primal warm start.
- For `ROFConfig`, `state.p` is the effective warm start; `u` is copied into the
  solver buffer and mainly affects the first relative-change check.
- `state.p` (dual variable) is retained across calls.
- State shape/eltype must match the solve buffer.

For GSTV, construct `GSTVState(problem.f, problem.tv_mode)`. Its ADMM split and
scaled-dual fields are retained between calls. `GSTVBatchState` instead owns one
contiguous spatial workspace and resets split variables between sequential batch
items, keeping peak workspace independent of batch size.

## Batch API

For arrays shaped `(spatial..., batch)`:

```julia
u_batch, stats = TotalVariationImageFiltering.solve_batch(
    f_batch,
    TotalVariationImageFiltering.PDHGConfig();
    lambda = 0.1,
    data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
    constraint = TotalVariationImageFiltering.BoxConstraint(0.0, 1.0),
)
```

TV operators act on spatial axes only; the last axis is batch index.

The batch `stats` aggregate per-sample results:

- `iterations`: max iterations across samples.
- `converged`: true only if all samples converged.
- `rel_change`: max relative change across samples.
