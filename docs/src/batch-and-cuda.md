# Batch & CUDA

## Batch API (CPU and Generic Arrays)

Use `solve_batch` for input arrays shaped:

```text
(spatial_dim_1, spatial_dim_2, ..., spatial_dim_k, batch)
```

The last axis is treated as batch index, and TV operators are applied only
across spatial axes.

Example:

```julia
u_batch, stats = TotalVariationImageFiltering.solve_batch(
    f_batch,
    TotalVariationImageFiltering.ROFConfig();
    lambda = 0.1,
    tv_mode = TotalVariationImageFiltering.IsotropicTV(),
)
```

For PDHG batch solves, you can additionally pass:

- `constraint = TotalVariationImageFiltering.NonnegativeConstraint()`, or
- `constraint = TotalVariationImageFiltering.BoxConstraint(lower, upper)`.

Batch state reuse:

- pass `state = [ROFState(slice1), ROFState(slice2), ...]` for ROF;
- pass `state = [PDHGState(slice1), PDHGState(slice2), ...]` for PDHG.
- pass one `GSTVBatchState(f_batch, mode)` for GSTV; it processes final-axis
  items sequentially through one contiguous spatial workspace.

ROF and PDHG state-vector lengths must match the batch size.

## CUDA Extension

The extension module `TotalVariationImageFilteringCUDAExt` is loaded automatically when:

- `CUDA.jl` is installed and loaded,
- a functional CUDA runtime/device is available.

Example:

```julia
using CUDA
using TotalVariationImageFiltering

f_gpu = CUDA.rand(Float32, 256, 256)
problem_gpu = TotalVariationImageFiltering.TVProblem(f_gpu; lambda = 0.15f0)
u_gpu, stats_gpu = TotalVariationImageFiltering.solve(problem_gpu, TotalVariationImageFiltering.ROFConfig())
```

## CUDA Coverage

Current behavior based on extension code/tests:

- CUDA kernels are provided for gradient/divergence/projection and separable
  overlapping-group sums.
- Single-image ROF, PDHG, and GSTV solves on `CuArray` are supported.
- Batched CUDA solve is specialized for `ROFConfig` and `PDHGConfig`;
  `GSTVConfig` uses the bounded-memory sequential workspace described above.
- Batched CUDA path currently requires:
  - `L2Fidelity` (ROF), or `L2Fidelity` / `PoissonFidelity` (PDHG),
  - `L2Fidelity` and `NoConstraint()` (GSTV),
  - `Neumann()` or `Periodic()` boundary.

ROF paths currently support only `constraint = NoConstraint()`.

If CUDA is unavailable, CPU paths continue to work.

## Numerical Equivalence Checks

Repository tests compare CPU and CUDA outputs with tolerances for:

- 2D and 3D gradients and group sums under both boundaries;
- single-image ROF, PDHG, and GSTV solves;
- GSTV state reuse and sequential batch solves.
