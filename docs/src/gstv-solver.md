# Group-Sparse TV Solver

`GSTVConfig` solves finite, scalar L2 denoising problems with overlapping
group-sparse anisotropic total variation:

```math
\min_u \frac12\|u-f\|_2^2
+ \lambda \sum_d\sum_i
  \sqrt{\sum_{o\in G_i}|(\nabla_d u)_{i+o}|^2}.
```

The spatial group is applied separately to every directional derivative. A
group shape of one along every axis is equivalent to `AnisotropicTV()`.

## 2D and 3D usage

An integer group size expands across all spatial axes:

```julia
mode_2d = TotalVariationImageFiltering.GroupSparseTV(3) # (3, 3)
mode_3d = TotalVariationImageFiltering.GroupSparseTV(3) # (3, 3, 3)
```

Axis-specific shapes are also supported:

```julia
image = rand(Float32, 512, 512)
problem_2d = TotalVariationImageFiltering.TVProblem(
    image;
    lambda = 0.06f0,
    tv_mode = TotalVariationImageFiltering.GroupSparseTV((3, 5)),
    boundary = TotalVariationImageFiltering.Periodic(),
)
u_2d, stats_2d = TotalVariationImageFiltering.solve(
    problem_2d,
    TotalVariationImageFiltering.GSTVConfig(),
)
```

For a 3D volume, the same code path accepts a three-entry group and physical
voxel spacing:

```julia
mode = TotalVariationImageFiltering.GroupSparseTV((3, 3, 1))
problem = TotalVariationImageFiltering.TVProblem(
    volume;
    lambda = 0.08f0,
    spacing = (0.4f0, 0.4f0, 1.5f0),
    tv_mode = mode,
    boundary = TotalVariationImageFiltering.Neumann(),
)
u, stats = TotalVariationImageFiltering.solve(
    problem,
    TotalVariationImageFiltering.GSTVConfig(),
)
```

Every group dimension must be positive and no larger than the corresponding
data axis. Even group sizes use offsets
`-floor((k-1)/2):floor(k/2)`.

## Boundaries

- `Neumann()` uses the existing zero-normal finite difference and truncates
  groups at image or volume boundaries.
- `Periodic()` wraps both finite differences and group neighborhoods.

## Algorithm and convergence

The solver uses ADMM with one split field per spatial derivative. Each
overlapping-group proximal map is computed by majorization-minimization, and
the quadratic primal update is solved by warm-started matrix-free conjugate
gradient. Group sums are separable, so storage is `O(ndims(u) * length(u))`
rather than one dual volume per group element.

`SolverStats.converged` is true only when the outer normalized primal/dual
criterion and the last CG/MM inner solves meet their configured tolerances. If
the output is stable but `converged` is false, increase `mm_maxiter` first for
groups larger than one.

## State, batches, and CUDA

Use `GSTVState(problem.f, problem.tv_mode)` for allocation reuse and warm
starts. Batched arrays use the last axis as batch and are solved sequentially
through `GSTVBatchState`, so peak workspace does not scale with batch count.

CPU and CUDA arrays use the same API. CUDA group operations and periodic
differences are implemented by package-extension kernels.

## Limitations

`GSTVConfig` currently supports only `L2Fidelity()`, `NoConstraint()`, and
scalar arrays. A three-dimensional `height × width × channels` array is treated
as a spatial volume, not as a multichannel 2D image. Automatic lambda selection
is currently limited to ROF.
