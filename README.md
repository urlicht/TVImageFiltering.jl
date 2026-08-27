# TotalVariationImageFiltering.jl

[![CI](https://github.com/urlicht/TotalVariationImageFiltering.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/urlicht/TotalVariationImageFiltering.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://urlicht.github.io/TotalVariationImageFiltering.jl/)

`TotalVariationImageFiltering.jl` is a Julia package for total-variation (TV) denoising and
reconstruction on `N`-dimensional arrays.

Original implementation: [GPUFilter.jl](https://github.com/flavell-lab/GPUFilter.jl)

![comparison: original, noisy, denoised](docs/img/demo1.png)
![x slice showing a comparison: original, noisy, denoised](docs/img/demo2.png)

## Documentation

Full manual: [https://urlicht.github.io/TotalVariationImageFiltering.jl/](https://urlicht.github.io/TotalVariationImageFiltering.jl/)

Recommended entry points:

- [Quick Start](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/quick-start/)
- [Problem & API](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/problem-and-api/)
- [ROF Solver](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/rof-solver/)
- [PDHG Solver](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/pdhg-solver/)
- [Lambda Selection](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/lambda-selection/)
- [Batch & CUDA](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/batch-and-cuda/)
- [API Reference](https://urlicht.github.io/TotalVariationImageFiltering.jl/stable/api-reference/)

## Features

- ROF denoising (`L2 + TV`) with a Chambolle-style dual projected-gradient method
- PDHG / Chambolle-Pock for `L2 + TV` and Poisson `KL + TV`
- PDHG primal constraints: non-negativity and box constraints
- Isotropic and anisotropic TV
- Overlapping group-sparse anisotropic TV for 2D images and 3D volumes
- Neumann and periodic boundaries
- Single-image and batched solves
- Automatic lambda selection for ROF (discrepancy principle and MC-SURE)
- Optional CUDA acceleration via package extension

## Installation
From the registry:
```julia
] add TotalVariationImageFiltering
```

From this repository:

```julia
julia --project=.
```

From another Julia environment (local path):

```julia
import Pkg
Pkg.develop(path="/absolute/path/to/TotalVariationImageFiltering.jl")
```

From a hosted repository:

```julia
import Pkg
Pkg.add(url="https://github.com/urlicht/TotalVariationImageFiltering.jl")
```

## Minimal Example

```julia
using TotalVariationImageFiltering

f = rand(Float32, 128, 128)
problem = TotalVariationImageFiltering.TVProblem(
    f;
    lambda = 0.1f0,
    data_fidelity = TotalVariationImageFiltering.L2Fidelity(),
    tv_mode = TotalVariationImageFiltering.IsotropicTV(),
)

u, stats = TotalVariationImageFiltering.solve(problem, TotalVariationImageFiltering.ROFConfig())
```

Group-sparse TV uses the low-memory ADMM/MM solver:

```julia
volume = rand(Float32, 96, 96, 64)
mode = TotalVariationImageFiltering.GroupSparseTV(3)
problem = TotalVariationImageFiltering.TVProblem(
    volume;
    lambda = 0.08f0,
    tv_mode = mode,
)
u, stats = TotalVariationImageFiltering.solve(
    problem,
    TotalVariationImageFiltering.GSTVConfig(),
)
```

## Benchmarking

Benchmark instructions and scripts are documented in
[benchmark/README.md](benchmark/README.md).

## Testing

```julia
import Pkg
Pkg.test()
```

CUDA tests run only when CUDA is installed and functional.

## License

MIT. See [LICENSE](LICENSE).
