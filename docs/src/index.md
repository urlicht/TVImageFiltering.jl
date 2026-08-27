# TotalVariationImageFiltering.jl

[![CI](https://github.com/urlicht/TotalVariationImageFiltering.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/urlicht/TotalVariationImageFiltering.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://urlicht.github.io/TotalVariationImageFiltering.jl/)

`TotalVariationImageFiltering.jl` is a Julia package for total-variation (TV) denoising and
reconstruction on `N`-dimensional arrays.

It currently provides:

- ROF denoising (`L2 + TV`) via a Chambolle-style dual projected-gradient algorithm.
- PDHG / Chambolle-Pock for `L2 + TV` and Poisson `KL + TV`.
- PDHG primal constraints: non-negativity and box constraints.
- Isotropic and anisotropic TV.
- Overlapping group-sparse anisotropic TV for 2D images and 3D volumes.
- Neumann and periodic boundaries.
- Single-image and batched solves.
- Automatic lambda selection for ROF (discrepancy principle and MC-SURE).
- Optional CUDA acceleration via package extension.

Core models:

```math
\min_u \frac{1}{2}\|u-f\|_2^2 + \lambda\,\mathrm{TV}(u)
```

```math
\min_u \sum_i \left(u_i - f_i \log(u_i)\right) + \lambda\,\mathrm{TV}(u)
```

## Documentation Guide

- [Installation](installation.md)
- [Quick Start](quick-start.md)
- [Problem & API](problem-and-api.md)
- [ROF Solver](rof-solver.md)
- [PDHG Solver](pdhg-solver.md)
- [Group-Sparse TV Solver](gstv-solver.md)
- [Lambda Selection](lambda-selection.md)
- [Batch & CUDA](batch-and-cuda.md)
- [Benchmarking](benchmarking.md)
- [API Reference](api-reference.md)
- [References](references.md)

## Notes

- This manual consolidates and expands content from the repository
  `README.md`, `benchmark/README.md`, source code, and docstrings.
- For the latest benchmark runs, see the benchmark scripts and generated CSVs
  in the repository.
