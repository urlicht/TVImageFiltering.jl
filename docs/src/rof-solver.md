# ROF Solver

This page documents the ROF (`L2 + TV`) solver implemented in `TotalVariationImageFiltering.jl`.

## Variational Model

Given observed image/volume `f`, the solver targets:

```math
\min_u \; \frac{1}{2}\|u-f\|_2^2 + \lambda\,\mathrm{TV}(u),
```

where `lambda >= 0` controls regularization strength.

- `IsotropicTV`: `TV(u) = \sum_i \sqrt{\sum_d (\nabla_d u_i)^2}`
- `AnisotropicTV`: `TV(u) = \sum_i \sum_d |\nabla_d u_i|`

In this package, ROF is implemented for `L2Fidelity` only, and only for
unconstrained problems (`constraint = NoConstraint()`).

For non-negativity or box constraints, use `PDHGConfig`.

## Dual Projected-Gradient Iteration (Chambolle-Style)

The implementation uses a dual projected-gradient scheme for the ROF dual
formulation. For dual field `p`, each iteration computes:

```math
g^k = \operatorname{div}(p^k) - \frac{f}{\lambda},
```

```math
p^{k+1} = \Pi_{\mathcal{B}}\!\left(p^k + \tau \nabla g^k\right),
```

```math
u^{k+1} = f - \lambda\,\operatorname{div}(p^{k+1}).
```

Here `\Pi_{\mathcal{B}}` is projection onto the TV dual unit ball:

- isotropic mode: `\|p_i\|_2 \le 1` per pixel/voxel `i`
- anisotropic mode: `|p_{d,i}| \le 1` per component

## Discretization Used in This Package

- `gradient!`: forward differences with either homogeneous `Neumann()` or wrapped
  `Periodic()` boundary handling.
- `divergence!`: backward-difference adjoint consistent with `gradient!`.
- `spacing`: all differential operators are scaled by `1 / spacing[d]`.

## Step-Size Condition

For grid spacing `h_d = spacing[d]` and shape `n_d = size(f, d)`, the code enforces:

```math
\tau < \frac{1}{2\sum_{d:\,n_d>1} h_d^{-2}}.
```

Dimensions with size `1` do not contribute to the bound.

The solver checks this at runtime and throws an error if violated.

## Stopping Rule

Convergence is checked every `check_every` iterations using:

```math
\mathrm{rel\_change}
= \frac{\|u^k-u^{k-1}\|_2}{\max(\|u^{k-1}\|_2,\varepsilon)}.
```

The solver stops when `rel_change <= tol`, or at `maxiter`.

## State Reuse Behavior

`ROFState` stores both scratch arrays and dual variable state:

- Reuse `state` to reduce allocations.
- Dual variable `p` is intentionally carried across calls as warm start.
- If shape or element type changes, construct a new `ROFState`.

## Configuration Summary

`ROFConfig` fields:

- `maxiter`: maximum iterations.
- `tau`: dual ascent step.
- `tol`: tolerance on relative primal change.
- `check_every`: convergence check frequency.

## References

1. L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise removal algorithms," *Physica D* 60(1-4):259-268, 1992. [DOI:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)
2. A. Chambolle, "An algorithm for total variation minimization and applications," *Journal of Mathematical Imaging and Vision* 20:89-97, 2004. [DOI:10.1023/B:JMIV.0000011325.36760.1E](https://doi.org/10.1023/B:JMIV.0000011325.36760.1E)
