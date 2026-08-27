# PDHG Solver

This page documents the PDHG / Chambolle-Pock solver in `TotalVariationImageFiltering.jl`.

## Variational Form

The solver handles:

```math
\min_u D(u,f) + \lambda\,\mathrm{TV}(u) + I_C(u),
```

with:

- `D = L2Fidelity`: `0.5 * ||u - f||_2^2`
- `D = PoissonFidelity`: `\sum_i (u_i - f_i\log u_i)` (up to constants)
- `C`: pointwise convex constraint set from:
  - `NoConstraint()`
  - `NonnegativeConstraint()`
  - `BoxConstraint(lower, upper)`

## Iteration Structure

The implementation uses the standard primal-dual pattern:

1. Dual ascent and projection onto TV dual ball of radius `lambda`.
2. Primal proximal step for data fidelity, followed by exact pointwise interval
   projection for `C`.
3. Over-relaxed update `u_bar = u + theta * (u - u_prev)`.

The primal prox operators are:

- L2:

```math
\operatorname{prox}_{\tau D}(v) = \frac{v + \tau f}{1+\tau}
```

- Poisson:

```math
\operatorname{prox}_{\tau D}(v)
= \frac{v - \tau + \sqrt{(v-\tau)^2 + 4\tau f}}{2},
```

followed by clamping to non-negative values.

## Step-Size Condition

PDHG requires:

```math
\tau \sigma \|\nabla\|^2 < 1.
```

The code enforces a conservative bound:

```math
\|\nabla\|^2 \le 4\sum_{d:\,n_d>1} h_d^{-2},
```

so it checks:

```math
\tau \sigma < \frac{1}{4\sum_{d:\,n_d>1} h_d^{-2}}.
```

Both `Neumann()` and `Periodic()` boundaries are supported. The same boundary
is used by the forward gradient and its negative-adjoint divergence.

## Stopping Criterion

Convergence is checked every `check_every` iterations with:

- relative primal change, and
- normalized primal-dual residual.

The solver stops when:

```math
\max(\text{relative\_primal\_change}, \text{primal\_dual\_residual}) \le \text{tol}.
```

## Poisson Data Requirements

For `PoissonFidelity`, `f` must be finite and non-negative. The solver validates
this before iterating.

## State Reuse

`PDHGState` stores primal/dual buffers and supports warm starts:

- reuse `state` across repeated solves to reduce allocations;
- dual state is not reset automatically.

## References

1. A. Chambolle and T. Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging," *Journal of Mathematical Imaging and Vision* 40:120-145, 2011. [DOI:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)
