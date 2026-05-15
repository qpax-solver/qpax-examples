"""`solve_qp_primal_with_info` shim over the unified qpax API.

The original `iqpax.diff_qp.solve_qp_primal_with_info` /
`eqpax.diff_qp.solve_qp_primal_with_info` returned `(x, info)` where `info`
exposed the inner PDIP and relaxation iteration counts used by these
benchmarks for plotting. The new `qpax` does not expose a `_with_info`
variant, so we rebuild it here from the public API:

* `x` comes from `qpax.solve_qp_primal` (already differentiable).
* `info["solve_iters"]`, `info["relax_iters"]` and `info["total_iters"]`
  come from a parallel non-differentiable `solve_qp` + `relax_qp` pair.

Iter counts are wrapped in `jax.lax.stop_gradient` so they can never feed
back into the cotangent of `x`.
"""

from functools import partial

import jax
import jax.numpy as jnp

import qpax


def _solve_with_info(
    Q, q, A, b, G, h,
    *,
    backend,
    solver_tol=1e-5,
    target_kappa=1e-3,
    max_iter=30,
):
    x = qpax.solve_qp_primal(
        Q, q, A, b, G, h,
        backend=backend,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
    )

    Q_sg = jax.lax.stop_gradient(Q)
    q_sg = jax.lax.stop_gradient(q)
    A_sg = jax.lax.stop_gradient(A)
    b_sg = jax.lax.stop_gradient(b)
    G_sg = jax.lax.stop_gradient(G)
    h_sg = jax.lax.stop_gradient(h)

    x0, s0, z0, y0, _, solve_iters = qpax.solve_qp(
        Q_sg, q_sg, A_sg, b_sg, G_sg, h_sg,
        backend=backend,
        solver_tol=solver_tol,
        max_iter=max_iter,
    )
    _, _, _, _, _, relax_iters = qpax.relax_qp(
        Q_sg, q_sg, A_sg, b_sg, G_sg, h_sg, x0, s0, z0, y0,
        backend=backend,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
    )
    info = {
        "solve_iters": solve_iters,
        "relax_iters": relax_iters,
        "total_iters": solve_iters + relax_iters,
    }
    return x, info


solve_qp_primal_with_info_explicit = partial(_solve_with_info, backend="e")
solve_qp_primal_with_info_implicit = partial(_solve_with_info, backend="i")
