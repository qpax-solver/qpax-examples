"""`solve_qp_primal_group_flags` shim over the unified qpax explicit backend.

The original `eqpax.diff_qp.solve_qp_primal_group_flags` returned the primal
solution plus seven int32 phase indices ``(scaling, predictor, centering,
corrector, linesearch, relax, backward)`` describing where a non-finite
intermediate first appeared. The new `qpax` keeps the underlying
``solve_qp_group_flags`` (in ``qpax.explicit.pdip``) and the
``return_bad_step`` flag on ``relax_qp`` (when ``backend="e"``), so we can
rebuild the wrapper here. The differentiable backward leg (``diff_qp``) no
longer exposes a ``return_bad_step`` knob, so we substitute a plain
finite-output check on its derivatives.
"""

import jax.numpy as jnp

from qpax import relax_qp
from qpax.explicit.diff_qp import diff_qp
from qpax.explicit.pdip import solve_qp_group_flags


def solve_qp_primal_group_flags(
    Q, q, A, b, G, h,
    solver_tol=1e-5,
    target_kappa=1e-3,
    max_iter=30,
    input_grad=None,
):
    """Return the explicit-backend primal plus per-operation first-fire phase indices.

    Each returned int32 is the time at which the operation first produced a
    non-finite intermediate; lower index = earlier. Forward sub-groups carry
    the 0-indexed PDIP iter; ``relax`` reports ``max_iter`` if relaxation
    misbehaves; ``backward`` reports ``max_iter + 1`` if the implicit-diff
    output is non-finite. Sentinel ``max_iter + 2`` means "never fired".
    """
    sentinel = jnp.int32(max_iter + 2)
    (
        x,
        s,
        z,
        y,
        _,
        _,
        scaling_iter,
        predictor_iter,
        centering_iter,
        corrector_iter,
        linesearch_iter,
    ) = solve_qp_group_flags(
        Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter
    )

    xr, sr, zr, yr, _, _, relax_bad = relax_qp(
        Q, q, A, b, G, h, x, s, z, y,
        backend="e",
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
        return_bad_step=True,
    )

    if input_grad is None:
        input_grad = jnp.ones_like(x)
    grads = diff_qp(Q, q, A, b, G, h, xr, sr, zr, yr, input_grad)
    backward_bad = jnp.logical_not(
        jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in grads]))
    )

    relax_iter = jnp.where(relax_bad, jnp.int32(max_iter), sentinel)
    backward_iter = jnp.where(backward_bad, jnp.int32(max_iter + 1), sentinel)

    return (
        x,
        scaling_iter,
        predictor_iter,
        centering_iter,
        corrector_iter,
        linesearch_iter,
        relax_iter,
        backward_iter,
    )
