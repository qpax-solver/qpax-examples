"""
Bezier trajopt: inner QP over control points + outer L-BFGS on segment times T.

`run_trajopt(world, cfg, params)` returns a dict of metrics. No globals: all
configuration is passed in. Plotting is in visualize.py.
"""

from __future__ import annotations

import time
from math import comb
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from functools import partial

import qpax
import pydecomp as pdc


SOLVERS = {
    "e": partial(qpax.solve_qp_primal, backend="e"),
    "i": partial(qpax.solve_qp_primal, backend="i"),
}

# Data lives one directory up (shared with trajopt_lbfgs.py).
_DATA_DIR = Path(__file__).resolve().parent / "environments"


# ----------------------------- Bezier helpers ------------------------------ #


def bezier_diff(n: int) -> np.ndarray:
    d = np.zeros((n, n + 1))
    for i in range(n):
        d[i, i], d[i, i + 1] = -n, n
    return d


def deriv_op(n: int, r: int) -> np.ndarray:
    m = np.eye(n + 1)
    for k in range(r):
        m = bezier_diff(n - k) @ m
    return m


def bernstein_gram(n: int) -> np.ndarray:
    g = np.empty((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            g[i, j] = comb(n, i) * comb(n, j) / ((2 * n + 1) * comb(2 * n, i + j))
    return g


def bernstein_basis(n: int, t: float) -> np.ndarray:
    return np.array([comb(n, i) * (1 - t) ** (n - i) * t**i for i in range(n + 1)])


def bezier_eval(p: np.ndarray, ts: np.ndarray) -> np.ndarray:
    n = p.shape[0] - 1
    return np.array([bernstein_basis(n, t) @ p for t in ts])


def ctrl_from_primal(x, k, n_ctrl, d):
    s = (n_ctrl + 1) * d
    xv = np.asarray(x, dtype=float)
    return [xv[seg * s : (seg + 1) * s].reshape(n_ctrl + 1, d) for seg in range(k)]


def _bcast(v, d):
    return np.broadcast_to(np.asarray(v, dtype=float), (d,)).copy()


# --------------------------- JAX QP matrix assembly ------------------------ #


def build_jax_mats(
    *,
    n,
    d,
    k,
    a_cells,
    b_cells,
    p_start,
    p_end,
    continuity,
    weights,
    deriv_lower,
    deriv_upper,
    deriv_init,
    deriv_end,
    dtype,
):
    """Return build_qp(t_seg) -> (Q, q, A_eq, b_eq, G, h)."""
    s = (n + 1) * d
    nv = k * s

    h_kron, active = [], []
    for r, w in enumerate(weights, start=1):
        if w == 0:
            continue
        mr = deriv_op(n, r)
        h_kron.append(
            jnp.asarray(
                np.kron(mr.T @ bernstein_gram(n - r) @ mr, np.eye(d)), dtype=dtype
            )
        )
        active.append((r, float(w)))

    # Cell-membership inequalities (T-independent).
    g_rows, h_vals = [], []
    for seg in range(k):
        ak, bk = a_cells[seg], b_cells[seg].ravel()
        for i in range(n + 1):
            for j in range(ak.shape[0]):
                row = np.zeros(nv)
                for dim in range(d):
                    row[seg * s + i * d + dim] = ak[j, dim]
                g_rows.append(row)
                h_vals.append(bk[j])
    g_cell = jnp.asarray(np.array(g_rows), dtype=dtype)
    h_cell = jnp.asarray(np.array(h_vals), dtype=dtype)

    # Derivative bounds: G constant, h scales with T_seg^r.
    g_di_rows, di_seg, di_r, di_bnd = [], [], [], []

    def _add_bnd(r, vec, *, upper):
        mr = deriv_op(n, r)
        sign = 1.0 if upper else -1.0
        for seg in range(k):
            for i in range(mr.shape[0]):
                for dim in range(d):
                    row = np.zeros(nv)
                    for jj in range(n + 1):
                        row[seg * s + jj * d + dim] = sign * mr[i, jj]
                    g_di_rows.append(row)
                    di_seg.append(seg)
                    di_r.append(r)
                    di_bnd.append(sign * vec[dim])

    for r, hi in deriv_upper.items():
        if 1 <= r <= n:
            _add_bnd(r, _bcast(hi, d), upper=True)
    for r, lo in deriv_lower.items():
        if 1 <= r <= n:
            _add_bnd(r, _bcast(lo, d), upper=False)

    if g_di_rows:
        g_di = jnp.asarray(np.array(g_di_rows), dtype=dtype)
        di_seg_j = jnp.asarray(np.array(di_seg, np.int32))
        di_r_j = jnp.asarray(np.array(di_r, np.int32))
        di_bnd_j = jnp.asarray(np.array(di_bnd), dtype=dtype)
    else:
        g_di = jnp.zeros((0, nv), dtype=dtype)
        di_seg_j = jnp.zeros((0,), jnp.int32)
        di_r_j = jnp.zeros((0,), jnp.int32)
        di_bnd_j = jnp.zeros((0,), dtype=dtype)

    # Endpoint position equalities.
    a_ep = np.zeros((2 * d, nv))
    b_ep = np.zeros(2 * d)
    for dim in range(d):
        a_ep[dim, dim] = 1.0
        b_ep[dim] = p_start[dim]
        a_ep[d + dim, (k - 1) * s + n * d + dim] = 1.0
        b_ep[d + dim] = p_end[dim]
    a_ep_j, b_ep_j = jnp.asarray(a_ep, dtype=dtype), jnp.asarray(b_ep, dtype=dtype)

    # Initial / final derivative equalities (RHS scales with T^r).
    de_rows, de_seg, de_r, de_target = [], [], [], []

    def _add_eq(end_seg, mr_row, r, vec):
        v_v = _bcast(vec, d)
        for dim in range(d):
            row = np.zeros(nv)
            for jj in range(n + 1):
                row[end_seg * s + jj * d + dim] = mr_row[jj]
            de_rows.append(row)
            de_seg.append(end_seg)
            de_r.append(r)
            de_target.append(v_v[dim])

    for r, v in deriv_init.items():
        if 1 <= r <= n:
            _add_eq(0, deriv_op(n, r)[0], r, v)
    for r, v in deriv_end.items():
        if 1 <= r <= n:
            _add_eq(k - 1, deriv_op(n, r)[-1], r, v)

    if de_rows:
        de_rows_j = jnp.asarray(np.array(de_rows), dtype=dtype)
        de_seg_j = jnp.asarray(np.array(de_seg, np.int32))
        de_r_j = jnp.asarray(np.array(de_r, np.int32))
        de_target_j = jnp.asarray(np.array(de_target), dtype=dtype)
    else:
        de_rows_j = jnp.zeros((0, nv), dtype=dtype)
        de_seg_j = jnp.zeros((0,), jnp.int32)
        de_r_j = jnp.zeros((0,), jnp.int32)
        de_target_j = jnp.zeros((0,), dtype=dtype)

    # Junction continuity equalities up to order `continuity`.
    cont_l, cont_r_, cont_kl, cont_kr, cont_ord = [], [], [], [], []
    for seg in range(k - 1):
        for r in range(continuity + 1):
            mr = deriv_op(n, r)
            for dim in range(d):
                lrow, rrow = np.zeros(nv), np.zeros(nv)
                for i in range(n + 1):
                    lrow[seg * s + i * d + dim] = mr[-1, i]
                    rrow[(seg + 1) * s + i * d + dim] = mr[0, i]
                cont_l.append(lrow)
                cont_r_.append(rrow)
                cont_kl.append(seg)
                cont_kr.append(seg + 1)
                cont_ord.append(r)

    if cont_l:
        cont_left_j = jnp.asarray(np.array(cont_l), dtype=dtype)
        cont_right_j = jnp.asarray(np.array(cont_r_), dtype=dtype)
        cont_kl_j = jnp.asarray(np.array(cont_kl, np.int32))
        cont_kr_j = jnp.asarray(np.array(cont_kr, np.int32))
        cont_r_j = jnp.asarray(np.array(cont_ord, np.int32))
    else:
        cont_left_j = jnp.zeros((0, nv), dtype=dtype)
        cont_right_j = jnp.zeros((0, nv), dtype=dtype)
        cont_kl_j = jnp.zeros((0,), jnp.int32)
        cont_kr_j = jnp.zeros((0,), jnp.int32)
        cont_r_j = jnp.zeros((0,), jnp.int32)

    def build_qp(t_seg, *, regularize):
        ridge = (1e-2 if dtype == jnp.float32 else 1e-3) if regularize else 0.0
        qm = ridge * jnp.eye(nv, dtype=dtype)
        for seg in range(k):
            qk = sum(
                2.0 * w / t_seg[seg] ** (2 * r - 1) * h
                for h, (r, w) in zip(h_kron, active)
            )
            qm = qm.at[seg * s : (seg + 1) * s, seg * s : (seg + 1) * s].add(qk)

        eq_rows, eq_vals = [a_ep_j], [b_ep_j]
        if cont_left_j.shape[0] > 0:
            sl = 1.0 / t_seg[cont_kl_j] ** cont_r_j
            sr = 1.0 / t_seg[cont_kr_j] ** cont_r_j
            a_cont = sl[:, None] * cont_left_j - sr[:, None] * cont_right_j
            eq_rows.append(a_cont)
            eq_vals.append(jnp.zeros(a_cont.shape[0], dtype=dtype))
        if de_rows_j.shape[0] > 0:
            eq_rows.append(de_rows_j)
            eq_vals.append(de_target_j * (t_seg[de_seg_j] ** de_r_j))
        a_eq = jnp.vstack(eq_rows) if len(eq_rows) > 1 else eq_rows[0]
        b_eq = jnp.concatenate(eq_vals) if len(eq_vals) > 1 else eq_vals[0]

        if g_di.shape[0] == 0:
            g_ineq, h_ineq = g_cell, h_cell
        else:
            h_di = di_bnd_j * (t_seg[di_seg_j] ** di_r_j)
            g_ineq = jnp.vstack([g_cell, g_di])
            h_ineq = jnp.concatenate([h_cell, h_di])

        return qm, jnp.zeros(nv, dtype=dtype), a_eq, b_eq, g_ineq, h_ineq

    return build_qp, nv


# --------------------------- Loss + solvers -------------------------------- #


def make_loss(build_qp, solve_fn, cfg, *, t_dtype, time_penalty):
    feas_tol = max(1e-3, 1e3 * cfg["solver_tol"])

    def loss_with_aux(t_seg):
        qm, qv, a_eq, b_eq, gi, hi = build_qp(t_seg, regularize=cfg["regularize"])
        x = solve_fn(
            qm,
            qv,
            a_eq,
            b_eq,
            gi,
            hi,
            solver_tol=cfg["solver_tol"],
            target_kappa=cfg["target_kappa"],
            max_iter=cfg["max_iter"],
        )
        loss = 0.5 * jnp.dot(x, qm @ x) + time_penalty * jnp.sum(t_seg)
        eq_res = jnp.max(jnp.abs(a_eq @ x - b_eq))
        ineq_res = jnp.max(jnp.maximum(gi @ x - hi, 0.0))
        return loss, (x, jnp.maximum(eq_res, ineq_res))

    vg_jit = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))
    primal_jit = jax.jit(loss_with_aux)

    def vg(t_np):
        (v, (_, r)), g = vg_jit(jnp.asarray(t_np, dtype=t_dtype))
        if not np.isfinite(float(r)) or float(r) > feas_tol:
            return float("nan"), np.full(t_np.shape, np.nan), float(r)
        return float(v), np.asarray(g, dtype=float), float(r)

    def primal(t_np):
        loss, (x, _) = primal_jit(jnp.asarray(t_np, dtype=t_dtype))
        return float(loss) - time_penalty * float(np.sum(t_np)), np.asarray(x)

    return vg, primal


# --------------------- L-BFGS with Armijo backtracking --------------------- #


def lbfgs_optimize(
    vg_fn,
    t_init,
    label,
    *,
    lbfgs_iters,
    lbfgs_memory,
    ls_c1,
    ls_backtrack,
    ls_max_backtrack,
    t_seg_min,
    step_tol,
    verbose=True,
):
    t = np.asarray(t_init, dtype=float)
    val, grad, res = vg_fn(t)

    # Bootstrap if t_init makes the inner QP infeasible.
    for _ in range(20):
        if np.isfinite(val) and np.all(np.isfinite(grad)):
            break
        t = t * 1.5
        val, grad, res = vg_fn(t)

    opt = optax.scale_by_lbfgs(memory_size=lbfgs_memory)
    state = opt.init(jnp.asarray(t))
    history, costs, residuals = [t.copy()], [val], [res]
    if verbose:
        print(f"[{label}] iter  0 | cost = {val:.4f}  | T = {t.sum():.6f}")

    converged, ls_failed = False, False
    for it in range(1, lbfgs_iters + 1):
        if not np.all(np.isfinite(grad)):
            if verbose:
                print(f"[{label}] iter {it:2d} | grad not finite, stopping")
            break

        update, new_state = opt.update(jnp.asarray(grad), state, params=jnp.asarray(t))
        d = np.asarray(update, dtype=float)
        gd = float(np.dot(d, grad))
        if not np.isfinite(gd) or gd >= 0.0:
            d = -grad
            gd = -float(np.dot(grad, grad))

        alpha, accepted = 1.0, False
        for _ in range(ls_max_backtrack):
            t_new = np.clip(t + alpha * d, t_seg_min, None)
            v_new, g_new, r_new = vg_fn(t_new)
            if np.isfinite(v_new) and v_new <= val + ls_c1 * alpha * gd:
                accepted = True
                break
            alpha *= ls_backtrack
        if not accepted:
            ls_failed = True
            if verbose:
                print(f"[{label}] iter {it:2d} | line search failed, stopping")
            break

        step = float(np.linalg.norm(t_new - t))
        t, val, grad, state, res = t_new, v_new, g_new, new_state, r_new
        history.append(t.copy())
        costs.append(val)
        residuals.append(res)
        if verbose:
            print(
                f"[{label}] iter {it:2d} | cost = {val:.4f}  | T = {t.sum():.6f}  | a = {alpha:.2e}  | step = {step:.2e}"
            )
        if step < step_tol:
            converged = True
            break

    return dict(
        t_opt=t,
        history=history,
        costs=costs,
        residuals=residuals,
        converged=converged,
        ls_failed=ls_failed,
    )


# ----------------------------- Scenario ------------------------------------ #


def load_scenario(world: str):
    obstacles = np.load(_DATA_DIR / world / "ptcloud.npy")[:, :2]
    if world == "office":
        path = np.array(
            [[8, 1], [11, 3], [11, 12], [8.16, 15.45], [8.8, 16.96], [8.8, 20]]
        )
    elif world == "forest":
        path = np.array(
            [[-8, 10], [-6, 2], [-1, 1.5], [-1, -6], [2.52, -5], [4.25, -7.5], [8, -8]]
        )
    else:
        raise ValueError(world)
    a_cells, b_cells = pdc.convex_decomposition_2D(obstacles, path, np.array([[2, 2]]))
    return obstacles, path, a_cells, b_cells, path[0], path[-1]


# --------------------------- Top-level driver ------------------------------ #


def solver_cfg(
    *, solver, use_f64, solver_tol, target_kappa, regularize=False, max_iter=100
):
    if solver not in ("e", "i"):
        raise ValueError(f"unknown solver {solver}")
    return dict(
        solver=solver,
        use_f64=use_f64,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        regularize=regularize,
        max_iter=max_iter,
    )


def run_trajopt(world, cfg, params, *, label="run", verbose=True):
    """Run one trajopt with `cfg` (solver_cfg) + `params` (problem/L-BFGS).

    Returns dict with: t_opt, T_total, best_cost, ctrl, history, costs,
    residuals, n_iters, wall_time, converged, ls_failed, cfg, scenario.
    """
    obstacles, path, a_cells, b_cells, p_start, p_end = load_scenario(world)
    k, d = len(a_cells), int(p_start.size)
    n = params["n_bezier"]

    jax.config.update("jax_enable_x64", bool(cfg["use_f64"]))
    dtype = jnp.float64 if cfg["use_f64"] else jnp.float32

    build_qp, _ = build_jax_mats(
        n=n,
        d=d,
        k=k,
        a_cells=a_cells,
        b_cells=b_cells,
        p_start=p_start,
        p_end=p_end,
        continuity=params["continuity"],
        weights=params["weights"],
        deriv_lower=params["deriv_lower"],
        deriv_upper=params["deriv_upper"],
        deriv_init=params["deriv_init"],
        deriv_end=params["deriv_end"],
        dtype=dtype,
    )
    t_init = np.full(k, params["t_init"])

    solve_fn = SOLVERS[cfg["solver"]]

    vg, primal = make_loss(
        build_qp, solve_fn, cfg, t_dtype=dtype, time_penalty=params["time_penalty"]
    )

    t0 = time.time()
    out = lbfgs_optimize(
        vg,
        t_init,
        label,
        lbfgs_iters=params["lbfgs_iters"],
        lbfgs_memory=params["lbfgs_memory"],
        ls_c1=params["ls_c1"],
        ls_backtrack=params["ls_backtrack"],
        ls_max_backtrack=params["ls_max_backtrack"],
        t_seg_min=params["t_seg_min"],
        step_tol=params["step_tol"],
        verbose=verbose,
    )
    wall = time.time() - t0

    i_best = int(np.argmin(out["costs"]))
    t_best = out["history"][i_best]
    _, x = primal(t_best)
    ctrl = ctrl_from_primal(x, k, n, d)

    return dict(
        t_opt=t_best,
        T_total=float(np.sum(t_best)),
        best_cost=float(out["costs"][i_best]),
        ctrl=ctrl,
        history=out["history"],
        costs=out["costs"],
        residuals=out["residuals"],
        n_iters=len(out["history"]) - 1,
        wall_time=wall,
        converged=out["converged"],
        ls_failed=out["ls_failed"],
        cfg=cfg,
        scenario=dict(obstacles=obstacles, path=path, a_cells=a_cells, b_cells=b_cells),
        deriv_lower=params["deriv_lower"],
        deriv_upper=params["deriv_upper"],
    )
