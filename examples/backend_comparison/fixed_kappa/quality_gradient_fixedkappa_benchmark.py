"""Polytope-projection gradient-accuracy benchmark sweeping N at fixed kappa.

Mirrors ../quality_gradient_benchmark.py but fixes kappa and sweeps the number
of variables N. For each N, the number of active constraints M is set from a
ratio M_RATIO so that M = ceil(N * M_RATIO).

Loop structure:
    loop N
        loop M_RATIO
            loop solvers
"""

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _with_info import (  # noqa: E402
    solve_qp_primal_with_info_explicit as e_primal_with_info,
    solve_qp_primal_with_info_implicit as i_primal_with_info,
)
from quality_gradient_fixedkappa_visualize import main as visualize_main  # noqa: E402

N_LIST = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
M_RATIO_LIST = [0.2, 0.6, 0.8]
M_TOTAL_RATIO = 1.25
KAPPA = 1e-4

D_LIST = np.logspace(-2, 2, 21)
N_SEEDS = 3

TOL_F64 = 1e-8
TOL_F32 = 1e-4

MAX_ITER = 100
HARD_SMALL = 1e-8

SOLVERS = {"e": e_primal_with_info, "i": i_primal_with_info}

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "quality_gradient_fixedkappa_benchmark.pkl"


def m_active_for(n, ratio):
    return int(math.ceil(n * ratio))


def m_total_for(n):
    return int(math.ceil(n * M_TOTAL_RATIO))


def sample_instance(n, m_total, m_active, d, seed):
    rng = np.random.default_rng(seed)
    y0 = rng.standard_normal(n)
    while True:
        GA = rng.standard_normal((m_active, n))
        if np.linalg.matrix_rank(GA) == m_active:
            break
    GA = GA / np.linalg.norm(GA, axis=1, keepdims=True)
    GI = rng.standard_normal((m_total - m_active, n))
    if m_total - m_active > 0:
        GI = GI / np.linalg.norm(GI, axis=1, keepdims=True)
    hA = GA @ y0
    hI = GI @ y0 + rng.uniform(0.1, 1.0, m_total - m_active)
    G = np.vstack([GA, GI])
    h = np.concatenate([hA, hI])
    lam = d * rng.uniform(0.5, 1.5, m_active)
    x = y0 + GA.T @ lam
    return x, G, h, GA


def exact_grad_hard(GA, u):
    J = np.eye(GA.shape[1]) - GA.T @ np.linalg.solve(GA @ GA.T, GA)
    return J @ u


def sample_u(n, seed):
    rng = np.random.default_rng(10_000 + seed)
    u = rng.standard_normal(n)
    return u / np.linalg.norm(u)


def compute_references():
    """Precompute g_hard for every (n, m_ratio, seed, d_idx) in fp64."""
    refs = {}
    total = len(N_LIST) * len(M_RATIO_LIST) * N_SEEDS * len(D_LIST)
    pbar = tqdm(total=total, desc="refs", smoothing=0)
    for n in N_LIST:
        m_total = m_total_for(n)
        for m_ratio in M_RATIO_LIST:
            m_active = min(m_active_for(n, m_ratio), m_total)
            for seed in range(N_SEEDS):
                u_np = sample_u(n, seed)
                for d_idx, d in enumerate(D_LIST):
                    x_np, G_np, h_np, GA_np = sample_instance(
                        n, m_total, m_active, d, seed
                    )
                    g_hard = exact_grad_hard(GA_np, u_np)
                    refs[(n, m_ratio, seed, d_idx)] = {
                        "x": x_np,
                        "G": G_np,
                        "h": h_np,
                        "u": u_np,
                        "g_hard": g_hard,
                        "m_active": m_active,
                    }
                    pbar.update(1)
        print(f"[refs] n {n} done")
    pbar.close()
    return refs


def errors(gx, g_hard):
    dh = np.linalg.norm(gx - g_hard)
    nh = np.linalg.norm(g_hard)
    eh_abs = float(dh)
    eh_rel = float(dh if nh < HARD_SMALL else dh / nh)
    return eh_abs, eh_rel


def make_grad_fn(primal_with_info, n, dtype, solver_tol):
    A_eq = jnp.zeros((0, n), dtype=dtype)
    b_eq = jnp.zeros((0,), dtype=dtype)
    Q = jnp.eye(n, dtype=dtype)

    def grad_fn(x, G, h, kappa, u):
        def scalar_with_info(xv):
            y, info = primal_with_info(
                Q,
                -xv,
                A_eq,
                b_eq,
                G,
                h,
                solver_tol=solver_tol,
                target_kappa=kappa,
                max_iter=MAX_ITER,
            )
            return jnp.dot(u, y), info["total_iters"]

        return jax.grad(scalar_with_info, has_aux=True)(x)

    return jax.jit(grad_fn)


def run_precision(precision, refs):
    x64 = precision == "f64"
    jax.config.update("jax_enable_x64", x64)
    dtype = jnp.float64 if x64 else jnp.float32
    np_dtype = np.float64 if x64 else np.float32
    solver_tol = TOL_F64 if x64 else TOL_F32
    kappa_j = np_dtype(KAPPA)

    rows = []
    total = len(N_LIST) * len(M_RATIO_LIST) * N_SEEDS * len(D_LIST)
    pbar = tqdm(total=total, desc=f"sweep {precision}", smoothing=0)
    for n in N_LIST:
        # Recompile per N because shapes change.
        grad_fns = {
            name: make_grad_fn(pri, n, dtype, np_dtype(solver_tol))
            for name, pri in SOLVERS.items()
        }
        m_total = m_total_for(n)
        for m_ratio in M_RATIO_LIST:
            m_active = min(m_active_for(n, m_ratio), m_total)
            for seed in range(N_SEEDS):
                for d_idx, d in enumerate(D_LIST):
                    ref = refs[(n, m_ratio, seed, d_idx)]
                    x_j = jnp.asarray(ref["x"], dtype=dtype)
                    G_j = jnp.asarray(ref["G"], dtype=dtype)
                    h_j = jnp.asarray(ref["h"], dtype=dtype)
                    u_j = jnp.asarray(ref["u"], dtype=dtype)

                    for solver_name, grad_fn in grad_fns.items():
                        gx, total_iters = grad_fn(x_j, G_j, h_j, kappa_j, u_j)
                        total_iters = int(np.asarray(total_iters))
                        gx_np = np.asarray(gx, dtype=np.float64)
                        is_nan = not np.all(np.isfinite(gx_np))
                        if is_nan:
                            eh_abs = eh_rel = np.nan
                        else:
                            eh_abs, eh_rel = errors(gx_np, ref["g_hard"])
                        rows.append(
                            {
                                "precision": precision,
                                "solver_tol": float(solver_tol),
                                "solver": solver_name,
                                "n": n,
                                "m_ratio": float(m_ratio),
                                "m": m_active,
                                "m_total": m_total,
                                "d": float(d),
                                "kappa": float(KAPPA),
                                "seed": seed,
                                "total_iters": total_iters,
                                "err_hard_abs": eh_abs,
                                "err_hard_rel": eh_rel,
                                "nan": bool(is_nan),
                            }
                        )
                    pbar.update(1)
    pbar.close()
    return rows


def main():
    DATA_DIR.mkdir(exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    refs = compute_references()

    rows = []
    for precision in ("f64", "f32"):
        rows.extend(run_precision(precision, refs))
        with open(DATA_PATH, "wb") as f:
            pickle.dump(
                {
                    "rows": rows,
                    "N_LIST": N_LIST,
                    "M_RATIO_LIST": M_RATIO_LIST,
                    "M_TOTAL_RATIO": M_TOTAL_RATIO,
                    "kappa": KAPPA,
                },
                f,
            )
    print(f"saved {DATA_PATH} ({len(rows)} rows)")
    visualize_main()


if __name__ == "__main__":
    main()
