"""Polytope-projection gradient-accuracy benchmark for the explicit (e) vs implicit (i) qpax backends.

Sweeps (precision, solver_tol, m, d, kappa, seed) and records, per solver,
the absolute and relative gradient errors against one reference:
  - g_hard: Jacobian of the hard projection (active-set based, fp64 exact)

References are computed once in fp64, then the fp64 and fp32 sweeps run.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

# jax.config.update("jax_default_matmul_precision", "highest")

import numpy as np
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _with_info import (  # noqa: E402
    solve_qp_primal_with_info_explicit as e_primal_with_info,
    solve_qp_primal_with_info_implicit as i_primal_with_info,
)
from quality_gradient_visualize import main as visualize_main  # noqa: E402

N = 20
M_TOTAL = 25
M_LIST = [10,12,15]#[N-m for m in range(M_TOTAL+1, M_TOTAL-min(N,3),-1)]
# N = 50
# M_TOTAL = 60
# M_LIST = [25,35,45]#[N-m for m in range(M_TOTAL+1, M_TOTAL-min(N,3),-1)]

D_LIST = np.logspace(-2, 2, 21)
N_SEEDS = 3

ACCURACY_KAPPAS = {
    "f64": [10.0**-i for i in range(2, 8)],
    # "f32": [1e-2, 3.16e-3, 1e-3, 3.16e-4, 1e-4],
    "f32": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    # "f32": [10.0**-i for i in range(2, 5)],
}
TOL_F64 = 1e-8
TOL_F32 = 1e-4

MAX_ITER = 100
HARD_SMALL = 1e-8

SOLVERS = {"e": e_primal_with_info, "i": i_primal_with_info}

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "quality_gradient_benchmark.pkl"


def sample_instance(n, m_total, m_active, d, seed):
    rng = np.random.default_rng(seed)
    y0 = rng.standard_normal(n)
    while True:
        GA = rng.standard_normal((m_active, n))
        if np.linalg.matrix_rank(GA) == m_active:
            break
    GA = GA / np.linalg.norm(GA, axis=1, keepdims=True)
    GI = rng.standard_normal((m_total - m_active, n))
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


def sample_u(seed):
    rng = np.random.default_rng(10_000 + seed)
    u = rng.standard_normal(N)
    return u / np.linalg.norm(u)


def compute_references():
    """Precompute g_hard for every (seed, m, d_idx) in fp64."""
    refs = {}
    total = N_SEEDS * len(M_LIST) * len(D_LIST)
    pbar = tqdm(total=total, desc="refs", smoothing=0)
    for seed in range(N_SEEDS):
        u_np = sample_u(seed)
        for m_active in M_LIST:
            for d_idx, d in enumerate(D_LIST):
                x_np, G_np, h_np, GA_np = sample_instance(N, M_TOTAL, m_active, d, seed)
                g_hard = exact_grad_hard(GA_np, u_np)

                x_j = jnp.asarray(x_np, dtype=jnp.float64)
                G_j = jnp.asarray(G_np, dtype=jnp.float64)
                h_j = jnp.asarray(h_np, dtype=jnp.float64)

                refs[(seed, m_active, d_idx)] = {
                    "x": x_np,
                    "G": G_np,
                    "h": h_np,
                    "u": u_np,
                    "g_hard": g_hard,
                }
                pbar.update(1)
        print(f"[refs] seed {seed} done")
    pbar.close()
    return refs


def errors(gx, g_hard):
    dh = np.linalg.norm(gx - g_hard)
    nh = np.linalg.norm(g_hard)
    eh_abs = float(dh)
    eh_rel = float(dh if nh < HARD_SMALL else dh / nh)
    return eh_abs, eh_rel


def make_grad_fn(primal_with_info, dtype):
    A_eq = jnp.zeros((0, N), dtype=dtype)
    b_eq = jnp.zeros((0,), dtype=dtype)
    Q = jnp.eye(N, dtype=dtype)

    def grad_fn(x, G, h, kappa, solver_tol, u):
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
    kappas = ACCURACY_KAPPAS[precision]
    solver_tol = TOL_F64 if x64 else TOL_F32

    # Keep one compiled gradient function per solver/precision.
    # `kappa` changes often, but its shape and dtype do not, so it can be
    # passed as a normal argument instead of creating a new jitted function
    # for every value in `kappas`.
    grad_fns = {
        name: make_grad_fn(pri, dtype)
        for name, pri in SOLVERS.items()
    }

    rows = []
    total = N_SEEDS * len(M_LIST) * len(D_LIST)
    pbar = tqdm(total=total, desc=f"sweep {precision}", smoothing=0)
    for seed in range(N_SEEDS):
        for m_active in M_LIST:
            for d_idx, d in enumerate(D_LIST):
                ref = refs[(seed, m_active, d_idx)]
                x_j = jnp.asarray(ref["x"], dtype=dtype)
                G_j = jnp.asarray(ref["G"], dtype=dtype)
                h_j = jnp.asarray(ref["h"], dtype=dtype)
                u_j = jnp.asarray(ref["u"], dtype=dtype)

                for kappa in kappas:
                    kappa_j = np_dtype(kappa)
                    solver_tol_j = np_dtype(min(kappa, solver_tol))
                    for solver_name, grad_fn in grad_fns.items():
                        gx, total_iters = grad_fn(
                            x_j, G_j, h_j, kappa_j, solver_tol_j, u_j
                        )
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
                                "solver_tol": float(solver_tol_j),
                                "solver": solver_name,
                                "m": m_active,
                                "d": float(d),
                                "kappa": float(kappa),
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
            pickle.dump({"rows": rows, "N": N, "M_TOTAL": M_TOTAL}, f)
    print(f"saved {DATA_PATH} ({len(rows)} rows)")
    visualize_main()


if __name__ == "__main__":
    main()
