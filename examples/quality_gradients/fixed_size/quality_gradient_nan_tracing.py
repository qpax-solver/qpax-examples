"""Explicit-backend NaN attribution per operation for the quality_gradient benchmark.

For each (m, d, kappa, seed) case at fp32, runs `solve_qp_primal_group_flags`
(local shim over qpax's explicit backend) to find which sub-operation FIRST
produced a non-finite intermediate, and attributes the case to that operation.
Saves a stacked-bar plot.

Notes:
  * `solve_qp_group_flags` mirrors production `solve_qp`'s `sqrt(eps)` floor
    on s, z (qpax/explicit/pdip.py), so attribution here matches what the
    JIT'd benchmark in `quality_gradient_benchmark.py` actually computes.
  * Only fp32 is run; fp64 NaN counts in this benchmark are negligible.
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality_gradient_benchmark import (  # noqa: E402
    ACCURACY_KAPPAS,
    D_LIST,
    M_LIST,
    M_TOTAL,
    MAX_ITER,
    N,
    N_SEEDS,
    TOL_F32,
    sample_instance,
)
from _group_flags import solve_qp_primal_group_flags  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "quality_gradient_nan_tracing.pkl"
GALLERY = Path(__file__).resolve().parent / "gallery"

# Order matters: forward sub-groups in PDIP iteration order, then relax,
# then backward, then the catch-all bucket for cases the JIT'd grad NaN'd
# but the diagnostic trace did not flag.
GROUP_NAMES = (
    "scaling", "predictor", "centering", "corrector", "linesearch",
    "relax", "backward",
)
OPS = (*GROUP_NAMES, "no_attribution")
COLORS = {
    "scaling":        "C0",
    "predictor":      "C1",
    "centering":      "C2",
    "corrector":      "C3",
    "linesearch":     "C4",
    "relax":          "C5",
    "backward":       "C6",
    "no_attribution": "0.5",
}
LABELS = {
    "scaling":        "scaling (z / s)",
    "predictor":      "predictor (ds_a, dz_a)",
    "centering":      "centering (sigma, mu)",
    "corrector":      "corrector (dx, ds, dz, dy)",
    "linesearch":     "linesearch (alpha)",
    "relax":          "relaxation",
    "backward":       "backward (dlam / lam)",
    "no_attribution": "unattributed NaN",
}


def make_runner(precision):
    x64 = precision == "f64"
    jax.config.update("jax_enable_x64", x64)
    dtype = jnp.float64 if x64 else jnp.float32
    np_dtype = np.float64 if x64 else np.float32
    A_eq = jnp.zeros((0, N), dtype=dtype)
    b_eq = jnp.zeros((0,), dtype=dtype)
    Q = jnp.eye(N, dtype=dtype)
    solver_tol = TOL_F32 if not x64 else 1e-8

    def runner(x, G, h, kappa):
        return solve_qp_primal_group_flags(
            Q, -x, A_eq, b_eq, G, h,
            solver_tol=np_dtype(solver_tol),
            target_kappa=kappa,
            max_iter=MAX_ITER,
            input_grad=jnp.ones(N, dtype=dtype),
        )

    return jax.jit(runner), dtype, np_dtype


def attribute(flags, max_iter):
    """Earliest-firing group, or None if all sentinels.

    `flags` is the 7-tuple (scaling, predictor, centering, corrector,
    linesearch, relax, backward). Forward sub-groups carry the 0-indexed PDIP
    iter at first NaN; relax = max_iter; backward = max_iter + 1; sentinel
    = max_iter + 2 means "never fired". argmin returns the earliest-in-time
    operation.
    """
    sentinel = max_iter + 2
    arr = np.asarray(flags, dtype=np.int64)
    if np.all(arr >= sentinel):
        return None
    return GROUP_NAMES[int(np.argmin(arr))]


def collect(precision):
    runner, dtype, np_dtype = make_runner(precision)
    kappas = ACCURACY_KAPPAS[precision]
    counts = {m: {k: {op: 0 for op in OPS} for k in kappas} for m in M_LIST}
    totals = {m: {k: 0 for k in kappas} for m in M_LIST}
    rows = []

    for seed in range(N_SEEDS):
        for m in M_LIST:
            for d in D_LIST:
                x_np, G_np, h_np, _ = sample_instance(N, M_TOTAL, m, d, seed)
                x_j = jnp.asarray(x_np, dtype=dtype)
                G_j = jnp.asarray(G_np, dtype=dtype)
                h_j = jnp.asarray(h_np, dtype=dtype)
                for kappa in kappas:
                    totals[m][kappa] += 1
                    out = runner(x_j, G_j, h_j, np_dtype(kappa))
                    x_out = np.asarray(out[0])
                    flags = tuple(int(np.asarray(f)) for f in out[1:8])
                    op = attribute(flags, MAX_ITER)
                    primal_nonfinite = bool(not np.all(np.isfinite(x_out)))
                    if op is not None:
                        counts[m][kappa][op] += 1
                    elif primal_nonfinite:
                        counts[m][kappa]["no_attribution"] += 1
                    rows.append(
                        {
                            "seed": seed,
                            "m": m,
                            "d": float(d),
                            "kappa": float(kappa),
                            "flags": dict(zip(GROUP_NAMES, flags)),
                            "attribution": op,
                            "primal_nonfinite": primal_nonfinite,
                        }
                    )
        print(f"[{precision}] seed {seed} done")
    return counts, totals, rows


def summarize(precision, counts, totals):
    kappas = ACCURACY_KAPPAS[precision]
    print(f"\n=== {precision} attribution table (rows: m, cols: kappa) ===")
    header = "  m  kappa     total  " + "  ".join(f"{LABELS[op][:11]:>11}" for op in OPS)
    print(header)
    for m in M_LIST:
        for k in kappas:
            tot = totals[m][k]
            cells = "  ".join(f"{counts[m][k][op]:>11d}" for op in OPS)
            print(f"  {m}  {k:>7.0e}  {tot:>5d}  {cells}")


def plot(precision, counts, totals):
    kappas = ACCURACY_KAPPAS[precision]
    fig, axes = plt.subplots(
        len(M_LIST), 1,
        figsize=(7.0, 2.6 * len(M_LIST)),
        constrained_layout=True, squeeze=False, sharey=True, sharex=True,
    )
    pos = np.arange(len(kappas))
    ymax = 0.0
    for r, m in enumerate(M_LIST):
        ax = axes[r, 0]
        bottom = np.zeros(len(kappas))
        for op in OPS:
            vals = np.array([
                100.0 * counts[m][k][op] / max(1, totals[m][k])
                for k in kappas
            ])
            ax.bar(pos, vals, bottom=bottom, width=0.8,
                   color=COLORS[op], edgecolor="white", linewidth=0.6)
            bottom += vals
        ymax = max(ymax, float(bottom.max()))
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{k:.0e}" for k in kappas])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel(f"m = {m}\nNaN (%)")
        if r == len(M_LIST) - 1:
            ax.set_xlabel(r"$\kappa_\mathrm{target}$")
    axes[0, 0].set_ylim(0.0, max(ymax * 1.05, 1.0))
    handles = [mpatches.Patch(facecolor=COLORS[op], label=LABELS[op]) for op in OPS]
    fig.legend(handles=handles, loc="outside upper center",
               ncol=4, fontsize=9)
    fig.suptitle(
        f"explicit-backend {precision} NaN attribution  (N={N}, M_TOTAL={M_TOTAL}, "
        f"max_iter={MAX_ITER}, tol={TOL_F32:g})",
        y=1.02,
    )
    out = GALLERY / f"quality_gradient_nan_operations_{precision}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_payload(payload):
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {DATA_PATH}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    GALLERY.mkdir(exist_ok=True)
    payload = {
        "ops": list(OPS),
        "group_names": list(GROUP_NAMES),
        "colors": COLORS,
        "labels": LABELS,
        "N": N,
        "M_TOTAL": M_TOTAL,
        "M_LIST": list(M_LIST),
        "D_LIST": [float(d) for d in D_LIST],
        "N_SEEDS": N_SEEDS,
        "MAX_ITER": MAX_ITER,
        "TOL_F32": TOL_F32,
        "kappas": {precision: [float(k) for k in ACCURACY_KAPPAS[precision]]
                   for precision in ("f32",)},
        "results": {},
    }
    for precision in ("f32",):
        counts, totals, rows = collect(precision)
        summarize(precision, counts, totals)
        payload["results"][precision] = {
            "counts": counts,
            "totals": totals,
            "rows": rows,
        }
    save_payload(payload)
    for precision in ("f32",):
        counts = payload["results"][precision]["counts"]
        totals = payload["results"][precision]["totals"]
        plot(precision, counts, totals)


if __name__ == "__main__":
    main()
