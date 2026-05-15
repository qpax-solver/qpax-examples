"""Plots for the fixed-kappa polytope-projection benchmark.

Mirrors ../quality_gradient_visualize.py but the x axis is N (number of
variables) instead of kappa, and curves are coded by m_ratio instead of m.

Saves one gallery figure with:
  4 rows x 2 columns
  Left column: fp64 results
  Right column: fp32 results
  Row 1: N vs hard error (median) for explicit (e) and implicit (i)
  Row 2: N vs median total forward iterations
  Row 3: N vs max total forward iterations
  Row 4: N vs NaN percentage
  Shared legend: solver via color, m_ratio via linestyle.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "quality_gradient_fixedkappa_benchmark.pkl"
)
GALLERY = Path(__file__).resolve().parent / "gallery"
print(f"data path: {DATA_PATH}")
print(f"gallery path: {GALLERY}")

SOLVERS = ("e", "i")
SOLVER_COLORS = {"e": "C0", "i": "C1"}
LINESTYLES = ("-", "--", ":", "-.")
ERROR_KIND = "rel"

_FLOOR = 1e-300
_X_BUFFER = 0.5


def plot_precision(ax_col, df, precision, kind):
    D = df[df.precision == precision]
    ratio_list = sorted(D["m_ratio"].unique())
    ratio_linestyles = {
        r: LINESTYLES[i % len(LINESTYLES)] for i, r in enumerate(ratio_list)
    }
    solver_tols = sorted(D["solver_tol"].unique())
    n_min = D["n"].min()
    n_max = D["n"].max()
    n_min_plot = n_min - _X_BUFFER
    n_max_plot = n_max + _X_BUFFER
    if len(solver_tols) != 1:
        raise ValueError(
            f"Expected exactly one solver_tol for {precision}, got {solver_tols}"
        )
    solver_tol = solver_tols[0]

    hard_col = f"err_hard_{kind}"
    label = "relative" if kind == "rel" else "absolute"

    sub = D[D.solver_tol == solver_tol]
    ax_col[0].set_title(f"{precision} (solver_tol = {solver_tol:g})")

    for solver in SOLVERS:
        s = sub[sub.solver == solver]
        for r in ratio_list:
            T = s[s.m_ratio == r]
            if T.empty:
                continue
            ax = ax_col[0]
            T_finite = T[~T["nan"]]
            g = T_finite.groupby("n")[hard_col]
            med = g.median()
            ax.plot(
                med.index.values,
                np.clip(med.values, _FLOOR, None),
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=ratio_linestyles[r],
            )

            g_iters = T.groupby("n")["total_iters"]
            med_iters = g_iters.median()
            max_iters = g_iters.max()
            ax_col[1].plot(
                med_iters.index.values,
                med_iters.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=ratio_linestyles[r],
            )
            ax_col[2].plot(
                max_iters.index.values,
                max_iters.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=ratio_linestyles[r],
            )

            ax = ax_col[3]
            g_nan = T.groupby("n")["nan"].mean() * 100.0
            ax.plot(
                g_nan.index.values,
                g_nan.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=ratio_linestyles[r],
            )

    for row in range(4):
        ax = ax_col[row]
        ax.set_xlim(n_min_plot, n_max_plot)
        ax.grid(True, alpha=0.25)
        if row < 1:
            ax.set_yscale("log")
        if row == 3:
            ax.set_ylim(-2, 102)
            ax.set_xlabel(r"$N$")

    solver_handles = [
        Line2D([0], [0], color=SOLVER_COLORS[s], label=s) for s in SOLVERS
    ]
    ratio_handles = [
        Line2D(
            [0], [0], color="black", linestyle=ratio_linestyles[r], label=f"m_ratio={r}"
        )
        for r in ratio_list
    ]
    return solver_handles + ratio_handles, label


def main():
    GALLERY.mkdir(exist_ok=True)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    kappa = data["kappa"]
    df = pd.DataFrame(data["rows"])
    precisions = ["f64", "f32"]
    fig, axes = plt.subplots(
        4,
        len(precisions),
        figsize=(10, 12),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
        sharex="col",
    )

    legend_handles = None
    label = "relative" if ERROR_KIND == "rel" else "absolute"
    for col, precision in enumerate(precisions):
        legend_handles, label = plot_precision(
            axes[:, col], df, precision, kind=ERROR_KIND
        )

    axes[0, 0].set_ylabel(f"hard {label} error (median)")
    axes[1, 0].set_ylabel("total iters (median)")
    axes[2, 0].set_ylabel("total iters (max)")
    axes[3, 0].set_ylabel("NaN (%)")

    fig.suptitle(rf"$\kappa_\mathrm{{target}} = {kappa:g}$")
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=len(legend_handles),
    )

    path = GALLERY / f"quality_gradient_fixedkappa_benchmark_kappa{kappa:g}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
