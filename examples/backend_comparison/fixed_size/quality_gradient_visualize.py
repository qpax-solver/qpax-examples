"""Plots for the polytope-projection benchmark (test variant).

Saves one gallery/polytope_projection.png figure with:
  4 rows x 2 columns
  Left column: fp64 results
  Right column: fp32 results
  Row 1: kappa vs hard error (median) for explicit (e) and implicit (i)
  Row 2: kappa vs median total forward iterations
  Row 3: kappa vs max total forward iterations
  Row 4: kappa vs NaN percentage
  Shared legend: solver via color, m via linestyle.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DATA_PATH = Path(__file__).resolve().parent / "data" / "quality_gradient_benchmark.pkl"
GALLERY = Path(__file__).resolve().parent / "gallery"
print(f"data path: {DATA_PATH}")
print(f"gallery path: {GALLERY}")

SOLVERS = ("e", "i")
SOLVER_COLORS = {"e": "C0", "i": "C1"}
LINESTYLES = ("-", "--", ":", "-.")
ERROR_KIND = "rel"

_FLOOR = 1e-300
_X_BUFFER_DECADES = 0.08


def plot_precision(ax_col, df, precision, kind):
    D = df[df.precision == precision]
    m_list = sorted(D["m"].unique())
    m_linestyles = {m: LINESTYLES[i % len(LINESTYLES)] for i, m in enumerate(m_list)}
    solver_tols = sorted(D["solver_tol"].unique())
    kappa_min = D["kappa"].min()
    kappa_max = D["kappa"].max()
    kappa_min_plot = kappa_min / (10**_X_BUFFER_DECADES)
    kappa_max_plot = kappa_max * (10**_X_BUFFER_DECADES)
    solver_tol_max = solver_tols[-1]

    hard_col = f"err_hard_{kind}"
    label = "relative" if kind == "rel" else "absolute"

    sub = D
    if len(solver_tols) == 1:
        ax_col[0].set_title(f"{precision} (solver_tol = {solver_tol_max:g})")
    else:
        ax_col[0].set_title(
            f"{precision} (solver_tol = min(kappa, {solver_tol_max:g}))"
        )

    for solver in SOLVERS:
        s = sub[sub.solver == solver]
        for m in m_list:
            T = s[s.m == m]
            if T.empty:
                continue
            ax = ax_col[0]
            T_finite = T[~T["nan"]]
            g = T_finite.groupby("kappa")[hard_col]
            med = g.median()
            ax.plot(
                med.index.values,
                np.clip(med.values, _FLOOR, None),
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=m_linestyles[m],
            )

            g_iters = T.groupby("kappa")["total_iters"]
            med_iters = g_iters.median()
            max_iters = g_iters.max()
            ax_col[1].plot(
                med_iters.index.values,
                med_iters.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=m_linestyles[m],
            )
            ax_col[2].plot(
                max_iters.index.values,
                max_iters.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=m_linestyles[m],
            )

            ax = ax_col[3]
            g_nan = T.groupby("kappa")["nan"].mean() * 100.0
            ax.plot(
                g_nan.index.values,
                g_nan.values,
                marker="o",
                color=SOLVER_COLORS[solver],
                linestyle=m_linestyles[m],
            )

    for row in range(4):
        ax = ax_col[row]
        ax.set_xscale("log")
        ax.set_xlim(kappa_max_plot, kappa_min_plot)
        ax.grid(True, alpha=0.25)
        if row < 1:
            ax.set_yscale("log")
        if row == 3:
            ax.set_ylim(-2, 102)
            ax.set_xlabel(r"$\kappa_\mathrm{target}$")

    solver_handles = [
        Line2D([0], [0], color=SOLVER_COLORS[s], label=s) for s in SOLVERS
    ]
    m_handles = [
        Line2D([0], [0], color="black", linestyle=m_linestyles[m], label=f"m={m}")
        for m in m_list
    ]
    return solver_handles + m_handles, label


def main():
    GALLERY.mkdir(exist_ok=True)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    N = data["N"]
    M_TOTAL = data["M_TOTAL"]
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

    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=len(legend_handles),
    )

    path = GALLERY / f"quality_gradient_benchmark_N{N}_M{M_TOTAL}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
