"""Load saved NaN tracing data and plot attribution trajectories.

This script is intentionally standalone: it reads the serialized output from
`quality_gradient_nan_tracing.py` and recreates the figure without importing
benchmark or tracing helpers.
"""

import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "quality_gradient_nan_tracing.pkl"
GALLERY = ROOT / "gallery"


def load_payload():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_precision(payload, precision):
    result = payload["results"][precision]
    counts = result["counts"]
    totals = result["totals"]
    m_list = payload["M_LIST"]
    kappas = payload["kappas"][precision]
    ops = payload["ops"]
    colors = payload["colors"]
    labels = payload["labels"]

    fig, axes = plt.subplots(
        len(m_list),
        1,
        figsize=(7.0, 2.6 * len(m_list)),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
        sharex=True,
    )
    pos = np.arange(len(kappas))
    ymax = 0.0

    for row_idx, m in enumerate(m_list):
        ax = axes[row_idx, 0]
        bottom = np.zeros(len(kappas))
        for op in ops:
            vals = np.array(
                [
                    100.0 * counts[m][k][op] / max(1, totals[m][k])
                    for k in kappas
                ]
            )
            ax.bar(
                pos,
                vals,
                bottom=bottom,
                width=0.8,
                color=colors[op],
                edgecolor="white",
                linewidth=0.6,
            )
            bottom += vals
        ymax = max(ymax, float(bottom.max()))
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{k:.0e}" for k in kappas])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel(f"m = {m}\nNaN (%)")
        if row_idx == len(m_list) - 1:
            ax.set_xlabel(r"$\kappa_\mathrm{target}$")

    axes[0, 0].set_ylim(0.0, max(ymax * 1.05, 1.0))
    handles = [mpatches.Patch(facecolor=colors[op], label=labels[op]) for op in ops]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=9)
    fig.suptitle(
        f"explicit-backend {precision} NaN attribution  (N={payload['N']}, "
        f"M_TOTAL={payload['M_TOTAL']}, max_iter={payload['MAX_ITER']}, "
        f"tol={payload['TOL_F32']:g})",
        y=1.02,
    )
    GALLERY.mkdir(exist_ok=True)
    out = GALLERY / f"quality_gradient_nan_operations_{precision}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    payload = load_payload()
    for precision in payload["results"]:
        plot_precision(payload, precision)


if __name__ == "__main__":
    main()
