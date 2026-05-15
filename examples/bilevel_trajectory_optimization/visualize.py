"""Plotting for the trajopt accuracy sweep."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pydecomp as pdc

from lbfgs import bezier_eval, deriv_op


def _trajectory_speed(ctrl, t_seg, n_samples=200):
    ts = np.linspace(0, 1, n_samples)
    pts, sp = [], []
    for pk, tk in zip(ctrl, t_seg):
        pts.append(bezier_eval(pk, ts))
        dpk = deriv_op(pk.shape[0] - 1, 1) @ pk
        sp.append(np.linalg.norm(bezier_eval(dpk, ts), axis=1) / tk)
    return np.vstack(pts), np.concatenate(sp)


_DERIV_LABELS = ("", "vel", "acc", "jerk", "snap")


def _trajectory_derivatives_vs_tau(ctrl, t_seg, *, n_samples=160):
    """Cumulative segment time τ and physical derivatives d^r p/dτ^r (concatenated)."""
    n = ctrl[0].shape[0] - 1
    tau_chunks, der_chunks = [], {r: [] for r in range(1, n + 1)}
    tau0 = 0.0
    for seg_i, (pk, Tk) in enumerate(zip(ctrl, t_seg)):
        ts = np.linspace(0.0, 1.0, n_samples)
        if seg_i > 0:
            ts = ts[1:]
        tau_chunks.append(tau0 + ts * Tk)
        for r in range(1, n + 1):
            mr = deriv_op(n, r)
            d_u = bezier_eval(mr @ pk, ts)
            der_chunks[r].append(d_u / (Tk**r))
        tau0 += Tk
    tau = np.concatenate(tau_chunks)
    derivs = {r: np.vstack(der_chunks[r]) for r in range(1, n + 1)}
    return tau, derivs


# ----------------------------- Sweep plot ---------------------------------- #


def plot_sweep_kappa_vs_T(records, savepath=None):
    """kappa vs total trajectory time T, one color per (solver, precision).

    Each tolerance sweep uses the same line weight and alpha. Legend has one
    entry per (solver, precision).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    by_group: dict = {}
    for r in records:
        use_f64 = r.get("use_f64", False)
        by_group.setdefault((r["solver"], use_f64), []).append(r)

    cmap = plt.get_cmap("tab10")
    for i, (key, rs) in enumerate(sorted(by_group.items())):
        solver, use_f64 = key
        col = cmap(i % cmap.N)
        label = f"{solver} f{'64' if use_f64 else '32'}"

        by_tol: dict = {}
        for r in rs:
            by_tol.setdefault(r["solver_tol"], []).append(r)

        lw, alpha = 1.0, 0.35
        first = True
        for tol, tol_rs in sorted(by_tol.items()):
            lab = label if first else None
            first = False
            tol_rs = sorted(tol_rs, key=lambda x: x["target_kappa"])
            Ts = np.array([r["T_total"] for r in tol_rs])
            kappas = np.array([r["target_kappa"] for r in tol_rs])
            ax.plot(kappas, Ts, "o-", color=col, lw=lw, ms=4, alpha=alpha, label=lab)

    ax.set_xscale("log")
    ax.invert_xaxis()  # large κ on the left, small κ on the right
    ax.set_xlabel(r"target $\kappa$")
    ax.set_ylabel("total trajectory time T")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# --------------------------- Trajectory plot ------------------------------- #


def plot_trajectory(result, label, savepath=None):
    """Workspace trajectory (left, speed-colored) and d^r p/dτ^r vs τ (right)."""
    sc = result["scenario"]
    obstacles, a_cells, b_cells = sc["obstacles"], sc["a_cells"], sc["b_cells"]
    ctrl, t_opt, cfg = result["ctrl"], result["t_opt"], result["cfg"]
    deriv_lower = result.get("deriv_lower") or {}
    deriv_upper = result.get("deriv_upper") or {}

    n = ctrl[0].shape[0] - 1
    d = ctrl[0].shape[1]
    tau, derivs = _trajectory_derivatives_vs_tau(ctrl, t_opt)

    # pydecomp planar path uses pyny.plot2d, which always creates a new figure and
    # ignores a pre-made axes — so we must take its figure, then place derivative axes.
    ax_env = pdc.visualize_environment(
        Al=a_cells, bl=b_cells, planar=True, ax_view=False, ax=None
    )
    fig = ax_env.figure
    fig.set_size_inches(11.5, 2.2 + 1.35 * n)

    # pydecomp calls plot2d once per polyhedron, each time adding a new full-width
    # subplot(111); keep only the returned axes so layout/ colorbar stay sane.
    for ax in list(fig.axes):
        if ax is not ax_env:
            fig.delaxes(ax)

    margin_l, margin_r, margin_b, margin_t = 0.06, 0.02, 0.08, 0.05
    gap = 0.04
    left_w = 0.48
    plot_h = 1.0 - margin_b - margin_t
    right_x0 = margin_l + left_w + gap
    right_w = 1.0 - margin_r - right_x0
    row_h = plot_h / n
    ax_env.set_position([margin_l, margin_b, left_w, plot_h])

    ax_env.plot(obstacles[:, 0], obstacles[:, 1], ".", color="gray", markersize=0.1, alpha=0.5)
    for pk in ctrl:
        ax_env.plot(pk[:, 0], pk[:, 1], "ko", markersize=4, alpha=0.5)
    pts, speeds = _trajectory_speed(ctrl, t_opt)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", linewidth=2)
    lc.set_array(0.5 * (speeds[:-1] + speeds[1:]))
    ax_env.add_collection(lc)
    cax = inset_axes(ax_env, width="3%", height="36%", loc="lower right", borderpad=0.8)
    fig.colorbar(lc, cax=cax, label="speed")
    ax_env.set_xlim([np.min(obstacles[:, 0]), np.max(obstacles[:, 0])])
    ax_env.set_ylim([np.min(obstacles[:, 1]), np.max(obstacles[:, 1])])
    ax_env.set_title(f"{label} (tol={cfg['solver_tol']:.0e}, k={cfg['target_kappa']:.0e})")

    bnd_kw = dict(color="0.45", ls="--", lw=0.85, alpha=0.9, zorder=0)
    dim_names = tuple("xyz"[j] for j in range(d)) if d <= 3 else tuple(f"d{j}" for j in range(d))
    ax_right = None
    for idx, r in enumerate(range(1, n + 1)):
        y0 = margin_b + (n - 1 - idx) * row_h
        ax_r = fig.add_axes([right_x0, y0, right_w, row_h * 0.92], sharex=ax_right)
        ax_right = ax_r
        Y = derivs[r]
        for j in range(d):
            ax_r.plot(tau, Y[:, j], lw=1.05, label=dim_names[j], zorder=2)
        if r in deriv_upper:
            hi = np.asarray(deriv_upper[r], dtype=float).ravel()
            for j in range(min(d, hi.size)):
                ax_r.axhline(hi[j], **bnd_kw)
        if r in deriv_lower:
            lo = np.asarray(deriv_lower[r], dtype=float).ravel()
            for j in range(min(d, lo.size)):
                ax_r.axhline(lo[j], **bnd_kw)
        ylab = _DERIV_LABELS[r] if r < len(_DERIV_LABELS) else f"r={r}"
        ax_r.set_ylabel(ylab)
        ax_r.grid(True, alpha=0.3)
        if d > 1 and r == 1:
            ax_r.legend(loc="upper right", fontsize=7)
        if r < n:
            ax_r.tick_params(labelbottom=False)
    ax_r.set_xlabel(r"time $\tau$ [s]")

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
