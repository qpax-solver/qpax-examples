"""Generate loss-curve, trajectory and animation figures from a finished run."""

import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils import (
    DATA_DIR,
    ASSETS_DIR,
    load_config,
    make_env,
    latest_run_dir,
    combo_name,
    STATUS_SUCCESS,
)

import importlib.util as _ilu

_SF_UTILS = Path(__file__).resolve().parents[1] / "src"


def _load(modname, file):
    spec = _ilu.spec_from_file_location(modname, _SF_UTILS / file)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_anim = _load("_sf_anim", "animation.py")
fig_to_pil = _anim.fig_to_pil
save_gif = _anim.save_gif


def _agent_colors(n):
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if cycle:
        return [cycle[i % len(cycle)] for i in range(n)]
    cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_losses(run, trained_dir, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for sub in sorted(p for p in trained_dir.iterdir() if p.is_dir()):
        losses = np.load(sub / "losses.npy")
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if len(losses) == 0:
            continue
        label = f"{sub.name} [{meta['training_status_name']}]"
        ax.semilogy(losses, lw=1.5, label=label)
        plotted += 1
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training loss — run {run.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  losses ({plotted} curves) -> {out_path.name}")


def _draw_obstacles(ax, oc, or_):
    for i in range(len(oc)):
        ax.add_patch(Circle(oc[i], or_[i], color="0.5", alpha=0.55, zorder=3))
        ax.add_patch(
            Circle(oc[i], or_[i], fill=False, edgecolor="0.2", ls="--", lw=0.8, zorder=3)
        )


def _setup_workspace(ax, env, agent_colors):
    ax.set_xlim(env.ws_lo, env.ws_hi)
    ax.set_ylim(env.ws_lo, env.ws_hi)
    ax.set_aspect("equal")
    for a in range(env.n_agents):
        ax.plot(*env.starts[a], "o", color=agent_colors[a], markersize=8, zorder=6)
        ax.plot(*env.goals[a], "*", color=agent_colors[a], markersize=12, zorder=6)
    ax.grid(True, alpha=0.25)


def plot_trajectories(env, sub, name, out_path):
    rollouts = np.load(sub / "rollouts.npz", allow_pickle=True)
    positions = rollouts["positions"]
    test = np.load(sub.parent.parent / "test.npz", allow_pickle=True)
    oc_all, or_all = test["cfg_obs_centers"], test["cfg_obs_radii"]

    n = min(len(positions), 6)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    colors = _agent_colors(env.n_agents)
    for k in range(n):
        ax = axes[k // cols, k % cols]
        _draw_obstacles(ax, oc_all[k], or_all[k])
        pos = positions[k]
        for a in range(env.n_agents):
            ax.plot(
                pos[:, a, 0], pos[:, a, 1], color=colors[a], lw=1.8, alpha=0.9, zorder=5
            )
        _setup_workspace(ax, env, colors)
        ax.set_title(f"config {k}")
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(f"BarrierNet[{name}] — test trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  trajectories[{name}] -> {out_path.name}")


def make_animation(env, sub, name, out_path, anim_idx=0, n_trail=8, arrow_scale=0.8):
    rollouts = np.load(sub / "rollouts.npz", allow_pickle=True)
    positions = rollouts["positions"][anim_idx]
    controls = rollouts["controls"][anim_idx]
    u_noms = rollouts["u_noms"][anim_idx]
    test = np.load(sub.parent.parent / "test.npz", allow_pickle=True)
    oc, or_ = test["cfg_obs_centers"][anim_idx], test["cfg_obs_radii"][anim_idx]

    colors = _agent_colors(env.n_agents)
    frames = []
    for t in range(len(controls)):
        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_obstacles(ax, oc, or_)
        t_start = max(0, t - n_trail)
        for a in range(env.n_agents):
            for s in range(t_start, t):
                fade = 0.2 + 0.8 * (s - t_start) / max(t - t_start, 1)
                ax.plot(
                    positions[s : s + 2, a, 0],
                    positions[s : s + 2, a, 1],
                    color=colors[a],
                    lw=2.0,
                    alpha=fade,
                    zorder=4,
                )
            px, py = positions[t, a]
            ax.plot(px, py, "o", color=colors[a], markersize=10, zorder=7)
            ax.annotate(
                "",
                xy=(
                    px + arrow_scale * u_noms[t, a, 0],
                    py + arrow_scale * u_noms[t, a, 1],
                ),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2, ls="--"),
            )
            ax.annotate(
                "",
                xy=(
                    px + arrow_scale * controls[t, a, 0],
                    py + arrow_scale * controls[t, a, 1],
                ),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=colors[a], lw=1.8),
            )
        _setup_workspace(ax, env, colors)
        ax.set_title(f"BarrierNet-{name} ({env.n_agents} agents) | step {t}")
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, str(out_path), fps=8)
    print(f"  animation[{name}] ({len(frames)} frames) -> {out_path.name}")


def main(run_dir=None, config_path=None):
    run = run_dir or latest_run_dir(DATA_DIR)
    if run is None:
        raise RuntimeError("No data run found.")
    cfg = load_config(path=config_path, run_dir=run)
    env = make_env(cfg["environment_parameters"])
    vp = cfg["visualize_parameters"]
    print(f"Visualizing run: {run}")

    trained_dir = run / "trained"
    out_dir = ASSETS_DIR / run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_losses(run, trained_dir, out_dir / "loss_comparison.png")

    by_solver = {c["solver"]: c for c in cfg["training_parameters"]["combinations"]}
    for solver in vp.get("figures", []):
        if solver not in by_solver:
            print(f"  [skip figure] {solver} not trained")
            continue
        sub = trained_dir / combo_name(by_solver[solver])
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if (
            meta["training_status"] != STATUS_SUCCESS
            or not (sub / "rollouts.npz").exists()
        ):
            print(f"  [skip figure] {solver}: status={meta['training_status_name']}")
            continue
        plot_trajectories(env, sub, solver, out_dir / f"trajectories_{solver}.png")

    for solver in vp.get("animations", []):
        if solver not in by_solver:
            print(f"  [skip anim] {solver} not trained")
            continue
        sub = trained_dir / combo_name(by_solver[solver])
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if (
            meta["training_status"] != STATUS_SUCCESS
            or not (sub / "rollouts.npz").exists()
        ):
            print(f"  [skip anim] {solver}: status={meta['training_status_name']}")
            continue
        make_animation(env, sub, solver, out_dir / f"anim_{solver}.gif")

    print(f"\nAssets saved under {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    main(run_dir=(DATA_DIR / args.run) if args.run else None, config_path=args.config)
