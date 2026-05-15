"""Replot nominal pipeline outputs: latest non-OOM data run -> figures under assets/nominal_<ts>/."""

from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
STATUS_SUCCESS = 0
DEFAULT_MAX_TRAJECTORY_CONFIGS = 12


def _circle_starts_goals(n: int, ws_lo: float, ws_hi: float, margin: float = 0.7):
    center = 0.5 * (ws_lo + ws_hi)
    radius = 0.5 * (ws_hi - ws_lo) - margin
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    perturb = 0.4 * np.sin(np.arange(n) * 1.7 + 0.5)
    goal_angles = angles + np.pi + perturb
    starts = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    goals = center + radius * np.stack([np.cos(goal_angles), np.sin(goal_angles)], axis=1)
    return starts, goals


def make_env(ep: dict):
    n = int(ep["n_agents"])
    starts, goals = _circle_starts_goals(n, ep["ws_lo"], ep["ws_hi"])
    pair_idx = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32).reshape(-1, 2)

    class E:
        pass

    env = E()
    env.dt = ep["dt"]
    env.u_max = ep["u_max"]
    env.n_agents = n
    env.ws_lo = ep["ws_lo"]
    env.ws_hi = ep["ws_hi"]
    env.starts = starts
    env.goals = goals
    env.pair_idx = pair_idx
    return env


def combo_name(solver: str, batch_size: int) -> str:
    return f"{solver}_bs{batch_size}"


def resolve_trained_subdir(trained: Path, solver: str, combo: dict) -> Path | None:
    """Prefer config path; else pick a successful summary entry for this solver (OOM-sweep layouts)."""
    direct = trained / combo_name(combo["solver"], int(combo["batch_size"]))
    if direct.is_dir():
        return direct
    summary_path = trained / "summary.yaml"
    if not summary_path.is_file():
        return None
    summary = yaml.safe_load(summary_path.read_text())
    best: Path | None = None
    best_bs = -1
    for folder_name, meta in summary.items():
        c = meta.get("combo") or {}
        if c.get("solver") != solver:
            continue
        if meta.get("training_status") != STATUS_SUCCESS:
            continue
        sub = trained / folder_name
        if not sub.is_dir() or not (sub / "rollouts.npz").exists():
            continue
        bs = int(c.get("batch_size", 0))
        if bs > best_bs:
            best_bs = bs
            best = sub
    return best


def latest_nominal_run() -> Path:
    cands = [p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith("oom_sweep")]
    if not cands:
        raise SystemExit(f"no non-oom_sweep run folders under {DATA}")
    # Names are YYYYMMDD_HHMMSS — lexicographic sort == chronological
    return sorted(cands, key=lambda p: p.name)[-1]


def plot_losses(run: Path, trained_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    n_curves = 0
    for k, sub in enumerate(sorted(p for p in trained_dir.iterdir() if p.is_dir())):
        losses = np.load(sub / "losses.npy")
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if len(losses) == 0:
            continue
        ax.semilogy(losses, lw=1.5, label=f"{sub.name} [{meta['training_status_name']}]")
        n_curves += 1
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training loss — run {run.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  losses ({n_curves} curves) -> {out_path.name}")


def _agent_colors(n: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if cycle:
        return [cycle[i % len(cycle)] for i in range(n)]
    cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def _draw_obstacles(ax, oc, or_):
    for i in range(len(oc)):
        ax.add_patch(Circle(oc[i], or_[i], color="0.5", alpha=0.55, zorder=3))
        ax.add_patch(Circle(oc[i], or_[i], fill=False, edgecolor="0.2", ls="--", lw=0.8, zorder=3))


def _setup_workspace(ax, env, colors):
    ax.set_xlim(env.ws_lo, env.ws_hi)
    ax.set_ylim(env.ws_lo, env.ws_hi)
    ax.set_aspect("equal")
    for a in range(env.n_agents):
        ax.plot(*env.starts[a], "o", color=colors[a], markersize=8, zorder=6)
        ax.plot(*env.goals[a], "*", color=colors[a], markersize=12, zorder=6)
    ax.grid(True, alpha=0.25)


def plot_trajectories(env, sub: Path, name: str, out_path: Path, *, max_configs: int) -> None:
    rollouts = np.load(sub / "rollouts.npz", allow_pickle=True)
    positions = rollouts["positions"]
    test = np.load(sub.parent.parent / "test.npz", allow_pickle=True)
    oc_all, or_all = test["cfg_obs_centers"], test["cfg_obs_radii"]

    n = min(len(positions), max(1, max_configs))
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    colors = _agent_colors(env.n_agents)
    for k in range(n):
        ax = axes[k // cols][k % cols]
        _draw_obstacles(ax, oc_all[k], or_all[k])
        pos = positions[k]
        for a in range(env.n_agents):
            ax.plot(pos[:, a, 0], pos[:, a, 1], color=colors[a], lw=1.8, alpha=0.9, zorder=5)
        _setup_workspace(ax, env, colors)
        ax.set_title(f"config {k}")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"BarrierNet[{name}] — test trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  trajectories[{name}] -> {out_path.name}")


def _fig_to_pil(fig, dpi: int = 100) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()


def _save_gif(frames: list[Image.Image], path: Path, fps: int = 8) -> None:
    duration = max(1, int(1000 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def make_animation(env, sub: Path, name: str, out_path: Path, anim_idx: int = 0, n_trail: int = 8, arrow_scale: float = 0.8):
    rollouts = np.load(sub / "rollouts.npz", allow_pickle=True)
    positions = rollouts["positions"][anim_idx]
    controls = rollouts["controls"][anim_idx]
    u_noms = rollouts["u_noms"][anim_idx]
    test = np.load(sub.parent.parent / "test.npz", allow_pickle=True)
    oc, or_ = test["cfg_obs_centers"][anim_idx], test["cfg_obs_radii"][anim_idx]

    colors = _agent_colors(env.n_agents)
    frames: list[Image.Image] = []
    for t in range(len(controls)):
        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_obstacles(ax, oc, or_)
        t_start = max(0, t - n_trail)
        for a in range(env.n_agents):
            for s in range(t_start, t):
                fade = 0.2 + 0.8 * (s - t_start) / max(t - t_start, 1)
                ax.plot(positions[s : s + 2, a, 0], positions[s : s + 2, a, 1], color=colors[a], lw=2.0, alpha=fade, zorder=4)
            px, py = positions[t, a]
            ax.plot(px, py, "o", color=colors[a], markersize=10, zorder=7)
            ax.annotate(
                "",
                xy=(px + arrow_scale * u_noms[t, a, 0], py + arrow_scale * u_noms[t, a, 1]),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2, ls="--"),
            )
            ax.annotate(
                "",
                xy=(px + arrow_scale * controls[t, a, 0], py + arrow_scale * controls[t, a, 1]),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=colors[a], lw=1.8),
            )
        _setup_workspace(ax, env, colors)
        ax.set_title(f"BarrierNet-{name} ({env.n_agents} agents) | step {t}")
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    _save_gif(frames, out_path, fps=8)
    print(f"  animation[{name}] ({len(frames)} frames) -> {out_path.name}")


def main(max_configs: int | None = None) -> None:
    run = latest_nominal_run()
    cfg_path = run / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"missing {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    env = make_env(cfg["environment_parameters"])
    vp = cfg.get("visualize_parameters") or {}
    n_traj = DEFAULT_MAX_TRAJECTORY_CONFIGS if max_configs is None else max_configs
    if max_configs is None and "max_trajectory_configs" in vp:
        n_traj = int(vp["max_trajectory_configs"])
    n_traj = max(1, n_traj)
    trained = run / "trained"
    if not trained.is_dir():
        raise SystemExit(f"missing trained dir {trained}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ASSETS / f"nominal_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run}\nOutput: {out_dir}")
    plot_losses(run, trained, out_dir / "loss_comparison.png")

    by_solver = {c["solver"]: c for c in cfg["training_parameters"]["combinations"]}
    for solver in vp.get("figures", []):
        if solver not in by_solver:
            print(f"  [skip figure] {solver} not in combinations")
            continue
        sub = resolve_trained_subdir(trained, solver, by_solver[solver])
        if sub is None:
            print(f"  [skip figure] {solver}: no trained subdir with rollouts")
            continue
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if meta["training_status"] != STATUS_SUCCESS or not (sub / "rollouts.npz").exists():
            print(f"  [skip figure] {solver}: status={meta['training_status_name']}")
            continue
        plot_trajectories(env, sub, solver, out_dir / f"trajectories_{solver}.png", max_configs=n_traj)

    for solver in vp.get("animations", []):
        if solver not in by_solver:
            print(f"  [skip anim] {solver} not in combinations")
            continue
        sub = resolve_trained_subdir(trained, solver, by_solver[solver])
        if sub is None:
            print(f"  [skip anim] {solver}: no trained subdir with rollouts")
            continue
        meta = yaml.safe_load((sub / "meta.yaml").read_text())
        if meta["training_status"] != STATUS_SUCCESS or not (sub / "rollouts.npz").exists():
            print(f"  [skip anim] {solver}: status={meta['training_status_name']}")
            continue
        make_animation(env, sub, solver, out_dir / f"anim_{solver}.gif")

    print(f"\nDone. Assets -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-configs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "how many test rollout configs to show per trajectory figure "
            f"(default: visualize_parameters.max_trajectory_configs in the run config, else {DEFAULT_MAX_TRAJECTORY_CONFIGS})"
        ),
    )
    args = ap.parse_args()
    main(max_configs=args.max_configs)
