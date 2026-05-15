"""Generate (state, u_safe) training data with the oracle QP solver."""
from datetime import datetime
from pathlib import Path
import shutil
import yaml
import numpy as np
import jax.numpy as jnp

from utils import (
    DATA_DIR, ROOT, load_config, make_env, random_obstacles,
    pairwise_cbf_np, build_oracle_layer, qp_dimensions,
)


def collect_trajectories(env, dg, split, layer):
    single, batched, dtype, is_jax = layer
    n_configs = dg[f"n_configs_{split}"]
    rng = np.random.RandomState(dg[f"seed_{split}"])

    obs_c_all, obs_r_all = [], []
    for _ in range(n_configs):
        oc, or_ = random_obstacles(env, rng)
        obs_c_all.append(oc); obs_r_all.append(or_)
    obs_c_all = np.array(obs_c_all); obs_r_all = np.array(obs_r_all)

    pos = np.tile(env.starts, (n_configs, 1, 1)).astype(np.float64)
    goals_b = np.broadcast_to(env.goals, (n_configs, env.n_agents, 2))
    active = np.ones(n_configs, dtype=bool)

    states, oc_list, or_list, controls = [], [], [], []
    pos_history = [pos.copy()]
    cast = (lambda a: jnp.asarray(a, dtype=dtype)) if is_jax else (lambda a: np.asarray(a, dtype=dtype))

    for _ in range(dg["n_steps"]):
        d_goal = np.linalg.norm(pos - goals_b, axis=2)
        active &= ~np.all(d_goal < env.tol_goal, axis=1)
        if not active.any():
            break

        diff = goals_b - pos
        norms = np.maximum(np.linalg.norm(diff, axis=2, keepdims=True), 0.1)
        u_nom = np.clip(dg["pd_gain"] * diff / norms, -env.u_max, env.u_max)
        u_nom[d_goal < env.tol_goal] = 0.0
        u_nom_flat = u_nom.reshape(n_configs, env.state_dim)

        d = pos[:, :, None, :] - obs_c_all[:, None, :, :]
        h_obs = np.sum(d ** 2, axis=3) - obs_r_all[:, None, :] ** 2
        grad_h_obs = 2.0 * d
        alpha_h_obs = dg["alpha_fixed"] * h_obs

        h_pair, grad_h_pair = pairwise_cbf_np(env, pos)
        alpha_h_pair = dg["alpha_fixed"] * h_pair

        u_safe_flat = np.asarray(batched(
            cast(u_nom_flat), cast(grad_h_obs), cast(alpha_h_obs),
            cast(grad_h_pair), cast(alpha_h_pair),
        ))
        u_safe = u_safe_flat.reshape(n_configs, env.n_agents, 2)

        idx = np.where(active)[0]
        for c in idx:
            states.append(pos[c].copy()); oc_list.append(obs_c_all[c].copy())
            or_list.append(obs_r_all[c].copy()); controls.append(u_safe[c].copy())

        pos[idx] = pos[idx] + env.dt * u_safe[idx]
        pos_history.append(pos.copy())

    pos_hist = np.array(pos_history)
    trajectories = []
    for c in range(n_configs):
        traj = pos_hist[:, c, :, :]
        for t in range(1, len(traj)):
            if np.all(np.linalg.norm(traj[t] - env.goals, axis=1) < env.tol_goal):
                traj = traj[:t + 1]; break
        trajectories.append((traj, obs_c_all[c], obs_r_all[c]))

    return (np.array(states), np.array(oc_list), np.array(or_list),
            np.array(controls), trajectories, obs_c_all, obs_r_all)


def make_run_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = DATA_DIR / stamp
    run.mkdir()
    return run


def main(config_path=None):
    cfg = load_config(config_path)
    env = make_env(cfg["environment_parameters"])
    dg = cfg["data_generation_parameters"]
    used_config_path = Path(config_path) if config_path else (ROOT / "config.yaml")

    print(f"Data-generation oracle: {dg['oracle_solver']}")
    print(f"Env: {env.n_agents} agents, dt={env.dt}, u_max={env.u_max}, "
          f"obstacles={env.n_obs}, workspace=[{env.ws_lo},{env.ws_hi}]^2")

    layer = build_oracle_layer(env, dg["oracle_solver"])
    qp_dims = qp_dimensions(env)
    print(f"QP dims: {qp_dims}")

    run = make_run_dir()
    print(f"Saving to {run}")

    for split in ("train", "test"):
        s, oc, or_, u, trajs, oc_all, or_all = collect_trajectories(env, dg, split, layer)
        print(f"  {split}: {len(s)} samples from {len(trajs)} configs")
        np.savez(
            run / f"{split}.npz",
            states=s, obs_centers=oc, obs_radii=or_, controls=u,
            cfg_obs_centers=oc_all, cfg_obs_radii=or_all,
            trajectories=np.array(trajs, dtype=object),
            allow_pickle=True,
        )

    with open(run / "qp_dims.yaml", "w") as f:
        yaml.safe_dump(qp_dims, f, sort_keys=False)
    shutil.copy(used_config_path, run / "config.yaml")
    print("Done.")
    return run


if __name__ == "__main__":
    main()
