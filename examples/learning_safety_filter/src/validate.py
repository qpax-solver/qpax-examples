"""Roll out each trained BarrierNet on the test configurations."""
import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils import (
    DATA_DIR, load_config, make_env, build_model,
    latest_run_dir, combo_name, STATUS_SUCCESS, STATUS_NAMES,
)


def _load_model(env, combo, path):
    skel, dtype = build_model(env, combo, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(str(path), skel), dtype


def simulate(env, model, dtype, obs_c, obs_r, n_steps):
    @eqx.filter_jit
    def step_fn(m, p, oc, or_):
        return m(p, oc, or_)

    pos = env.starts.copy()
    positions = [pos.copy()]
    controls, u_noms, alphas_l = [], [], []
    static_v, pair_v = 0, 0
    oc_j = jnp.asarray(obs_c, dtype=dtype); or_j = jnp.asarray(obs_r, dtype=dtype)

    for _ in range(n_steps):
        if np.all(np.linalg.norm(pos - env.goals, axis=1) < env.tol_goal):
            break
        u_safe, u_nom, alphas = step_fn(model, jnp.asarray(pos, dtype=dtype), oc_j, or_j)
        u = np.asarray(u_safe).reshape(env.n_agents, 2)
        controls.append(u); u_noms.append(np.asarray(u_nom).reshape(env.n_agents, 2))
        alphas_l.append(np.asarray(alphas))
        pos = pos + env.dt * u
        positions.append(pos.copy())

        for a in range(env.n_agents):
            for i in range(len(obs_c)):
                if np.linalg.norm(pos[a] - obs_c[i]) < obs_r[i]:
                    static_v += 1
        for (i, j) in env.pair_idx:
            if np.linalg.norm(pos[i] - pos[j]) < 2 * env.agent_radius:
                pair_v += 1

    reached = bool(np.all(np.linalg.norm(positions[-1] - env.goals, axis=1) < 0.5))
    return dict(
        positions=np.array(positions),
        controls=np.array(controls) if controls else np.zeros((0, env.n_agents, 2)),
        u_noms=np.array(u_noms) if u_noms else np.zeros((0, env.n_agents, 2)),
        alphas=np.array(alphas_l) if alphas_l else np.zeros((0,)),
        static_violations=static_v, pair_violations=pair_v,
        violations=static_v + pair_v, reached_goal=reached,
    )


def main(run_dir=None, config_path=None):
    run = run_dir or latest_run_dir(DATA_DIR)
    if run is None:
        raise RuntimeError("No data run found.")
    cfg = load_config(path=config_path, run_dir=run)
    env = make_env(cfg["environment_parameters"])
    n_steps = cfg["data_generation_parameters"]["n_steps"]
    print(f"Validating run: {run}")

    test = np.load(run / "test.npz", allow_pickle=True)
    test_oc = test["cfg_obs_centers"]; test_or = test["cfg_obs_radii"]
    n_test = len(test_oc)
    print(f"Test configurations: {n_test}")

    summary = {}
    trained_dir = run / "trained"
    for combo in cfg["training_parameters"]["combinations"]:
        name = combo_name(combo)
        out = trained_dir / name
        meta = yaml.safe_load((out / "meta.yaml").read_text())
        if meta["training_status"] != STATUS_SUCCESS or not (out / "model.eqx").exists():
            print(f"  [{name}] skipped (status={meta['training_status_name']})")
            summary[name] = dict(meta, n_violations=None, n_reached=None)
            continue

        print(f"  [{name}] simulating...")
        model, dtype = _load_model(env, combo, out / "model.eqx")
        rollouts = []
        for k in range(n_test):
            rollouts.append(simulate(env, model, dtype, test_oc[k], test_or[k], n_steps))

        np.savez(
            out / "rollouts.npz",
            positions=np.array([r["positions"] for r in rollouts], dtype=object),
            controls=np.array([r["controls"] for r in rollouts], dtype=object),
            u_noms=np.array([r["u_noms"] for r in rollouts], dtype=object),
            alphas=np.array([r["alphas"] for r in rollouts], dtype=object),
            static_violations=np.array([r["static_violations"] for r in rollouts]),
            pair_violations=np.array([r["pair_violations"] for r in rollouts]),
            reached_goal=np.array([r["reached_goal"] for r in rollouts]),
            allow_pickle=True,
        )
        v = sum(r["violations"] for r in rollouts)
        g = sum(1 for r in rollouts if r["reached_goal"])
        cwv = sum(1 for r in rollouts if r["violations"] > 0)
        print(f"     violations={v} (configs_with_v={cwv}/{n_test}), reached_goal={g}/{n_test}")
        summary[name] = dict(meta, n_violations=int(v),
                             configs_with_violations=int(cwv), n_reached=int(g))

    with open(trained_dir / "validation_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nValidation summary -> {trained_dir / 'validation_summary.yaml'}")
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    main(run_dir=(DATA_DIR / args.run) if args.run else None, config_path=args.config)
