"""Train one BarrierNet per (solver, batch_size) combination."""
import argparse
import time
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from utils import (
    DATA_DIR, load_config, make_env, build_model, count_params,
    latest_run_dir, combo_name, is_plain,
    STATUS_SUCCESS, STATUS_NAN, STATUS_OOM, STATUS_NAMES,
)


def _is_oom(exc):
    msg = repr(exc).lower()
    return "out of memory" in msg or "resource_exhausted" in msg or "oom" in msg


def train_combo(env, train_data, combo, n_epochs, lr):
    model, dtype = build_model(env, combo, jax.random.PRNGKey(42))
    print(f"  Parameters: {count_params(model):,}")

    s64 = jnp.asarray(train_data["states"], dtype=jnp.float64)
    oc64 = jnp.asarray(train_data["obs_centers"], dtype=jnp.float64)
    or64 = jnp.asarray(train_data["obs_radii"], dtype=jnp.float64)
    u64 = jnp.asarray(train_data["controls"].reshape(-1, env.state_dim), dtype=jnp.float64)
    n = len(s64)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(model, pos_b, oc_b, or_b, u_b):
        u_pred, _, _ = jax.vmap(model)(pos_b, oc_b, or_b)
        return jnp.mean((u_pred - u_b) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, pos_b, oc_b, or_b, u_b):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, pos_b, oc_b, or_b, u_b)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    losses = []
    status = STATUS_SUCCESS
    rng = np.random.RandomState(0)
    bs = combo["batch_size"]

    try:
        for epoch in range(n_epochs):
            t0 = time.perf_counter()
            perm = rng.permutation(n)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                pos_b = s64[idx].astype(dtype); oc_b = oc64[idx].astype(dtype)
                or_b = or64[idx].astype(dtype); u_b = u64[idx].astype(dtype)
                model, opt_state, loss = step(model, opt_state, pos_b, oc_b, or_b, u_b)
                lf = float(loss)
                if not np.isfinite(lf):
                    print(f"  NaN at epoch {epoch+1}, batch {n_batches+1} -> abort.")
                    status = STATUS_NAN
                    raise StopIteration
                epoch_loss += lf; n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: loss = {avg:.6f} | {time.perf_counter()-t0:.2f}s")
    except StopIteration:
        pass
    except Exception as e:
        if _is_oom(e):
            print(f"  OOM caught: {e}")
            status = STATUS_OOM
        else:
            raise

    return model, losses, status


def main(run_dir=None, config_path=None):
    run = run_dir or latest_run_dir(DATA_DIR)
    if run is None:
        raise RuntimeError("No data run found. Run data_generation.py first.")
    cfg = load_config(path=config_path, run_dir=run)
    env = make_env(cfg["environment_parameters"])
    tp = cfg["training_parameters"]
    print(f"Using data run: {run}")

    train_data = np.load(run / "train.npz", allow_pickle=True)
    print(f"Train samples: {len(train_data['states'])}")

    summary = {}
    for combo in tp["combinations"]:
        name = combo_name(combo)
        kind = "PlainNN" if is_plain(combo) else "BarrierNet"
        print(f"\n[{name}] Training {kind} (solver={combo['solver']}, batch_size={combo['batch_size']})")
        out = run / "trained" / name
        out.mkdir(parents=True, exist_ok=True)
        try:
            model, losses, status = train_combo(env, train_data, combo, tp["n_epochs"], tp["lr"])
        except Exception as e:
            print(f"  Unexpected failure: {e}")
            losses, status = [], STATUS_OOM if "memory" in repr(e).lower() else STATUS_NAN
            model = None

        if model is not None:
            eqx.tree_serialise_leaves(str(out / "model.eqx"), model)

        np.save(out / "losses.npy", np.asarray(losses, dtype=np.float64))
        meta = dict(combo=combo, training_status=int(status),
                    training_status_name=STATUS_NAMES[status],
                    n_epochs_completed=len(losses))
        with open(out / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        summary[name] = meta
        print(f"  -> status: {STATUS_NAMES[status]} ({len(losses)} epochs)")

    with open(run / "trained" / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nSaved trained models under {run / 'trained'}")
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None)
    args = p.parse_args()
    main(run_dir=(DATA_DIR / args.run) if args.run else None)
