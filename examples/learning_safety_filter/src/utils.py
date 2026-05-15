"""Shared building blocks: environment geometry, QP layers, models."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
import cvxpy as cp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from functools import partial

import qpax

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

# Status codes used by train.py to record how a run terminated.
STATUS_SUCCESS, STATUS_NAN, STATUS_OOM = 0, 1, 2
STATUS_NAMES = {0: "success", 1: "nan", 2: "oom"}

SOLVER_CONFIGS = {
    "i64":      dict(solver="i", use_f64=True,  solver_tol=1e-8, target_kappa=1e-4),
    "i32":      dict(solver="i", use_f64=False, solver_tol=1e-8, target_kappa=1e-4),
    "e64":      dict(solver="e", use_f64=True,  solver_tol=1e-3, target_kappa=1e-3),
    "e32":      dict(solver="e", use_f64=False, solver_tol=1e-3, target_kappa=1e-3),
    "external": dict(solver="external", use_f64=True),
}
_SOLVERS = {
    "i": partial(qpax.solve_qp_primal, backend="i"),
    "e": partial(qpax.solve_qp_primal, backend="e"),
}


def load_config(path=None, run_dir=None):
    if path is not None:
        path = Path(path)
    elif run_dir is not None and (Path(run_dir) / "config.yaml").exists():
        path = Path(run_dir) / "config.yaml"
    else:
        path = ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Env:
    """Cached geometry derived from environment_parameters."""
    dt: float
    u_max: float
    tol_goal: float
    n_obs: int
    n_agents: int
    ws_lo: float
    ws_hi: float
    agent_radius: float
    starts: np.ndarray
    goals: np.ndarray
    pair_idx: np.ndarray

    @property
    def state_dim(self): return 2 * self.n_agents

    @property
    def n_pairs(self): return self.n_agents * (self.n_agents - 1) // 2


def make_env(env_cfg):
    n_agents = env_cfg["n_agents"]
    starts, goals = _circle_starts_goals(n_agents, env_cfg["ws_lo"], env_cfg["ws_hi"])
    pair_idx = np.array(
        [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)],
        dtype=np.int32,
    ).reshape(-1, 2)
    return Env(
        dt=env_cfg["dt"], u_max=env_cfg["u_max"], tol_goal=env_cfg["tol_goal"],
        n_obs=env_cfg["n_obs"], n_agents=n_agents,
        ws_lo=env_cfg["ws_lo"], ws_hi=env_cfg["ws_hi"],
        agent_radius=env_cfg["agent_radius"],
        starts=starts, goals=goals, pair_idx=pair_idx,
    )


def _circle_starts_goals(n, ws_lo, ws_hi, margin=0.7):
    center = 0.5 * (ws_lo + ws_hi)
    radius = 0.5 * (ws_hi - ws_lo) - margin
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    perturb = 0.4 * np.sin(np.arange(n) * 1.7 + 0.5)
    goal_angles = angles + np.pi + perturb
    starts = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    goals = center + radius * np.stack([np.cos(goal_angles), np.sin(goal_angles)], axis=1)
    return starts, goals


def random_obstacles(env, rng):
    endpoints = np.vstack([env.starts, env.goals])
    centers, radii = [], []
    for _ in range(env.n_obs):
        for _attempt in range(100):
            c = rng.uniform(1.5, 8.5, size=2)
            r = rng.uniform(0.4, 0.9)
            if all(np.linalg.norm(c - p) > r + 1.0 for p in endpoints):
                if all(np.linalg.norm(c - c2) > r + r2 + 0.5
                       for c2, r2 in zip(centers, radii)):
                    centers.append(c)
                    radii.append(r)
                    break
    return np.array(centers), np.array(radii)


def pairwise_cbf_np(env, pos):
    """Pairwise CBF (h, grad_h) over an arbitrary leading batch."""
    pi = env.pair_idx
    if env.n_pairs == 0:
        base = pos.shape[:-2]
        return (np.zeros(base + (0,), dtype=pos.dtype),
                np.zeros(base + (0, env.state_dim), dtype=pos.dtype))
    diff = pos[..., pi[:, 0], :] - pos[..., pi[:, 1], :]
    h_pair = np.sum(diff ** 2, axis=-1) - (2 * env.agent_radius) ** 2
    grad = np.zeros(diff.shape[:-2] + (env.n_pairs, env.state_dim))
    for k, (i, j) in enumerate(pi):
        grad[..., k, 2*i:2*i+2] = 2.0 * diff[..., k, :]
        grad[..., k, 2*j:2*j+2] = -2.0 * diff[..., k, :]
    return h_pair, grad


# ---------------------------------------------------------------------- #
# QP layers


def make_qp_layer(env, cfg):
    """Build (single, batched) jitted QP solvers for a given solver config."""
    solve_fn = _SOLVERS[cfg["solver"]]
    dtype = jnp.float64 if cfg["use_f64"] else jnp.float32
    sd, na, no = env.state_dim, env.n_agents, env.n_obs

    Q_const = jnp.eye(sd, dtype=dtype)
    A_const = jnp.zeros((0, sd), dtype=dtype)
    b_const = jnp.zeros((0,), dtype=dtype)
    box_G = jnp.concatenate([jnp.eye(sd, dtype=dtype), -jnp.eye(sd, dtype=dtype)], axis=0)
    box_h = jnp.full((2 * sd,), env.u_max, dtype=dtype)

    def single(u_nom, grad_h_obs, alpha_h_obs, grad_h_pair, alpha_h_pair):
        G_obs = jnp.zeros((na, no, sd), dtype=dtype)
        for i in range(na):
            G_obs = G_obs.at[i, :, 2*i:2*i+2].set(-grad_h_obs[i])
        G_obs = G_obs.reshape(na * no, sd)
        G = jnp.concatenate([box_G, G_obs, -grad_h_pair], axis=0)
        h = jnp.concatenate([box_h, alpha_h_obs.reshape(-1), alpha_h_pair], axis=0)
        return solve_fn(
            Q_const, -u_nom, A_const, b_const, G, h,
            solver_tol=cfg["solver_tol"], target_kappa=cfg["target_kappa"], max_iter=30,
        )

    return single, jax.jit(jax.vmap(single)), dtype


def make_external_qp_layer(env):
    """float64 QP via cvxpy + clarabel — used as data-generation oracle."""
    sd, na, no, npairs = env.state_dim, env.n_agents, env.n_obs, env.n_pairs
    dtype = np.float64
    Q_const = np.eye(sd, dtype=dtype)
    box_G = np.concatenate([np.eye(sd, dtype=dtype), -np.eye(sd, dtype=dtype)], axis=0)
    box_h = np.full((2 * sd,), env.u_max, dtype=dtype)
    n_ineq = 2 * sd + na * no + npairs

    u = cp.Variable(sd)
    q_param = cp.Parameter(sd)
    G_param = cp.Parameter((n_ineq, sd))
    h_param = cp.Parameter(n_ineq)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(u, Q_const) - q_param @ u),
        [G_param @ u <= h_param],
    )

    def single(u_nom, grad_h_obs, alpha_h_obs, grad_h_pair, alpha_h_pair):
        G_obs = np.zeros((na, no, sd), dtype=dtype)
        for i in range(na):
            G_obs[i, :, 2*i:2*i+2] = -np.asarray(grad_h_obs[i], dtype=dtype)
        G_obs = G_obs.reshape(na * no, sd)
        G = np.concatenate([box_G, G_obs, -np.asarray(grad_h_pair, dtype=dtype)], axis=0)
        h = np.concatenate([box_h, np.asarray(alpha_h_obs, dtype=dtype).reshape(-1),
                            np.asarray(alpha_h_pair, dtype=dtype)], axis=0)
        q_param.value = np.asarray(u_nom, dtype=dtype)
        G_param.value = G
        h_param.value = h
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        if u.value is None:
            raise RuntimeError(f"External solver failed: {problem.status}")
        return np.asarray(u.value, dtype=dtype)

    def batched(u_nom, *rest):
        return np.stack([single(u_nom[k], *(r[k] for r in rest))
                         for k in range(len(u_nom))], axis=0)

    return single, batched, dtype


def build_oracle_layer(env, name):
    """Returns (single, batched, dtype, is_jax)."""
    if name == "external":
        s, b, d = make_external_qp_layer(env)
        return s, b, d, False
    s, b, d = make_qp_layer(env, SOLVER_CONFIGS[name])
    return s, b, d, True


def qp_dimensions(env):
    return dict(
        n_variables=env.state_dim,
        n_constraints_total=2 * env.state_dim + env.n_agents * env.n_obs + env.n_pairs,
        n_constraints_per_agent_cbf=env.n_obs,
        n_constraints_per_pairwise_cbf=1,
        n_box_constraints=2 * env.state_dim,
    )


# ---------------------------------------------------------------------- #
# Models


class BarrierNet(eqx.Module):
    """MLP -> (u_nom, alphas) -> centralized multi-agent CBF-QP -> u_safe."""
    backbone: eqx.nn.MLP
    u_nom_head: eqx.nn.Linear
    alpha_head: eqx.nn.Linear
    n_obs: int = eqx.field(static=True)
    n_agents: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    n_pairs: int = eqx.field(static=True)
    pair_idx: tuple = eqx.field(static=True)
    agent_radius: float = eqx.field(static=True)
    goals_flat: tuple = eqx.field(static=True)
    qp_solve: object = eqx.field(static=True)
    dtype: object = eqx.field(static=True)

    def __init__(self, env, key, qp_solve, dtype):
        keys = jax.random.split(key, 3)
        input_dim = env.state_dim + env.state_dim + env.n_obs * 3
        n_alpha_out = env.n_agents * env.n_obs + env.n_pairs
        self.backbone = eqx.nn.MLP(
            in_size=input_dim, out_size=64, width_size=64, depth=1,
            activation=jax.nn.relu, final_activation=jax.nn.relu, key=keys[0],
        )
        self.u_nom_head = eqx.nn.Linear(64, env.state_dim, key=keys[1])
        self.alpha_head = eqx.nn.Linear(64, n_alpha_out, key=keys[2])
        self.n_obs = env.n_obs
        self.n_agents = env.n_agents
        self.state_dim = env.state_dim
        self.n_pairs = env.n_pairs
        self.pair_idx = tuple((int(i), int(j)) for i, j in env.pair_idx)
        self.agent_radius = env.agent_radius
        self.goals_flat = tuple(env.goals.reshape(-1).tolist())
        self.qp_solve = qp_solve
        self.dtype = dtype

    def __call__(self, pos, obs_centers, obs_radii):
        d = self.dtype
        pos = pos.astype(d); obs_centers = obs_centers.astype(d); obs_radii = obs_radii.astype(d)
        goals_j = jnp.asarray(self.goals_flat, dtype=d)
        x = jnp.concatenate([pos.reshape(-1), goals_j, obs_centers.reshape(-1), obs_radii]).astype(d)

        h = self.backbone(x).astype(d)
        u_nom = self.u_nom_head(h).astype(d)
        alpha_logits = self.alpha_head(h).astype(d)
        alphas = (jax.nn.softplus(alpha_logits) + jnp.asarray(0.01, dtype=d)).astype(d)
        alpha_obs = alphas[: self.n_agents * self.n_obs].reshape(self.n_agents, self.n_obs)
        alpha_pair = alphas[self.n_agents * self.n_obs:]

        two = jnp.asarray(2.0, dtype=d)
        rsq = jnp.asarray((2 * self.agent_radius) ** 2, dtype=d)

        diff = (pos[:, None, :] - obs_centers[None, :, :]).astype(d)
        h_obs = ((diff ** 2).sum(axis=2) - obs_radii ** 2).astype(d)
        grad_h_obs = (two * diff).astype(d)
        alpha_h_obs = (alpha_obs * h_obs).astype(d)

        if self.n_pairs > 0:
            pi = jnp.asarray([k[0] for k in self.pair_idx], dtype=jnp.int32)
            pj = jnp.asarray([k[1] for k in self.pair_idx], dtype=jnp.int32)
            diff_p = (pos[pi] - pos[pj]).astype(d)
            h_pair = ((diff_p ** 2).sum(axis=1) - rsq).astype(d)
            grad_h_pair = jnp.zeros((self.n_pairs, self.state_dim), dtype=d)
            for k, (i, j) in enumerate(self.pair_idx):
                grad_h_pair = grad_h_pair.at[k, 2*i:2*i+2].set((two * diff_p[k]).astype(d))
                grad_h_pair = grad_h_pair.at[k, 2*j:2*j+2].set((-two * diff_p[k]).astype(d))
            alpha_h_pair = (alpha_pair * h_pair).astype(d)
        else:
            grad_h_pair = jnp.zeros((0, self.state_dim), dtype=d)
            alpha_h_pair = jnp.zeros((0,), dtype=d)

        u_safe = self.qp_solve(u_nom, grad_h_obs, alpha_h_obs, grad_h_pair, alpha_h_pair)
        return u_safe.astype(d), u_nom, alphas


class PlainNN(eqx.Module):
    """MLP-only baseline: maps (pos, obstacles) directly to a clamped control."""
    backbone: eqx.nn.MLP
    head: eqx.nn.Linear
    n_obs: int = eqx.field(static=True)
    n_agents: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    u_max: float = eqx.field(static=True)
    goals_flat: tuple = eqx.field(static=True)
    dtype: object = eqx.field(static=True)

    def __init__(self, env, key, dtype):
        keys = jax.random.split(key, 2)
        input_dim = env.state_dim + env.state_dim + env.n_obs * 3
        self.backbone = eqx.nn.MLP(
            in_size=input_dim, out_size=64, width_size=128, depth=2,
            activation=jax.nn.relu, final_activation=jax.nn.relu, key=keys[0],
        )
        self.head = eqx.nn.Linear(64, env.state_dim, key=keys[1])
        self.n_obs = env.n_obs
        self.n_agents = env.n_agents
        self.state_dim = env.state_dim
        self.u_max = env.u_max
        self.goals_flat = tuple(env.goals.reshape(-1).tolist())
        self.dtype = dtype

    def __call__(self, pos, obs_centers, obs_radii):
        d = self.dtype
        pos = pos.astype(d); obs_centers = obs_centers.astype(d); obs_radii = obs_radii.astype(d)
        goals_j = jnp.asarray(self.goals_flat, dtype=d)
        x = jnp.concatenate([pos.reshape(-1), goals_j, obs_centers.reshape(-1), obs_radii]).astype(d)
        h = self.backbone(x).astype(d)
        u = (self.u_max * jnp.tanh(self.head(h).astype(d))).astype(d)
        # Mirror BarrierNet's signature so train/validate can share code paths.
        return u, u, jnp.zeros((0,), dtype=d)


def is_plain(combo):
    return combo["solver"] == "plain"


def build_model(env, combo, key):
    """Returns (model, dtype). Picks PlainNN for solver=='plain', else BarrierNet."""
    if is_plain(combo):
        model = PlainNN(env, key, dtype=jnp.float64)
        return cast_inexact(model, jnp.float64), jnp.float64
    qp_single, _, dtype, _ = build_oracle_layer(env, combo["solver"])
    model = BarrierNet(env, key, qp_solve=qp_single, dtype=dtype)
    return cast_inexact(model, dtype), dtype


def cast_inexact(model, dtype):
    return jax.tree.map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, model,
    )


def count_params(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(int(np.prod(l.shape)) for l in leaves)


# ---------------------------------------------------------------------- #
# I/O helpers


def latest_run_dir(base):
    base = Path(base)
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def combo_name(combo):
    return f"{combo['solver']}_bs{combo['batch_size']}"
