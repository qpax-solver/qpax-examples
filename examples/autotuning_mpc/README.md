# Autotuning MPC

Minimal notebook example showing how to **learn MPC cost matrices from
demonstrations** by differentiating through a qpax QP solve. The setup uses a
2D double-integrator tracking a figure-eight reference:

1. generate noisy expert rollouts with a "true" MPC,
2. fit a `LearnedMPC` to match the demonstrated first control action,
3. visualize the learned controller against the expert.

In this example, the learner is given the true dynamics and control bounds, and
only learns the cost terms `Q` and `R`.

## Run

From the repo root, open:

```bash
jupyter notebook examples/autotuning_mpc/autotune_mpc.ipynb
```

All main knobs live in the first code cell:

* `PRECISION` — `f32` or `f64`
* `BACKEND` — qpax backend: `i` or `e`
* `TARGET_KAPPA` — qpax relaxation used for differentiation
* `SOLVER_TOL` — IPM convergence tolerance
* `MAX_ITER` — solver iteration cap

The notebook also exposes the demonstration and training settings
(`N_EXPERT`, `N_TRAJ`, `N_STEPS`, `N_EPOCHS`, `BATCH_SIZE`, `LR`) near the
start of sections 1 and 2.

## Outputs

Running the notebook writes the main figures to `examples/autotuning_mpc/assets/`:

* `loss.png` — training loss across optimization epochs
* `compare.png` — reference, true closed-loop trajectory, and learned controls
* `learned_mpc_anim.gif` — animated receding-horizon rollout of the learned MPC

## Reference
[Adabag et al., *Differentiable Model Predictive Control on the GPU*](https://arxiv.org/abs/2510.06179)
