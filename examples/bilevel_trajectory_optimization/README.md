# Bilevel trajectory optimization

Bilevel optimization for robotic trajectory planning. The **inner** QP
(solved with qpax) enforces safety and smoothness over Bezier control
points; the **outer** L-BFGS loop differentiates through the inner QP to
reduce total navigation time `T`.

## Run

From the repo root:

```bash
python examples/bilevel_trajectory_optimization/run_benchmark.py \
    --solvers i32 e32 \
    --kappa 1e-4 \
    --environment forest
```

## Options

| Flag | Values | Notes |
| --- | --- | --- |
| `--solvers` | any of `i32`, `i64`, `e32`, `e64` | One or more solver/precision tokens. |
| `--kappa` | floats, e.g. `1e-3 1e-4` | Target relaxation values to sweep. |
| `--solver-tol` | floats | Solver tolerance sweep (default `1e-4`). |
| `--environment` | `forest`, `office` | Scenario to plan through. |
| `--no-save-trajectories` | flag | Skip per-run trajectory PNGs. |

## Outputs

Each run writes to a timestamped folder under
`examples/bilevel_trajectory_optimization/gallery/<timestamp>/`:

* `*_sweep.csv` — aggregated metrics
* `*_kappa_vs_T.png` — kappa vs. navigation-time overview
* `trajectories/*.png` — per-configuration trajectory plots (unless
  `--no-save-trajectories` is set)

To re-plot a previous run from its CSV (without redoing the sweep):

```bash
python examples/bilevel_trajectory_optimization/run_visualization.py
```

## Reference

[Mellinger et al., *Minimum snap trajectory generation and control for
quadrotors*, ICRA 2011](https://ieeexplore.ieee.org/abstract/document/5980409/?casa_token=s0gH-F5fiMAAAAAA:D9MR5jPBzJ6sRLVuqPaOUojz_rHMyWj6K1ustjUnrOYKgRN6CszvmTtullcCaLQv5iclZrD1Ig)
