# Learning safety filter (multi-agent)

End-to-end pipeline that learns a multi-agent **control barrier function
(CBF) safety filter** from expert demonstrations. An oracle QP-based
safety filter generates `(state, u_safe)` pairs; a neural network is
trained to imitate it, optionally with a differentiable qpax QP layer on
top.

Pipeline stages, all driven by `main.py`:

1. `data_generation` — roll out the oracle filter to collect training pairs.
2. `train` — fit the learned filter (multiple solver variants in parallel).
3. `validate` — evaluate on held-out test configurations.
4. `visualize` — render figures and animations.

## Run

From the repo root:

```bash
python examples/learning_safety_filter/main.py \
    --config examples/learning_safety_filter/nominal_run/config_nominal_run.yaml
```

## Options

All tunables live in the YAML config — there are no other CLI flags.

| Section | Key knobs |
| --- | --- |
| `environment_parameters` | `n_agents`, `n_obs`, `agent_radius`, `dt`, `u_max` |
| `data_generation_parameters` | `oracle_solver`, `n_configs_train`, `n_configs_test`, `n_steps` |
| `training_parameters` | `n_epochs`, `lr`, `combinations` (list of `solver` + `batch_size`) |
| `visualize_parameters` | `figures`, `animations` (which solver variants to render) |

See [`nominal_run/config_nominal_run.yaml`](nominal_run/config_nominal_run.yaml)
for a complete annotated example.

## Outputs

Generated artifacts land under a timestamped folder:

```
examples/learning_safety_filter/assets/<timestamp>/
    figures/      # validation plots per solver variant
    animations/   # rollouts for the selected variants
    models/       # trained network checkpoints
```

## Reference

[Xiao et al., *BarrierNet: Differentiable Control Barrier Functions for
Learning of Safe Robot
Control*, IEEE T-RO 2023](https://ieeexplore.ieee.org/abstract/document/10077790?casa_token=buet2dfHOkwAAAAA:ewUqvUrxVszaXjj3iXqfkEq7MCeRLTs1q7PadFM0H2c8e0jgwfaMSS5tblJY2usrpuxIkM6Gvg)
