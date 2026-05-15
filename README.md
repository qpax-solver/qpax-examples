# qpax examples

Standalone examples and benchmarks for the [qpax](#) differentiable QP solver.

The examples here exercise both `qpax` backends:

* `e` — explicit predictor-corrector PDIP
* `i` — implicit retraction-manifold PDIP

each in single (`32`) or double (`64`) precision, giving four solver tokens
(`e32`, `e64`, `i32`, `i64`) used by the CLI flags below.

## Installation

1. Install the `qpax` solver first by following the instructions in
   [qpax](#) *(todo: add details)*.
2. Install the additional Python dependencies for the examples:

   ```bash
   pip install -r requirements.txt
   ```

The examples are tested on Ubuntu 20.04 with an NVIDIA GPU running CUDA 12.
JAX picks up the GPU automatically when `qpax[cuda12]` is installed.

## Running the examples

### Examples 1: Gradient fidelity and robustness in single precision

For comparing both explicit/implicit backends for multiple problem sizes at a fixed relaxation value, run

```bash
python examples/quality_gradients/fixed_kappa/quality_gradient_fixedkappa_benchmark.py
```

This generates a figure at
`examples/quality_gradients/fixed_kappa/gallery/quality_gradient_fixedkappa_benchmark_kappa0.0001.png`.

For comparing both explicit/implicit backends for multiple relaxation parameters at a fixed problem size at a fixed relaxation value, run

```bash
python examples/quality_gradients/fixed_size/quality_gradient_benchmark.py
```

This generates a figure at
`examples/quality_gradients/fixed_size/gallery/quality_gradient_benchmark_N20_M25.png`.

### Examples 2: Bilevel trajectory optimization

```bash
python examples/trajopt/run_benchmark.py --solvers i32 e32 --kappa 1e-4 --environment forest
```

Options:

* `--solvers`: any combination of `i32`, `i64`, `e32`, `e64`
* `--kappa`: relaxation parameter, e.g. `1e-4`
* `--environment`: `forest` or `office`

Outputs a per-solver/kappa trajectory figure plus a kappa-vs-navigation-time overview under `examples/trajopt/gallery/<timestamp>/`.

### Examples 3: Multi-agent safety filter

Run the full pipeline (data generation, training, validation, visualization):

```bash
python examples/safety_filter_multiagent/main.py \
    --config examples/safety_filter_multiagent/nominal_run/config_nominal_run.yaml
```

Tunable parameters live in
[`config_nominal_run.yaml`](examples/safety_filter_multiagent/nominal_run/config_nominal_run.yaml).

Generated figures and animations land under
`examples/safety_filter_multiagent/assets/<timestamp>/`.


## License

This project is licensed under the Apache License 2.0 — see the
[qpax](#) repository for details.
