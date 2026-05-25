# Backend comparison

Empirical comparison of the **explicit** (`e`) and **implicit** (`i`) qpax
backends in terms of gradient fidelity and robustness, especially in
single precision (`32`). Two sweeps:

* `fixed_kappa/` — sweep problem size `N` at fixed relaxation `kappa`.
* `fixed_size/` — sweep relaxation `kappa` at fixed problem size.

Each script compares solver gradients against a hard-projection
reference and reports absolute / relative error.

## Run

From the repo root:

```bash
# Sweep over problem size N (fixed kappa = 1e-4):
python examples/backend_comparison/fixed_kappa/quality_gradient_fixedkappa_benchmark.py

# Sweep over relaxation kappa (fixed N = 20, M = 25):
python examples/backend_comparison/fixed_size/quality_gradient_benchmark.py
```

Neither script takes CLI flags — parameters live at the top of each file.

## Outputs

Figures are written under each variant's local `gallery/` directory:

* `examples/backend_comparison/fixed_kappa/gallery/quality_gradient_fixedkappa_benchmark_kappa0.0001.png`
* `examples/backend_comparison/fixed_size/gallery/quality_gradient_benchmark_N20_M25.png`

A standalone NaN-tracing analysis can be run via:

```bash
python examples/backend_comparison/fixed_size/quality_gradient_nan_tracing.py
python examples/backend_comparison/fixed_size/plot_nan_tracing.py
```

## Reference
[Arrizabalaga et al., *A Differentiable Interior-Point Method in Single Precision*](https://arxiv.org/abs/2605.17913)
