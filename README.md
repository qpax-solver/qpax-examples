# qpax examples

Standalone examples and benchmarks for the [qpax](#) differentiable QP solver.

Each example exercises both `qpax` backends:

* `e` — explicit predictor-corrector PDIP
* `i` — implicit retraction-manifold PDIP

each in single (`32`) or double (`64`) precision, giving four solver tokens
(`e32`, `e64`, `i32`, `i64`) used by the CLI flags inside each example.

## Quickstart

```bash
# 1. Install qpax (see Installation below) and the example deps:
pip install -r requirements.txt

# 2. Run the smallest example end-to-end:
python examples/backend_comparison/fixed_kappa/quality_gradient_fixedkappa_benchmark.py
```

## Examples

| Example | Method | Description | Paper |
| --- | --- | --- | --- |
| [`backend_comparison`](examples/backend_comparison/)<br><img src="docs/quality_gradient_hero_figure_cropped_qpax.png" width="220"> | Theoretical comparison | Comparison of implicit and explicit backend of the qpax solver. | [Arrizabalaga et al.](todo) |
| [`bilevel_trajectory_optimization`](examples/bilevel_trajectory_optimization/)<br><img src="docs/forest_bilevel_tol_1e-05_qpax.png" width="220"> | Bilevel trajectory optimization | Inner QP ensures safety and smoothness; outer L-BFGS reduces navigation time. | [Mellinger et al.](https://ieeexplore.ieee.org/abstract/document/5980409/?casa_token=s0gH-F5fiMAAAAAA:D9MR5jPBzJ6sRLVuqPaOUojz_rHMyWj6K1ustjUnrOYKgRN6CszvmTtullcCaLQv5iclZrD1Ig) |
| [`learning_safety_filter`](examples/learning_safety_filter/)<br><img src="docs/learning_safety_filter.gif" width="220"> | Learning from demonstrations | Learning a multi-agent safety-filter CBF from expert demonstrations. | [Xiao et al.](https://ieeexplore.ieee.org/abstract/document/10077790?casa_token=buet2dfHOkwAAAAA:ewUqvUrxVszaXjj3iXqfkEq7MCeRLTs1q7PadFM0H2c8e0jgwfaMSS5tblJY2usrpuxIkM6Gvg) |

Each example folder ships its own `README.md` with the exact run command,
CLI/config options, and output paths.

## Installation

1. Install the `qpax` solver first by following the instructions in
   [qpax](#) *(todo: add details)*.
2. Install the additional Python dependencies for the examples:

   ```bash
   pip install -r requirements.txt
   ```

The examples are tested on Ubuntu 20.04 with an NVIDIA GPU running CUDA 12.
JAX picks up the GPU automatically when `qpax[cuda12]` is installed.

## Citation

If you use these examples or the qpax solver in academic work, please cite:

```bibtex
@article{arrizabalaga_qpax,
  title   = {qpax: TODO},
  author  = {Arrizabalaga, Jon and others},
  year    = {TODO}
}
```

*(BibTeX entry will be finalized at solver release.)*

## License

This project is licensed under the Apache License 2.0 — see the
[qpax](#) repository for details.
