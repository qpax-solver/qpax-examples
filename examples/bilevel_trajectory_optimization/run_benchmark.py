"""
Accuracy sweep over (solver_tol, target_kappa) pairs for the L-BFGS trajopt.

Calls `lbfgs.run_trajopt` per (solver, tol, kappa), aggregates metrics, and
plots the kappa-vs-T sweep curve via visualize.py.

Run from the repo root:
    python examples/bilevel_trajectory_optimization/run_benchmark.py --solvers i32 e32 --kappa 1e-3 1e-4 --environment office
"""

from __future__ import annotations

import argparse
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lbfgs import run_trajopt, solver_cfg
from visualize import plot_sweep_kappa_vs_T, plot_trajectory


# ----------------------------- Configuration ------------------------------- #

WORLD = "office"  # "office" or "forest"
DEFAULT_SOLVER_TOL = 1e-4

PROBLEM = dict(
    n_bezier=4,
    continuity=2,
    weights=[0.1, 0.1, 1.0, 0.1],
    deriv_lower={1: np.array([-10.0, -10.0]), 2: np.array([-2.0, -2.0])},
    deriv_upper={1: np.array([10.0, 10.0]), 2: np.array([2.0, 2.0])},
    # deriv_lower={2: np.array([-2.0, -2.0])},
    # deriv_upper={2: np.array([2.0, 2.0])},
    deriv_init={1: np.zeros(2), 2: np.zeros(2)},
    deriv_end={1: np.zeros(2), 2: np.zeros(2)},
    t_init=3.0,
    time_penalty=100.0,
    # L-BFGS
    lbfgs_iters=35,
    lbfgs_memory=10,
    ls_c1=1e-4,
    ls_backtrack=0.5,
    ls_max_backtrack=30,
    t_seg_min=0.1,
    step_tol=1e-8,
)


# Sweep grid: list of (solver_tol, target_kappa) pairs evaluated for each solver.
DEFAULT_TOL_KAPPA_PAIRS = [
    (DEFAULT_SOLVER_TOL, 1e0),
    (DEFAULT_SOLVER_TOL, 1e-1),
    (DEFAULT_SOLVER_TOL, 1e-2),
    (DEFAULT_SOLVER_TOL, 1e-3),
]

# Solvers to sweep. Each entry is (solver, use_f64, regularize).
DEFAULT_SOLVERS_SWEEP = [
    ("i", False, False),  # i32
]

GALLERY_DIR = Path(__file__).resolve().parent / "gallery"
SAVE_TRAJECTORIES = True  # save each (solver,tol,kappa) trajectory image

SOLVER_ALIASES = {
    "i32": ("i", False, False),
    "i64": ("i", True, False),
    "e32": ("e", False, False),
    "e64": ("e", True, False),
}


def _solver_label(solver, use_f64):
    return f"{solver}{'64' if use_f64 else '32'}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the trajopt solver sweep and save the aggregate sweep plot plus "
            "per-configuration trajectory plots."
        )
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=sorted(SOLVER_ALIASES),
        default=None,
        help="Solver/precision variants to benchmark, e.g. i32 e32.",
    )
    parser.add_argument(
        "--kappa",
        nargs="+",
        type=float,
        default=None,
        help="Target kappa values to sweep, e.g. 1e-3 1e-4.",
    )
    parser.add_argument(
        "--solver-tol",
        nargs="+",
        type=float,
        default=None,
        help="Solver tolerance values to sweep. Defaults to 1e-4.",
    )
    parser.add_argument(
        "--environment",
        nargs="?",
        choices=("office", "forest"),
        const=WORLD,
        default=WORLD,
        help=(
            "Scenario to run. Accepts 'office' or 'forest'. If passed without a "
            "value, defaults to the configured world."
        ),
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Skip writing per-run trajectory PNGs.",
    )
    return parser.parse_args()


def _build_solvers_sweep(solver_names):
    if solver_names is None:
        return list(DEFAULT_SOLVERS_SWEEP)
    return [SOLVER_ALIASES[name] for name in solver_names]


def _build_tol_kappa_pairs(kappas, solver_tols):
    default_kappas = [kappa for _, kappa in DEFAULT_TOL_KAPPA_PAIRS]
    active_kappas = default_kappas if kappas is None else kappas
    active_tols = [DEFAULT_SOLVER_TOL] if solver_tols is None else solver_tols
    return [(tol, kappa) for tol in active_tols for kappa in active_kappas]


# --------------------------------- Main ------------------------------------ #


def main():
    args = parse_args()
    world = args.environment
    solvers_sweep = _build_solvers_sweep(args.solvers)
    tol_kappa_pairs = _build_tol_kappa_pairs(args.kappa, args.solver_tol)
    save_trajectories = not args.no_save_trajectories

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = GALLERY_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== sweep {timestamp} | world={world} ===")
    print(f"  pairs   : {len(tol_kappa_pairs)}")
    print(f"  solvers : {[_solver_label(s, u) for s, u, _ in solvers_sweep]}")
    print(f"  out dir : {out_dir.resolve()}")

    records = []
    results_for_plot = {}  # one trajectory per (solver, tol, kappa)

    for solver, use_f64, regularize in solvers_sweep:
        for tol, kappa in tol_kappa_pairs:
            cfg = solver_cfg(
                solver=solver,
                use_f64=use_f64,
                solver_tol=tol,
                target_kappa=kappa,
                regularize=regularize,
            )
            label = f"{solver}{'64' if use_f64 else '32'}_tol{tol:.0e}_k{kappa:.0e}"
            print(f"\n--- {label} ---")
            try:
                res = run_trajopt(world, cfg, PROBLEM, label=label, verbose=True)
            except Exception as e:
                print(f"  FAILED: {e}")
                records.append(
                    dict(
                        solver=solver,
                        use_f64=use_f64,
                        solver_tol=tol,
                        target_kappa=kappa,
                        T_total=np.nan,
                        best_cost=np.nan,
                        wall_time=np.nan,
                        n_iters=0,
                        converged=False,
                        ls_failed=True,
                    )
                )
                continue

            records.append(
                dict(
                    solver=solver,
                    use_f64=use_f64,
                    solver_tol=tol,
                    target_kappa=kappa,
                    T_total=res["T_total"],
                    best_cost=res["best_cost"],
                    wall_time=res["wall_time"],
                    n_iters=res["n_iters"],
                    converged=res["converged"],
                    ls_failed=res["ls_failed"],
                )
            )
            results_for_plot[label] = res

    # ----- Summary table ----- #
    print("\n=== summary ===")
    w = max(8, max(len(f"{r['solver']}") for r in records))
    print(
        f"  {'solver'.ljust(w)}  {'tol':>9}  {'kappa':>9}  {'T':>10}  {'iters':>6}  {'wall[s]':>8}  ok"
    )
    for r in records:
        ok = "y" if (r["converged"] and not r["ls_failed"]) else "n"
        T = "nan" if not np.isfinite(r["T_total"]) else f"{r['T_total']:10.4f}"
        wt = "nan" if not np.isfinite(r["wall_time"]) else f"{r['wall_time']:8.2f}"
        print(
            f"  {r['solver'].ljust(w)}  {r['solver_tol']:9.1e}  {r['target_kappa']:9.1e}  "
            f"{T:>10}  {r['n_iters']:>6d}  {wt:>8}  {ok}"
        )

    # ----- Save records as CSV (lightweight, no pandas) ----- #
    csv_path = out_dir / f"{world}_sweep.csv"
    with open(csv_path, "w") as f:
        f.write(
            "solver,use_f64,solver_tol,target_kappa,T_total,best_cost,wall_time,n_iters,converged,ls_failed\n"
        )
        for r in records:
            f.write(
                f"{r['solver']},{int(r['use_f64'])},{r['solver_tol']:.6e},{r['target_kappa']:.6e},"
                f"{r['T_total']:.6e},{r['best_cost']:.6e},{r['wall_time']:.4f},"
                f"{r['n_iters']},{int(r['converged'])},{int(r['ls_failed'])}\n"
            )
    print(f"Saved sweep CSV: {csv_path.resolve()}")

    # ----- Save primals (ctrl + t_opt + cfg) and scenario for post-hoc plotting ----- #
    if results_for_plot:
        scenario = next(iter(results_for_plot.values()))["scenario"]
        scen_path = out_dir / f"{world}_scenario.pkl"
        with open(scen_path, "wb") as f:
            pickle.dump(scenario, f)
        primals_dir = out_dir / "primals"
        primals_dir.mkdir(exist_ok=True)
        for label, res in results_for_plot.items():
            with open(primals_dir / f"{label}.pkl", "wb") as f:
                pickle.dump(
                    dict(ctrl=res["ctrl"], t_opt=res["t_opt"], cfg=res["cfg"]), f
                )
        print(f"Saved scenario: {scen_path.resolve()}")
        print(f"Saved {len(results_for_plot)} primals under {primals_dir.resolve()}")

    # ----- Sweep plot ----- #
    sweep_png = out_dir / f"{world}_kappa_vs_T.png"
    plot_sweep_kappa_vs_T(records, savepath=sweep_png)
    plt.close()
    print(f"Saved sweep plot: {sweep_png.resolve()}")
    png_paths = [sweep_png.resolve()]

    # ----- Optional: save individual trajectories ----- #
    traj_dir = out_dir / "trajectories"
    if save_trajectories:
        traj_dir.mkdir(exist_ok=True)
        for label, res in results_for_plot.items():
            tp = traj_dir / f"{label}.png"
            plot_trajectory(res, label, savepath=tp)
            plt.close()
            png_paths.append(tp.resolve())
        print(
            f"Saved {len(results_for_plot)} trajectory plot(s) under "
            f"{traj_dir.resolve()}"
        )
    else:
        print(
            f"Trajectory plots skipped (SAVE_TRAJECTORIES=False); would use {traj_dir.resolve()}"
        )

    print("\n=== saved PNG paths (click in terminal) ===")
    for p in png_paths:
        print(p)

    print("Done.")


if __name__ == "__main__":
    main()
