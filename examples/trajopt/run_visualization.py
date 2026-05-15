"""
Re-plot a saved sweep from CSV (no re-use of the benchmark's cached primals).

By default, loads the latest `*_sweep.csv` from `examples/trajopt/sweep/gallery/`
(timestamped subfolders from `run_benchmark.py`). Set `GALLERY_RUN` to a
specific folder or CSV path to override.

With `SAVE_TRAJECTORIES=True`, re-runs `run_trajopt` per CSV row and saves PNGs
under `<run_dir>/trajectories/` using `PROBLEM` from `run_benchmark.py` (must
match the setup used to produce the CSV for meaningful comparison).

From the repo root:
    python examples/trajopt/sweep/run_visualization.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lbfgs import run_trajopt, solver_cfg
from run_benchmark import PROBLEM
from visualize import plot_sweep_kappa_vs_T, plot_trajectory


# ----------------------------- Configuration ------------------------------- #

# Output root: same as run_benchmark.py (timestamped subfolders with CSV + plots)
GALLERY_DIR = Path(__file__).resolve().parent / "gallery"

# If set, use this run (directory with exactly one `*_sweep.csv`, or path to the CSV).
# If None, use the latest timestamp subfolder of GALLERY_DIR that contains a sweep CSV.
GALLERY_RUN: Path | None = None

# If True, overwrite the PNGs next to the loaded CSV. If False, only show.
SAVE_PLOTS = True

# If True, re-run trajopt per CSV row and save trajectory PNGs (uses PROBLEM from run_benchmark.py).
SAVE_TRAJECTORIES = True


# --------------------------------- Helpers --------------------------------- #


def _sweep_csv_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*_sweep.csv"))


def _latest_run_dir(gallery_root: Path) -> Path | None:
    """Latest timestamp subdir of `gallery_root` that contains a sweep CSV, or None."""
    if not gallery_root.is_dir():
        return None
    candidates = [p for p in gallery_root.iterdir() if p.is_dir() and _sweep_csv_files(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _resolve_sweep_csv() -> Path:
    if GALLERY_RUN is not None:
        p = Path(GALLERY_RUN).resolve()
        if p.is_file():
            if p.name.endswith("_sweep.csv"):
                return p
            raise SystemExit(f"Not a sweep CSV: {p}")
        if p.is_dir():
            files = _sweep_csv_files(p)
            if len(files) != 1:
                raise SystemExit(
                    f"Expected exactly one *_sweep.csv in {p}, found {len(files)}"
                )
            return files[0]
        raise SystemExit(f"Path does not exist: {p}")

    latest = _latest_run_dir(GALLERY_DIR)
    if latest is None:
        raise SystemExit(
            f"No run folders with *_sweep.csv under {GALLERY_DIR}. "
            f"Run examples/trajopt/sweep/run_benchmark.py first, or set GALLERY_RUN "
            f"to a folder that contains a sweep CSV."
        )
    files = _sweep_csv_files(latest)
    if len(files) != 1:
        raise SystemExit(
            f"Expected exactly one *_sweep.csv in {latest}, found {len(files)}"
        )
    return files[0]


def _world_from_sweep_name(csv_path: Path) -> str:
    name = csv_path.name
    if name.endswith("_sweep.csv"):
        return name[: -len("_sweep.csv")]
    return csv_path.stem


def _record_label(r: dict) -> str:
    s, u = r["solver"], r["use_f64"]
    return f"{s}{'64' if u else '32'}_tol{r['solver_tol']:.0e}_k{r['target_kappa']:.0e}"


def load_records(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        rows: list[dict] = []
        for row in rdr:
            u64 = row.get("use_f64")
            use_f64 = bool(int(u64)) if u64 is not None and u64 != "" else False
            rows.append(
                {
                    "solver": row["solver"],
                    "use_f64": use_f64,
                    "solver_tol": float(row["solver_tol"]),
                    "target_kappa": float(row["target_kappa"]),
                    "T_total": float(row["T_total"]),
                    "best_cost": float(row["best_cost"]),
                    "wall_time": float(row["wall_time"]),
                    "n_iters": int(row["n_iters"]),
                    "converged": bool(int(row["converged"])),
                    "ls_failed": bool(int(row["ls_failed"])),
                }
            )
        return rows


# --------------------------------- Main ------------------------------------ #


def main() -> None:
    csv_path = _resolve_sweep_csv()
    world = _world_from_sweep_name(csv_path)
    out_dir = csv_path.parent
    records = load_records(csv_path)

    print(f"=== visualization | world={world} ===")
    print(f"  csv     : {csv_path.resolve()}")
    print(f"  out dir : {out_dir.resolve()}")
    print(f"  records : {len(records)}")

    print("\n=== summary (from CSV) ===")
    w = max(8, max(len(f"{r['solver']}") for r in records))
    print(
        f"  {'solver'.ljust(w)}  {'tol':>9}  {'kappa':>9}  {'T':>10}  "
        f"{'iters':>6}  {'wall[s]':>8}  ok"
    )
    for r in records:
        ok = "y" if (r["converged"] and not r["ls_failed"]) else "n"
        t_str = "nan" if not np.isfinite(r["T_total"]) else f"{r['T_total']:10.4f}"
        wt = "nan" if not np.isfinite(r["wall_time"]) else f"{r['wall_time']:8.2f}"
        print(
            f"  {r['solver'].ljust(w)}  {r['solver_tol']:9.1e}  {r['target_kappa']:9.1e}  "
            f"{t_str:>10}  {r['n_iters']:>6d}  {wt:>8}  {ok}"
        )

    p1 = out_dir / f"{world}_kappa_vs_T.png"
    png_paths: list[Path] = []
    if SAVE_PLOTS:
        plot_sweep_kappa_vs_T(records, savepath=p1)
        plt.close()
        print(f"\nSaved sweep plot: {p1.resolve()}")
        png_paths.append(p1.resolve())
    else:
        plot_sweep_kappa_vs_T(records, savepath=None)
        plt.show()
        print("\nSweep plot not written (SAVE_PLOTS=False).")

    traj_dir = out_dir / "trajectories"
    if SAVE_TRAJECTORIES:
        if not SAVE_PLOTS:
            print(
                f"\nTrajectory PNGs not written (SAVE_PLOTS=False); "
                f"would use {traj_dir.resolve()} with SAVE_PLOTS=True."
            )
        else:
            traj_dir.mkdir(exist_ok=True)
            n_ok, n_fail = 0, 0
            for r in records:
                label = _record_label(r)
                cfg = solver_cfg(
                    solver=r["solver"],
                    use_f64=r["use_f64"],
                    solver_tol=r["solver_tol"],
                    target_kappa=r["target_kappa"],
                    regularize=False,
                )
                try:
                    res = run_trajopt(world, cfg, PROBLEM, label=label, verbose=False)
                except Exception as e:
                    print(f"  trajectory FAILED {label}: {e}")
                    n_fail += 1
                    continue
                dest = traj_dir / f"{label}.png"
                plot_trajectory(res, label, savepath=dest)
                plt.close()
                png_paths.append(dest.resolve())
                n_ok += 1
            print(
                f"\nSaved {n_ok} trajectory plot(s) under {traj_dir.resolve()}"
                + (f" ({n_fail} run(s) failed)" if n_fail else "")
            )
    else:
        print(
            f"\nTrajectory replot skipped (SAVE_TRAJECTORIES=False); "
            f"would write under {traj_dir.resolve()}"
        )

    print("\n=== saved PNG paths (click in terminal) ===")
    if png_paths:
        for p in png_paths:
            print(p)
    else:
        print("  (none — enable SAVE_PLOTS to write PNGs)")

    print("Done.")


if __name__ == "__main__":
    main()
