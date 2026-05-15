"""End-to-end pipeline: data generation -> training -> validation -> figures."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_generation import main as gen_main
from train import main as train_main
from validate import main as val_main
from visualize import main as viz_main


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"YAML config path (default: {default_config})",
    )
    args = parser.parse_args()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else default_config
    )

    run = gen_main(config_path=config_path)
    train_main(run_dir=run, config_path=config_path)

    if config_path.resolve() == default_config.resolve() or config_path.name == "config_nominal_run.yaml":
        val_main(run_dir=run, config_path=config_path)
        viz_main(run_dir=run, config_path=config_path)
        print(f"\nAll done. Run id: {run.name}")
