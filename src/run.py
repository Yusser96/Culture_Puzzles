"""
src/run.py
==========
Unified CLI dispatcher for the 12-step isolated pipeline.

Usage:
    python -m src.run <step> [--config src/configs/config.yaml]

Steps:
    collect, metadata, extract, normalize, probes, directions, cross,
    flores-decomp, rep-similarity, steering, data-stats, report

Each step is dispatched to the corresponding module's run(cfg) function.
"""

import argparse
import os
import sys


def _default_config() -> str:
    """Return default config path."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "configs", "config.yaml")


# Build STEPS dict by importing each module's run function.
# This must happen at module load time so that the STEPS dict is available
# for import by the CLI and tests. Imports are done directly on the module
# objects, not calling anything at import time, to avoid model loading.

# Import modules (not calling run yet)
from src.modules.collect import run as collect_run
from src.modules.metadata import run as metadata_run
from src.modules.extract import run as extract_run
from src.modules.normalize import run as normalize_run
from src.modules.probes import run as probes_run
from src.modules.directions import run as directions_run
from src.modules.cross import run as cross_run
from src.modules.flores_decomp import run as flores_decomp_run
from src.modules.rep_similarity import run as rep_similarity_run
from src.modules.steering import run as steering_run
from src.modules.data_stats import run as data_stats_run
from src.modules.report import run as report_run

# Map step names to run functions.
STEPS = {
    "collect": collect_run.run,
    "metadata": metadata_run.run,
    "extract": extract_run.run,
    "normalize": normalize_run.run,
    "probes": probes_run.run,
    "directions": directions_run.run,
    "cross": cross_run.run,
    "flores-decomp": flores_decomp_run.run,
    "rep-similarity": rep_similarity_run.run,
    "steering": steering_run.run,
    "data-stats": data_stats_run.run,
    "report": report_run.run,
}


def main():
    """Parse arguments and dispatch to the appropriate step."""
    parser = argparse.ArgumentParser(
        description="Run a single step of the unified pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Steps:
  {', '.join(sorted(STEPS.keys()))}

Example:
  python -m src.run collect --config src/configs/config.yaml
  python -m src.run metadata --config src/configs/config.yaml
  python -m src.run extract --config src/configs/config.yaml
        """,
    )

    parser.add_argument(
        "step",
        choices=sorted(STEPS.keys()),
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--config",
        default=_default_config(),
        help=f"Path to config.yaml (default: {_default_config()}).",
    )

    args = parser.parse_args()

    # Load config
    from src.shared_utils.io import load_config
    cfg = load_config(args.config)

    # Dispatch to step
    step_func = STEPS[args.step]
    try:
        step_func(cfg)
    except Exception as e:
        print(f"Error running step '{args.step}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
