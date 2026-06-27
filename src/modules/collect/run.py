"""
src/modules/collect/run.py
===========================
CLI entry-point for the collect sub-pipeline.

Usage:
    python -m src.modules.collect.run --what {puzzles,parallel,topics,sib200,all} \\
           [--config src/configs/config.yaml]
"""

import argparse
import os
import sys


def _default_config() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "configs", "config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the collect sub-pipeline.")
    parser.add_argument(
        "--what",
        choices=["puzzles", "parallel", "topics", "sib200", "all"],
        default="all",
        help="Which collector(s) to run (default: all).",
    )
    parser.add_argument(
        "--config",
        default=_default_config(),
        help="Path to config.yaml (default: src/configs/config.yaml).",
    )
    args = parser.parse_args()

    from src.shared_utils.io import load_config
    cfg = load_config(args.config)

    what = args.what
    run_all = what == "all"

    if run_all or what == "puzzles":
        from src.modules.collect import puzzles
        puzzles.collect(cfg)

    if run_all or what == "parallel":
        from src.modules.collect import parallel
        parallel.collect(cfg)

    if run_all or what == "topics":
        from src.modules.collect import topics
        topics.collect(cfg)

    if run_all or what == "sib200":
        from src.modules.collect import sib200
        sib200.collect(cfg)


if __name__ == "__main__":
    main()
