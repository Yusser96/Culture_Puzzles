"""
src/modules/metadata/run.py
============================
CLI entry-point for the metadata builder.

Usage:
    python -m src.modules.metadata.run [--config src/configs/config.yaml]
                                       [--tokenizer <hf-model-id>]
"""

import argparse
import os


def _default_config() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "configs", "config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified metadata table.")
    parser.add_argument(
        "--config",
        default=_default_config(),
        help="Path to config.yaml (default: src/configs/config.yaml).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace model ID to use for token counting (optional).",
    )
    args = parser.parse_args()

    from src.shared_utils.io import load_config
    from src.shared_utils.store import MetadataTable
    from src.modules.metadata.build import build_metadata

    cfg = load_config(args.config)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    df = build_metadata(cfg, tokenizer)
    out_path = cfg["paths"]["metadata"]
    MetadataTable.save(df, out_path)
    print(f"Saved {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
