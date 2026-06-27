"""
src/modules/report/run.py
==========================
§15 success-criteria checker + final-report aggregation.

Public API
----------
success_criteria(record) -> dict[str, bool]
    Pure function (unit-tested, no I/O).  Evaluates the §15 checklist
    against a single direction/probe result record (dict with numeric fields).

run(cfg) -> pd.DataFrame
    Load the analysis CSVs produced by the upstream modules, aggregate per
    candidate direction, apply success_criteria to each aggregated row, and
    write ``report_summary.csv`` to ``cfg['paths']['analysis_dir']``.
    Returns the summary DataFrame.
    Requires the analysis CSVs to exist; smoke-validated by CI.

Success-criteria thresholds
---------------------------
decodable:
    If ``record['decodable_threshold']`` is present → probe_macro_f1 > that
    threshold.  Else → probe_macro_f1 > 0.6.

persists_after_controls:
    macro_f1_centered > 0.55  (above chance even after centering).

transfers:
    heldout_macro_f1 > record.get('transfer_threshold', 0.5).

layer_stable:
    n_stable_layers >= 2.

coherent:
    mean_contrast_cosine > 0.3.

not_confounded:
    NOT (script_macro_f1 > 0.8 AND topic_macro_f1 < script_macro_f1).
    i.e. topic decoding accuracy must not be dominated entirely by script.

causal:
    steering_effect > 0.
"""

import logging
import os
from typing import Dict

import pandas as pd

from src.shared_utils.io import ensure_dir, save_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Success-criteria (pure, unit-tested)
# ---------------------------------------------------------------------------

def success_criteria(record: dict) -> Dict[str, bool]:
    """
    Evaluate the §15 checklist for a single direction/probe result record.

    Parameters
    ----------
    record : dict
        Must contain numeric fields (see module docstring for full list).

    Returns
    -------
    dict[str, bool]
        Seven boolean flags: decodable, persists_after_controls, transfers,
        layer_stable, coherent, not_confounded, causal.
    """
    # ---- decodable ----------------------------------------------------------
    # Use record-level threshold if provided; default: macro_f1 > 0.6
    if "decodable_threshold" in record:
        decodable = float(record["probe_macro_f1"]) > float(record["decodable_threshold"])
    else:
        chance = float(record.get("chance", 0.5))
        # margin above chance; use a fixed 0.1 above-chance margin → 0.6 for 0.5 chance
        decodable = float(record["probe_macro_f1"]) > chance + 0.1

    # ---- persists_after_controls --------------------------------------------
    # Centered representation macro_f1 must still be above a meaningful threshold
    persists_after_controls = float(record["macro_f1_centered"]) > 0.55

    # ---- transfers ----------------------------------------------------------
    transfer_threshold = float(record.get("transfer_threshold", 0.5))
    transfers = float(record["heldout_macro_f1"]) > transfer_threshold

    # ---- layer_stable -------------------------------------------------------
    layer_stable = int(record["n_stable_layers"]) >= 2

    # ---- coherent -----------------------------------------------------------
    coherent = float(record["mean_contrast_cosine"]) > 0.3

    # ---- not_confounded -----------------------------------------------------
    # Topic representation should not be dominated by script alone.
    # Confounded if: script probe beats 0.8 F1 AND topic F1 < script F1.
    script_f1 = float(record["script_macro_f1"])
    topic_f1 = float(record["topic_macro_f1"])
    not_confounded = not (script_f1 > 0.8 and topic_f1 < script_f1)

    # ---- causal -------------------------------------------------------------
    causal = float(record.get("steering_effect", 0)) > 0.0

    return {
        "decodable": decodable,
        "persists_after_controls": persists_after_controls,
        "transfers": transfers,
        "layer_stable": layer_stable,
        "coherent": coherent,
        "not_confounded": not_confounded,
        "causal": causal,
    }


# ---------------------------------------------------------------------------
# Report runner
# ---------------------------------------------------------------------------

# CSV filenames produced by upstream modules
_PROBE_CSV = "layer_probe_scores.csv"
_TRANSFER_CSV = "transfer_scores.csv"
_DIRECTIONS_CSV = "topic_vector_cosines.csv"
_STEERING_CSV = "steering_results.csv"


def _safe_load(analysis_dir: str, fname: str) -> pd.DataFrame:
    """Load a CSV from analysis_dir; return an empty DataFrame on missing file."""
    path = os.path.join(analysis_dir, fname)
    if not os.path.exists(path):
        logger.warning("Missing analysis file: %s — skipping.", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def run(cfg: dict) -> pd.DataFrame:
    """
    Aggregate analysis CSVs into ``report_summary.csv``.

    Loads the probe scores, transfer scores, direction cosines, and steering
    results.  For each candidate direction (model × topic combination) it:
      1. Picks the best-layer probe macro_f1.
      2. Picks the heldout transfer macro_f1 for that direction.
      3. Picks the mean direction cosine (coherence).
      4. Picks the mean steering effect (if any).
      5. Calls ``success_criteria`` on the aggregated record.
      6. Writes ``report_summary.csv`` to ``analysis_dir``.

    Parameters
    ----------
    cfg : dict
        Must contain ``cfg['paths']['analysis_dir']``.

    Returns
    -------
    pd.DataFrame
        The report_summary DataFrame (also written to CSV).
    """
    analysis_dir: str = cfg["paths"]["analysis_dir"]
    ensure_dir(analysis_dir)

    probes = _safe_load(analysis_dir, _PROBE_CSV)
    transfer = _safe_load(analysis_dir, _TRANSFER_CSV)
    directions = _safe_load(analysis_dir, _DIRECTIONS_CSV)
    steering = _safe_load(analysis_dir, _STEERING_CSV)

    # Discover candidate directions: (model, topic) pairs from probe scores
    if probes.empty:
        logger.warning("No probe scores found; report_summary will be empty.")
        candidates = []
    else:
        # topic-canonical probe rows only
        topic_probes = probes[probes.get("factor", pd.Series(dtype=str)) == "topic_canonical"] \
            if "factor" in probes.columns else probes
        candidates = (
            topic_probes[["model", "factor"]].drop_duplicates().values.tolist()
            if "model" in topic_probes.columns and "factor" in topic_probes.columns
            else []
        )

    rows = []
    figures = _collect_figures(analysis_dir)

    for model, factor in candidates:
        topic = factor  # repurpose for non-topic factors too

        # ---- probe macro_f1 (best layer, raw representation, random split) ----
        mask = (
            (probes["model"] == model)
            & (probes.get("factor", pd.Series(dtype=str)) == factor)
        )
        if "representation" in probes.columns:
            mask &= probes["representation"] == "raw"
        if "split" in probes.columns:
            mask &= probes["split"] == "random"
        probe_slice = probes[mask]
        probe_macro_f1 = float(probe_slice["macro_f1"].max()) if len(probe_slice) else 0.0

        # Count how many layers show macro_f1 > 0.6 (layer stability proxy)
        n_stable_layers = int((probe_slice["macro_f1"] > 0.6).sum()) if len(probe_slice) else 0

        # ---- centered representation macro_f1 --------------------------------
        macro_f1_centered = 0.0
        if "representation" in probes.columns:
            mask_c = (
                (probes["model"] == model)
                & (probes.get("factor", pd.Series(dtype=str)) == factor)
                & (probes["representation"].str.contains("centered", na=False))
                & (probes.get("split", pd.Series(dtype=str)) == "random")
            )
            c_slice = probes[mask_c]
            if len(c_slice):
                macro_f1_centered = float(c_slice["macro_f1"].max())

        # ---- script / topic confound probes ----------------------------------
        script_macro_f1 = 0.0
        topic_macro_f1 = probe_macro_f1  # alias
        if "factor" in probes.columns:
            for _factor, _col in [("script", "script_macro_f1"), ("topic_canonical", "topic_macro_f1")]:
                mask_f = (probes["model"] == model) & (probes["factor"] == _factor)
                if "representation" in probes.columns:
                    mask_f &= probes["representation"] == "raw"
                if "split" in probes.columns:
                    mask_f &= probes["split"] == "random"
                fslice = probes[mask_f]
                val = float(fslice["macro_f1"].max()) if len(fslice) else 0.0
                if _factor == "script":
                    script_macro_f1 = val
                else:
                    topic_macro_f1 = val

        # ---- transfer (heldout split) ----------------------------------------
        heldout_macro_f1 = 0.0
        if not transfer.empty and "factor" in transfer.columns:
            t_mask = (transfer["model"] == model) & (transfer["factor"] == factor)
            t_slice = transfer[t_mask]
            if len(t_slice):
                heldout_macro_f1 = float(t_slice["macro_f1"].max())

        # ---- direction coherence (mean cosine from directions module) ---------
        mean_contrast_cosine = 0.0
        if not directions.empty and "topic" in directions.columns:
            d_mask = (directions["topic"] == factor) & (directions["language"] == "ALL")
            if "model" in directions.columns:
                d_mask &= directions["model"] == model
            d_slice = directions[d_mask]
            if len(d_slice):
                # cos_diffmean_logistic as coherence proxy
                col = "cos_diffmean_logistic"
                if col in d_slice.columns:
                    mean_contrast_cosine = float(d_slice[col].mean())

        # ---- steering effect -------------------------------------------------
        steering_effect = 0.0
        if not steering.empty and "topic" in steering.columns:
            s_mask = (steering["topic"] == factor)
            if "model" in steering.columns:
                s_mask &= steering["model"] == model
            s_slice = steering[s_mask]
            if len(s_slice):
                # proxy: mean probe_margin (>0 means positive steering effect)
                col = "probe_margin"
                if col in s_slice.columns:
                    steering_effect = float(s_slice[col].mean())

        record = {
            "probe_macro_f1": probe_macro_f1,
            "macro_f1_centered": macro_f1_centered,
            "heldout_macro_f1": heldout_macro_f1,
            "n_stable_layers": n_stable_layers,
            "mean_contrast_cosine": mean_contrast_cosine,
            "script_macro_f1": script_macro_f1,
            "topic_macro_f1": topic_macro_f1,
            "steering_effect": steering_effect,
        }
        criteria = success_criteria(record)

        rows.append({
            "model": model,
            "topic": factor,
            **record,
            **criteria,
            "n_criteria_met": sum(criteria.values()),
            "figures": "; ".join(figures),
        })

    summary_df = pd.DataFrame(rows)

    out_path = os.path.join(analysis_dir, "report_summary.csv")
    if summary_df.empty:
        # Write header-only CSV
        save_csv(out_path, ["model", "topic", "n_criteria_met"], [])
    else:
        summary_df.to_csv(out_path, index=False)
    logger.info("Wrote report_summary.csv (%d rows) -> %s", len(summary_df), out_path)
    logger.info("Figures referenced: %s", figures)
    return summary_df


def _collect_figures(analysis_dir: str):
    """Return a sorted list of PNG figure paths found under analysis_dir."""
    figs = []
    for root, _dirs, files in os.walk(analysis_dir):
        for fn in sorted(files):
            if fn.endswith(".png"):
                figs.append(os.path.join(root, fn))
    return figs


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.shared_utils.io import load_config, setup_logging

    setup_logging(__name__)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/configs/config.yaml"
    run(load_config(cfg_path))
