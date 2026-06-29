#!/usr/bin/env python3
"""
data_stats.py
=============
Detailed statistics over all collected data types, framed by the research plan's
factors: topic, language, region, language_region, script, source, length.

Covers:
  - puzzles   (riddles per lang_region: original + translation + topic)
  - parallel  (FLORES, OPUS-100: aligned sentences per lang_region)
  - cultural  (Wikipedia per topic x lang_region)

Produces, under <out_dir> (default: <va_analysis_dir>/data_stats):
  CSVs:
    overview_by_source.csv          counts/length per source
    puzzles_by_region.csv           per-region riddle + topic + length stats
    puzzles_topic_by_region.csv     region x topic count matrix
    cultural_topic_by_region.csv    topic x region sentence-count matrix
    parallel_by_region.csv          flores/opus per-region counts + shared groups
    length_by_factor.csv            length distribution by language/script/source
    script_by_region.csv            detected script per region
    confounds.csv                   factor-balance / coverage diagnostics
  Plots (PNG):
    fig_counts_by_source.png, fig_riddles_per_region.png,
    fig_topic_coverage_puzzles.png, fig_topic_coverage_cultural.png,
    fig_length_hist_by_source.png, fig_length_by_script.png,
    fig_sentences_per_region_parallel.png

Token length uses the model tokenizer if available (cached), else word counts.

Usage:
    python data_stats.py [--config configs/riddles_config.yaml] [--out DIR] [--no-tokenizer]
"""

import argparse
import csv
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import load_config, load_sentences, setup_logging

logger = setup_logging("data_stats")


# -- script detection ----------------------------------------------------------

def detect_script(text: str) -> str:
    """Dominant Unicode script of the (letter) characters in text."""
    counts = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        token = name.split(" ")[0]  # e.g. "ARABIC", "LATIN", "CJK", "HANGUL"
        if token in ("CJK", "IDEOGRAPHIC"):
            token = "CJK"
        counts[token] += 1
    return counts.most_common(1)[0][0] if counts else "UNKNOWN"


# -- tokenizer (optional) ------------------------------------------------------

def load_tokenizer(cfg, enabled):
    if not enabled:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
        logger.info(f"Token lengths via tokenizer: {cfg['model']['name']}")
        return tok
    except Exception as e:
        logger.warning(f"Tokenizer unavailable ({e}); using word counts for 'tokens'.")
        return None


def length_stats(texts, tok):
    """Return dict of char/word/token length summaries for a list of strings."""
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    if tok is not None and texts:
        toks = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    else:
        toks = words
    def summ(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return dict(mean=0, median=0, p10=0, p90=0, min=0, max=0)
        return dict(mean=float(a.mean()), median=float(np.median(a)),
                    p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                    min=float(a.min()), max=float(a.max()))
    return {"n": len(texts), "char": summ(chars), "word": summ(words),
            "token": summ(toks), "_token_vals": toks}


def _registry_maps(cfg):
    base, scriptmap = {}, {}
    for r in cfg.get("lang_regions", []):
        base[r["key"]] = r.get("base_language", "UNKNOWN")
    return base


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -- collectors of stats per source --------------------------------------------

def stat_puzzles(cfg, out_dir, tok, base_lang):
    pdir = cfg["paths"]["va_puzzles_dir"]
    if not os.path.isdir(pdir):
        logger.warning("No puzzles dir; skipping."); return None
    regions = sorted(d for d in os.listdir(pdir)
                     if os.path.isdir(os.path.join(pdir, d)))
    per_region, topic_by_region, script_by_region = [], {}, {}
    all_orig, all_trans = [], []
    script_token_pairs = []   # (script, token_len) per riddle, for confound analysis
    topic_set = set()
    for r in regions:
        jpath = os.path.join(pdir, r, "riddles.jsonl")
        if not os.path.exists(jpath):
            continue
        rows = [json.loads(l) for l in open(jpath, encoding="utf-8") if l.strip()]
        orig = [x["riddle_original"] for x in rows if x.get("riddle_original")]
        trans = [x["riddle_translation"] for x in rows if x.get("riddle_translation")]
        topics = Counter(x.get("topic_key") or x.get("topic") or "UNKNOWN" for x in rows)
        topic_by_region[r] = topics
        topic_set.update(topics)
        script_by_region[r] = detect_script(" ".join(orig[:50]))
        ls = length_stats(orig, tok)
        script_token_pairs += [(script_by_region[r], v) for v in ls["_token_vals"]]
        all_orig += orig; all_trans += trans
        per_region.append([r, base_lang.get(r, "?"), script_by_region[r], len(rows),
                           len(orig), len(trans), len(topics),
                           round(ls["char"]["mean"], 1), round(ls["token"]["mean"], 1)])
    write_csv(os.path.join(out_dir, "puzzles_by_region.csv"),
              ["region", "base_language", "script", "n_riddles", "n_original",
               "n_translation", "n_topics", "orig_char_mean", "orig_token_mean"],
              per_region)
    # topic x region matrix
    topics_sorted = sorted(topic_set)
    mrows = [[r] + [topic_by_region.get(r, {}).get(t, 0) for t in topics_sorted]
             for r in regions if r in topic_by_region]
    write_csv(os.path.join(out_dir, "puzzles_topic_by_region.csv"),
              ["region"] + topics_sorted, mrows)
    # plots
    _bar(os.path.join(out_dir, "fig_riddles_per_region.png"),
         [p[0] for p in per_region], [p[3] for p in per_region],
         "Riddles per lang_region", "riddles")
    _heatmap(os.path.join(out_dir, "fig_topic_coverage_puzzles.png"),
             np.array(mrows, dtype=object)[:, 1:].astype(float) if mrows else np.zeros((1, 1)),
             [m[0] for m in mrows], topics_sorted, "Puzzles: topic x region counts")
    return {"source": "puzzles", "n_regions": len(per_region),
            "n_original": len(all_orig), "n_translation": len(all_trans),
            "topics": topics_sorted, "orig_len": length_stats(all_orig, tok),
            "script_by_region": script_by_region,
            "script_token_pairs": script_token_pairs,
            "topic_by_region": topic_by_region}


def stat_parallel(cfg, out_dir, tok, base_lang):
    pdir = cfg["paths"]["va_parallel_dir"]
    rows, summary = [], {}
    for ds in ("flores", "opus100"):
        ddir = os.path.join(pdir, ds)
        if not os.path.isdir(ddir):
            continue
        man = {}
        mpath = os.path.join(ddir, "manifest.json")
        if os.path.exists(mpath):
            man = json.load(open(mpath)).get("lang_regions", {})
        texts = []
        n_regions = 0
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".txt"):
                continue
            region = fn[:-4]
            sents = load_sentences(os.path.join(ddir, fn))
            texts += sents
            n_regions += 1
            shared = man.get(region, {}).get("shared_with", [])
            ls = length_stats(sents, tok)
            rows.append([ds, region, base_lang.get(region, "?"), len(sents),
                         round(ls["char"]["mean"], 1), round(ls["token"]["mean"], 1),
                         len(shared)])
        summary[ds] = {"n_regions": n_regions, "n_sentences": len(texts),
                       "len": length_stats(texts, tok)}
    write_csv(os.path.join(out_dir, "parallel_by_region.csv"),
              ["dataset", "region", "base_language", "n_sentences",
               "char_mean", "token_mean", "n_shared_with"], rows)
    if rows:
        _grouped_counts(os.path.join(out_dir, "fig_sentences_per_region_parallel.png"),
                        rows, "Parallel sentences per region")
    return summary


def stat_cultural(cfg, out_dir, tok, base_lang):
    cdir = cfg["paths"]["va_cultural_dir"]
    if not os.path.isdir(cdir):
        logger.warning("No cultural dir; skipping."); return None
    topics = sorted(d for d in os.listdir(cdir)
                    if os.path.isdir(os.path.join(cdir, d)))
    region_set = set()
    matrix = defaultdict(dict)
    all_texts = []
    for t in topics:
        tdir = os.path.join(cdir, t)
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith(".txt"):
                continue
            region = fn[:-4]; region_set.add(region)
            sents = load_sentences(os.path.join(tdir, fn))
            matrix[t][region] = len(sents)
            all_texts += sents
    regions = sorted(region_set)
    mrows = [[t] + [matrix[t].get(r, 0) for r in regions] for t in topics]
    write_csv(os.path.join(out_dir, "cultural_topic_by_region.csv"),
              ["topic"] + regions, mrows)
    if mrows:
        _heatmap(os.path.join(out_dir, "fig_topic_coverage_cultural.png"),
                 np.array(mrows, dtype=object)[:, 1:].astype(float),
                 topics, regions, "Cultural: topic x region sentence counts")
    return {"source": "cultural", "n_topics": len(topics),
            "n_regions": len(regions), "n_sentences": len(all_texts),
            "len": length_stats(all_texts, tok)}


# -- plotting helpers ----------------------------------------------------------

def _bar(path, labels, values, title, ylabel):
    plt.figure(figsize=(max(8, len(labels) * 0.22), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=5)
    plt.ylabel(ylabel); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def _heatmap(path, mat, rowlabels, collabels, title):
    plt.figure(figsize=(max(8, len(collabels) * 0.25), max(4, len(rowlabels) * 0.3)))
    plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.yticks(range(len(rowlabels)), rowlabels, fontsize=6)
    plt.xticks(range(len(collabels)), collabels, rotation=90, fontsize=5)
    plt.colorbar(label="count"); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def _grouped_counts(path, rows, title):
    by_ds = defaultdict(list)
    for ds, region, *_rest in rows:
        by_ds[ds].append(_rest[2])  # n_sentences at index after base_language
    plt.figure(figsize=(8, 5))
    for ds, vals in by_ds.items():
        plt.hist(vals, bins=20, alpha=0.5, label=ds)
    plt.xlabel("sentences per region"); plt.ylabel("regions"); plt.legend()
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def _length_hist(path, length_by_source):
    plt.figure(figsize=(9, 5))
    for src, vals in length_by_source.items():
        if vals:
            plt.hist(vals, bins=40, alpha=0.5, label=src, density=True)
    plt.xlabel("token length"); plt.ylabel("density"); plt.legend()
    plt.title("Length distribution by source"); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def _length_by_script(path, rows):
    by_script = defaultdict(list)
    for script, toklen in rows:
        by_script[script].append(toklen)
    scripts = sorted(by_script)
    plt.figure(figsize=(max(7, len(scripts) * 0.8), 5))
    plt.boxplot([by_script[s] for s in scripts], labels=scripts, showfliers=False)
    plt.xticks(rotation=90, fontsize=7); plt.ylabel("token length")
    plt.title("Length by script (puzzles original)"); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Data statistics across all sources")
    ap.add_argument("--config", default="configs/riddles_config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out or os.path.join(cfg["paths"]["va_analysis_dir"], "data_stats")
    os.makedirs(out_dir, exist_ok=True)
    base_lang = _registry_maps(cfg)
    tok = load_tokenizer(cfg, enabled=not args.no_tokenizer)

    logger.info("=== puzzles ===")
    pz = stat_puzzles(cfg, out_dir, tok, base_lang)
    logger.info("=== parallel ===")
    pa = stat_parallel(cfg, out_dir, tok, base_lang)
    logger.info("=== cultural ===")
    cu = stat_cultural(cfg, out_dir, tok, base_lang)

    # -- overview + cross-source length plots ----------------------------------
    overview = []
    length_by_source = {}
    if pz:
        overview.append(["puzzles_original", pz["n_regions"], pz["n_original"],
                         round(pz["orig_len"]["token"]["mean"], 1)])
        length_by_source["puzzles_original"] = pz["orig_len"]["_token_vals"]
    if pa:
        for ds, s in pa.items():
            overview.append([ds, s["n_regions"], s["n_sentences"],
                             round(s["len"]["token"]["mean"], 1)])
            length_by_source[ds] = s["len"]["_token_vals"]
    if cu:
        overview.append(["cultural", cu["n_regions"], cu["n_sentences"],
                         round(cu["len"]["token"]["mean"], 1)])
        length_by_source["cultural"] = cu["len"]["_token_vals"]
    write_csv(os.path.join(out_dir, "overview_by_source.csv"),
              ["source", "n_groups", "n_sentences", "token_len_mean"], overview)
    _bar(os.path.join(out_dir, "fig_counts_by_source.png"),
         [o[0] for o in overview], [o[2] for o in overview],
         "Sentence count by source", "sentences")
    _length_hist(os.path.join(out_dir, "fig_length_hist_by_source.png"), length_by_source)

    # script-by-region + length-by-script (puzzles)
    if pz:
        sbr = pz["script_by_region"]
        write_csv(os.path.join(out_dir, "script_by_region.csv"),
                  ["region", "base_language", "script"],
                  [[r, base_lang.get(r, "?"), s] for r, s in sorted(sbr.items())])
        if pz.get("script_token_pairs"):
            _length_by_script(os.path.join(out_dir, "fig_length_by_script.png"),
                              pz["script_token_pairs"])

    # -- confound diagnostics (central to the research plan) -------------------
    conf = []
    for src, vals in length_by_source.items():
        a = np.asarray(vals, dtype=float)
        if a.size:
            conf.append(["length_token_mean", src, round(float(a.mean()), 2)])
            conf.append(["length_token_std", src, round(float(a.std()), 2)])
    # same-language regions sharing a base language (parallel/cultural content is shared)
    lang_groups = Counter(base_lang.values())
    multi = {k: v for k, v in lang_groups.items() if v > 1}
    conf.append(["languages_with_multiple_regions", "registry", len(multi)])
    conf.append(["regions_in_multi_region_languages", "registry", sum(multi.values())])
    if pz:
        canon = {"politics", "sports", "arts", "history", "geography",
                 "kids_world", "national_heritage", "everyday_life"}
        tbr = pz["topic_by_region"]
        full = sum(1 for r, tc in tbr.items() if canon.issubset(set(tc)))
        conf.append(["puzzle_distinct_topic_labels", "puzzles", len(pz["topics"])])
        conf.append(["puzzle_regions_with_all_8_canonical", "puzzles", full])
        conf.append(["puzzle_n_scripts", "puzzles", len(set(pz["script_by_region"].values()))])
    write_csv(os.path.join(out_dir, "confounds.csv"), ["metric", "scope", "value"], conf)

    # console summary
    logger.info("\n========== DATA STATS SUMMARY ==========")
    for o in overview:
        logger.info(f"  {o[0]:<22} groups={o[1]:<4} sentences={o[2]:<7} token_len_mean={o[3]}")
    if pz:
        logger.info(f"  puzzles topics: {pz['topics']}")
        logger.info(f"  scripts present: {sorted(set(pz['script_by_region'].values()))}")
    logger.info(f"\nOutputs (CSVs + figures) -> {out_dir}")


if __name__ == "__main__":
    main()
