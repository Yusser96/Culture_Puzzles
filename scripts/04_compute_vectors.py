#!/usr/bin/env python3
"""
04_compute_vectors.py
=====================
Unified DiffMean vector builder. Discovers each data collection produced by the
three collectors and writes a vector set named by the data source ("data name
extension"), loading the model once.

Sources:
  - parallel/flores      -> per-lang_region language vectors      (keys lang_<region>)
  - parallel/opus100      -> per-lang_region language vectors      (keys lang_<region>)
  - puzzles/original      -> per-lang_region puzzle vectors        (keys puzzle_<region>)
  - puzzles/translation   -> per-lang_region puzzle vectors        (keys puzzle_<region>)
  - cultural              -> topic vectors (pooled across regions)  (keys topic_<topic>)
                            + culture vectors (topic x region)      (keys culture_<topic>_<region>)

Output:
  vector_analysis/results/vectors/
    <dataname>/ layer_XXX.npz + raw_means/ + metadata.json
    topics/     topic_layer_XXX.npz + culture_layer_XXX.npz + metadata.json
    metadata.json
  # backward-compatible mirrors for 05/06:
  vector_analysis/results/vectors/language_vectors/{flores,opus100}/...
  vector_analysis/results/vectors/topic_vectors/...

Method (DiffMean, unchanged): for each class C and layer l,
  v_C = normalize(mean(acts_C) - mean(acts_all_others)).

Usage:
    python 04_compute_vectors.py [--config configs/riddles_config.yaml]
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils.data import (
    load_config, ensure_dirs, setup_logging, load_sentences, save_json,
)
from shared_utils.vectors import diffmean_vector, save_vectors
from shared_utils.model import get_model_and_tokenizer, get_num_layers, get_layers_to_probe
from shared_utils.activation_extraction import extract_activations_batch
from shared_utils.embeddings import save_embedding_store, build_group_map

logger = setup_logging("compute_vectors")


# -- Loaders: each returns {key: [sentences]} ----------------------------------

def load_flat_source(data_dir: str) -> Dict[str, List[str]]:
    """flores/ or opus100/ : one <key>.txt per lang_region."""
    out = {}
    if not os.path.isdir(data_dir):
        return out
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".txt"):
            key = fn[:-4]
            sents = load_sentences(os.path.join(data_dir, fn))
            if sents:
                out[key] = sents
    return out


def load_puzzles_source(puzzles_dir: str, variant: str) -> Dict[str, List[str]]:
    """puzzles/<region>/<variant>.txt  (variant in {original, translation})."""
    out = {}
    if not os.path.isdir(puzzles_dir):
        return out
    for region in sorted(os.listdir(puzzles_dir)):
        rdir = os.path.join(puzzles_dir, region)
        if not os.path.isdir(rdir):
            continue
        path = os.path.join(rdir, f"{variant}.txt")
        if os.path.exists(path):
            sents = load_sentences(path)
            if sents:
                out[region] = sents
    return out


def load_cultural_source(cultural_dir: str) -> Dict[tuple, List[str]]:
    """cultural/<topic>/<region>.txt -> {(topic, region): sentences}."""
    out = {}
    if not os.path.isdir(cultural_dir):
        return out
    for topic in sorted(os.listdir(cultural_dir)):
        tdir = os.path.join(cultural_dir, topic)
        if not os.path.isdir(tdir):
            continue
        for fn in sorted(os.listdir(tdir)):
            if fn.endswith(".txt"):
                region = fn[:-4]
                sents = load_sentences(os.path.join(tdir, fn))
                if sents:
                    out[(topic, region)] = sents
    return out


# -- Activation extraction with content dedup ----------------------------------

def extract_for_keys(model, tokenizer, key_to_sents, layers, max_seq_len, batch_size,
                     include_embedding=False):
    """
    Extract per-key activations, computing identical corpora only once (shared
    FLORES/Wikipedia baselines replicate across same-language regions).
    Returns ({key: {layer: acts}}, shared_groups{rep_key: [keys]}).
    """
    by_hash = {}              # hash -> rep_key
    groups = defaultdict(list)
    for key in key_to_sents:
        h = hash(tuple(key_to_sents[key]))
        if h not in by_hash:
            by_hash[h] = key
        groups[by_hash[h]].append(key)

    acts_by_rep = {}
    for rep in groups:
        sents = key_to_sents[rep]
        logger.info(f"    extracting {rep} ({len(sents)} sents)"
                    + (f" [shared by {len(groups[rep])}]" if len(groups[rep]) > 1 else ""))
        acts_by_rep[rep] = extract_activations_batch(
            model, tokenizer, sents, layers, max_seq_len, batch_size,
            include_embedding=include_embedding,
        )

    acts = {}
    shared_groups = {}
    for rep, members in groups.items():
        for k in members:
            acts[k] = acts_by_rep[rep]
        if len(members) > 1:
            shared_groups[rep] = sorted(members)
    return acts, shared_groups


def _emb_layers(cfg, layers):
    depth = cfg.get("analysis", {}).get("depth_layers")
    chosen = layers if not depth else [l for l in depth if l in layers]
    return ["embed"] + list(chosen)


# -- DiffMean over a set of classes --------------------------------------------

def diffmean_set(acts_by_key, layers) -> Dict[int, Dict[str, np.ndarray]]:
    """{layer: {key: vec}} where vec = DiffMean(key vs all other keys)."""
    keys = sorted(acts_by_key.keys())
    vectors = {}
    for layer in layers:
        vectors[layer] = {}
        for target in keys:
            target_acts = acts_by_key[target][layer]
            others = [acts_by_key[k][layer] for k in keys if k != target]
            other_acts = np.concatenate(others, axis=0) if others else np.zeros((0, 1))
            vectors[layer][target] = diffmean_vector(target_acts, other_acts, normalize=True)
    return vectors


def save_vector_set(out_dir, vectors, acts_by_key, layers, key_prefix,
                    shared_groups, extra_meta=None, mirror_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw_means")
    os.makedirs(raw_dir, exist_ok=True)

    for layer in layers:
        layer_vecs = {f"{key_prefix}{k}": vectors[layer][k] for k in vectors[layer]}
        save_vectors(layer_vecs, os.path.join(out_dir, f"layer_{layer:03d}.npz"))
        if mirror_dir:
            os.makedirs(mirror_dir, exist_ok=True)
            save_vectors(layer_vecs, os.path.join(mirror_dir, f"layer_{layer:03d}.npz"))

    for k in acts_by_key:
        for layer in layers:
            np.save(os.path.join(raw_dir, f"{k}_layer_{layer:03d}.npy"),
                    acts_by_key[k][layer].mean(axis=0))

    meta = {
        "layers": layers,
        "keys": sorted(f"{key_prefix}{k}" for k in vectors[layers[0]]),
        "shared_groups": shared_groups,
    }
    if extra_meta:
        meta.update(extra_meta)
    save_json(meta, os.path.join(out_dir, "metadata.json"))
    if mirror_dir:
        save_json(meta, os.path.join(mirror_dir, "metadata.json"))
    return meta


# -- Per-source drivers --------------------------------------------------------

def process_flat(model, tok, cfg, layers, name, data_dir, key_prefix, out_base,
                 mirror_base=None, emb_base=None):
    key_to_sents = load_flat_source(data_dir)
    if len(key_to_sents) < 2:
        logger.warning(f"  [{name}] need >=2 classes, found {len(key_to_sents)}; skipping.")
        return None
    inc = cfg["model"].get("include_embedding_layer", False)
    acts, shared = extract_for_keys(
        model, tok, key_to_sents, layers,
        cfg["model"]["max_seq_len"], cfg["model"]["batch_size"], include_embedding=inc,
    )
    vectors = diffmean_set(acts, layers)
    mirror = os.path.join(mirror_base, name) if mirror_base else None
    meta = save_vector_set(os.path.join(out_base, name), vectors, acts, layers,
                           key_prefix, shared, {"source": name}, mirror)
    if inc and emb_base:
        gm = build_group_map(cfg, name, list(key_to_sents.keys()))
        save_embedding_store(os.path.join(emb_base, name), acts, _emb_layers(cfg, layers),
                             key_prefix, gm)
    return meta


def process_puzzles(model, tok, cfg, layers, variant, puzzles_dir, out_base, emb_base=None):
    name = f"puzzles_{variant}"
    key_to_sents = load_puzzles_source(puzzles_dir, variant)
    if len(key_to_sents) < 2:
        logger.warning(f"  [{name}] need >=2 regions, found {len(key_to_sents)}; skipping.")
        return None
    inc = cfg["model"].get("include_embedding_layer", False)
    acts, shared = extract_for_keys(
        model, tok, key_to_sents, layers,
        cfg["model"]["max_seq_len"], cfg["model"]["batch_size"], include_embedding=inc,
    )
    vectors = diffmean_set(acts, layers)
    meta = save_vector_set(os.path.join(out_base, name), vectors, acts, layers,
                           "puzzle_", shared, {"source": name})
    if inc and emb_base:
        gm = build_group_map(cfg, name, list(key_to_sents.keys()))
        save_embedding_store(os.path.join(emb_base, name), acts, _emb_layers(cfg, layers),
                             "puzzle_", gm)
    return meta


def process_topics(model, tok, cfg, layers, cultural_dir, out_base, mirror_dir, emb_base=None):
    """Topic vectors (pooled across regions) + culture vectors (topic x region)."""
    pair_to_sents = load_cultural_source(cultural_dir)
    if not pair_to_sents:
        logger.warning("  [topics] no cultural data found; skipping.")
        return None

    inc = cfg["model"].get("include_embedding_layer", False)
    acts, _ = extract_for_keys(
        model, tok, pair_to_sents, layers,
        cfg["model"]["max_seq_len"], cfg["model"]["batch_size"], include_embedding=inc,
    )
    topics = sorted({t for (t, _r) in acts})

    if inc and emb_base:
        emb_layers = _emb_layers(cfg, layers)
        # culture: each (topic, region) pair
        culture_acts = {f"{t}_{r}": acts[(t, r)] for (t, r) in acts}
        save_embedding_store(os.path.join(emb_base, "culture"), culture_acts, emb_layers,
                             "culture_", build_group_map(cfg, "culture", list(culture_acts)))
        # topics: pool all regions per topic, per layer
        topic_acts = {}
        for t in topics:
            per_layer = {}
            for L in emb_layers:
                parts = [acts[(tt, r)][L] for (tt, r) in acts if tt == t]
                if parts:
                    per_layer[L] = np.concatenate(parts, axis=0)
            if per_layer:
                topic_acts[t] = per_layer
        save_embedding_store(os.path.join(emb_base, "topics"), topic_acts, emb_layers,
                             "topic_", build_group_map(cfg, "topics", list(topic_acts)))
    out_dir = os.path.join(out_base, "topics")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mirror_dir, exist_ok=True)

    for layer in layers:
        # Topic vectors: pool all regions per topic, DiffMean across topics.
        pooled = {}
        for t in topics:
            parts = [acts[(tt, r)][layer] for (tt, r) in acts if tt == t]
            if parts:
                pooled[t] = np.concatenate(parts, axis=0)
        topic_vecs = {}
        for t in pooled:
            other = np.concatenate([pooled[o] for o in pooled if o != t], axis=0)
            topic_vecs[f"topic_{t}"] = diffmean_vector(pooled[t], other, normalize=True)

        # Culture vectors: each (topic, region) vs all other pairs.
        keys = sorted(acts.keys())
        culture_vecs = {}
        for (t, r) in keys:
            target = acts[(t, r)][layer]
            other = np.concatenate([acts[k][layer] for k in keys if k != (t, r)], axis=0)
            culture_vecs[f"culture_{t}_{r}"] = diffmean_vector(target, other, normalize=True)

        for d in (out_dir, mirror_dir):
            save_vectors(topic_vecs, os.path.join(d, f"topic_layer_{layer:03d}.npz"))
            save_vectors(culture_vecs, os.path.join(d, f"culture_layer_{layer:03d}.npz"))

    meta = {
        "source": "topics",
        "layers": layers,
        "topics": topics,
        "culture_keys": sorted(f"{t}_{r}" for (t, r) in acts),
    }
    save_json(meta, os.path.join(out_dir, "metadata.json"))
    save_json(meta, os.path.join(mirror_dir, "metadata.json"))
    return meta


def main():
    parser = argparse.ArgumentParser(description="Compute vectors for all data sources")
    parser.add_argument("--config", default="configs/riddles_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    parallel_dir = cfg["paths"]["va_parallel_dir"]
    puzzles_dir = cfg["paths"]["va_puzzles_dir"]
    cultural_dir = cfg["paths"]["va_cultural_dir"]
    vec_dir = cfg["paths"]["va_vector_dir"]
    lang_mirror = cfg["paths"]["va_language_vector_dir"]
    topic_mirror = cfg["paths"]["va_topic_vector_dir"]
    emb_dir = cfg["paths"]["va_embeddings_dir"]

    model, tokenizer = get_model_and_tokenizer(cfg)
    num_layers = get_num_layers(model)
    layers = get_layers_to_probe(cfg, num_layers)
    logger.info(f"Model has {num_layers} layers. Probing: {layers}")

    all_meta = {"layers": layers, "sources": []}

    def record(name, meta):
        if meta:
            all_meta["sources"].append(name)
            all_meta[name] = meta

    logger.info("Source: parallel/flores")
    record("flores", process_flat(model, tokenizer, cfg, layers, "flores",
            os.path.join(parallel_dir, "flores"), "lang_", vec_dir, lang_mirror,
            emb_base=emb_dir))

    logger.info("Source: parallel/opus100")
    record("opus100", process_flat(model, tokenizer, cfg, layers, "opus100",
            os.path.join(parallel_dir, "opus100"), "lang_", vec_dir, lang_mirror,
            emb_base=emb_dir))

    logger.info("Source: puzzles/original")
    record("puzzles_original", process_puzzles(model, tokenizer, cfg, layers,
            "original", puzzles_dir, vec_dir, emb_base=emb_dir))

    logger.info("Source: puzzles/translation")
    record("puzzles_translation", process_puzzles(model, tokenizer, cfg, layers,
            "translation", puzzles_dir, vec_dir, emb_base=emb_dir))

    logger.info("Source: cultural/topics")
    record("topics", process_topics(model, tokenizer, cfg, layers,
            cultural_dir, vec_dir, topic_mirror, emb_base=emb_dir))

    save_json(all_meta, os.path.join(vec_dir, "metadata.json"))
    logger.info(f"\nVector computation complete. Sources: {all_meta['sources']}")


if __name__ == "__main__":
    main()
