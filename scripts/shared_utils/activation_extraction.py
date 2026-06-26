"""
shared_utils/activation_extraction.py
Batched activation extraction from NNsight-wrapped models.
"""

import gc
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def get_embedding_module(model):
    """Return the token-embedding module/envoy (embed_tokens), with fallbacks."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    inner = getattr(model, "_model", None)
    if inner is not None and hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
        return inner.model.embed_tokens
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    raise RuntimeError(f"Cannot locate embedding module on {type(model)}")


def extract_activations_batch(
    model,
    tokenizer,
    sentences: List[str],
    layers: List[int],
    max_seq_len: int = 128,
    batch_size: int = 4,
    desc: str = "Batches",
    include_embedding: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Extract mean residual-stream activations for each layer.

    Returns {layer_idx: np.ndarray of shape (num_sentences, hidden_dim)}.
    """
    probe_keys = list(layers) + (["embed"] if include_embedding else [])
    layer_activations = {k: [] for k in probe_keys}

    for i in tqdm(range(0, len(sentences), batch_size), desc=desc, leave=False):
        batch = sentences[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        saved = {}
        with model.trace(input_ids, attention_mask=attention_mask):
            for layer_idx in layers:
                hidden = model.model.layers[layer_idx].output
                saved[layer_idx] = hidden.save()
            if include_embedding:
                saved["embed"] = get_embedding_module(model).output.save()

        attention_mask = attention_mask.unsqueeze(-1)
        for layer_idx in probe_keys:
            hidden = saved[layer_idx]
            mask = attention_mask.to(hidden.device)

            for b in range(hidden.shape[0]):
                sum_act = hidden[b] * mask[b]
                mean_act = sum_act.sum(axis=0) / int(mask[b].sum())
                layer_activations[layer_idx].append(
                    mean_act.detach().float().cpu().numpy()
                )

        del inputs, saved
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for l in probe_keys:
        if layer_activations[l]:
            layer_activations[l] = np.stack(layer_activations[l], axis=0)
        else:
            layer_activations[l] = np.zeros((0, 1))

    return layer_activations
