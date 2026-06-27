"""
src/shared_utils/extraction.py
Multi-readout batched activation extraction for decoder-only models.
Ported from scripts/shared_utils/activation_extraction.py, generalized to
support readouts ⊂ {mean_content, last_content, embed}.

NNsight execution-order requirement: embed_tokens output must be captured
BEFORE the transformer blocks inside the trace context.
"""

import gc
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.shared_utils.text import content_token_offsets


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _pool(hidden: np.ndarray, mask: np.ndarray, readout: str) -> np.ndarray:
    """
    Pool a batch of hidden states using a content-token boolean mask.

    Parameters
    ----------
    hidden : np.ndarray, shape (B, S, D)
    mask   : np.ndarray, shape (B, S), dtype bool
    readout : one of "mean_content", "last_content", "embed"
              embed is treated identically to mean_content (masked mean).

    Returns
    -------
    np.ndarray, shape (B, D)
    """
    out = []
    for b in range(hidden.shape[0]):
        m = mask[b]
        if not m.any():
            m = np.ones_like(m, dtype=bool)
        if readout == "last_content":
            out.append(hidden[b][np.where(m)[0][-1]])
        else:  # mean_content / embed both mean-pool over content tokens
            out.append(hidden[b][m].mean(0))
    return np.stack(out)


def _get_embedding_module(model):
    """Return the embed_tokens module/envoy, with fallbacks."""
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


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract(
    handle,
    texts: List[str],
    layers: List[int],
    readouts: List[str],
    max_seq_len: int = 128,
    batch_size: int = 4,
    answers: Optional[List[Optional[str]]] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extract activations for each (readout, layer) pair.

    Parameters
    ----------
    handle    : DecoderHandle returned by models.load_decoder
    texts     : list of N input strings
    layers    : list of integer layer indices to probe
    readouts  : subset of {"mean_content", "last_content", "embed"}
    max_seq_len : tokeniser truncation length
    batch_size  : samples per NNsight trace call
    answers   : optional list of N answer substrings to exclude from pooling
                (passed to content_token_offsets). None means no exclusion.

    Returns
    -------
    {readout: {layer_key: ndarray(N, D)}}
    Layer key is the integer layer index for transformer layers, or "embed"
    for the embedding readout.
    """
    valid_readouts = {"mean_content", "last_content", "embed"}
    for r in readouts:
        if r not in valid_readouts:
            raise ValueError(f"Unknown readout {r!r}. Must be one of {valid_readouts}.")

    want_embed = "embed" in readouts
    embed_readouts = [r for r in readouts if r != "embed"]

    # Accumulator: {readout: {layer_key: list of (D,) arrays}}
    accum: Dict[str, Dict] = {r: {} for r in readouts}
    for r in readouts:
        if r == "embed":
            accum[r]["embed"] = []
        for li in layers:
            for r2 in readouts:
                if r2 != "embed":
                    accum[r2][li] = []
    # rebuild cleanly
    accum = {r: {} for r in readouts}
    for r in readouts:
        if r == "embed":
            accum[r]["embed"] = []
        else:
            for li in layers:
                accum[r][li] = []

    model = handle.model
    tokenizer = handle.tokenizer

    for batch_start in tqdm(
        range(0, len(texts), batch_size), desc="Extracting", leave=False
    ):
        batch_texts = texts[batch_start: batch_start + batch_size]
        batch_answers = (
            answers[batch_start: batch_start + batch_size]
            if answers is not None
            else [None] * len(batch_texts)
        )

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Build per-sample content masks (shape: B x S_padded)
        # We build masks token-by-token against the padded sequence.
        batch_content_masks = []
        for i, (text, ans) in enumerate(zip(batch_texts, batch_answers)):
            raw_mask = content_token_offsets(tokenizer, text, ans)
            seq_len = input_ids.shape[1]
            # raw_mask is for the unpadded encoding; pad/align to padded length
            if len(raw_mask) >= seq_len:
                aligned = raw_mask[:seq_len]
            else:
                # left-pad with False (padding tokens are not content)
                pad_len = seq_len - len(raw_mask)
                aligned = [False] * pad_len + list(raw_mask)
            batch_content_masks.append(aligned)
        content_mask_np = np.array(batch_content_masks, dtype=bool)  # (B, S)

        saved: Dict = {}
        with model.trace(input_ids, attention_mask=attention_mask):
            # IMPORTANT: embed_tokens runs before the transformer blocks.
            # NNsight requires accessing outputs in execution order, so
            # capture the embedding FIRST to avoid a stale-value bug.
            if want_embed:
                saved["embed"] = _get_embedding_module(model).output.save()
            for layer_idx in layers:
                hidden = model.model.layers[layer_idx].output
                saved[layer_idx] = hidden.save()

        # Convert and pool
        for key, tensor_val in saved.items():
            # Some NNsight versions return tuples for layer outputs
            if isinstance(tensor_val, tuple):
                tensor_val = tensor_val[0]
            hidden_np = tensor_val.detach().float().cpu().numpy()  # (B, S, D)

            for r in readouts:
                if r == "embed" and key != "embed":
                    continue
                if r != "embed" and key == "embed":
                    continue
                pooled = _pool(hidden_np, content_mask_np, r)  # (B, D)
                for b in range(pooled.shape[0]):
                    target_key = "embed" if key == "embed" else key
                    accum[r][target_key].append(pooled[b])

        del inputs, saved
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack lists into arrays
    result: Dict[str, Dict] = {}
    for r in readouts:
        result[r] = {}
        for lk, lst in accum[r].items():
            if lst:
                result[r][lk] = np.stack(lst, axis=0)
            else:
                result[r][lk] = np.zeros((0, 1))

    return result
