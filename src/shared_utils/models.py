"""
src/shared_utils/models.py
Decoder-only model loader using NNsight.
Ported from scripts/shared_utils/model.py, adapted for the src pipeline.
"""

import torch
from transformers import AutoTokenizer


def _patch_check_model_inputs():
    """
    Fix transformers >=4.57 compatibility with Qwen3.

    The @check_model_inputs decorator requires **kwargs in the forward
    signature, but Qwen3Model.forward lacks it. We neutralize the
    signature check in the decorator so it never raises TypeError.
    """
    try:
        import transformers.utils.generic as generic_mod
        if hasattr(generic_mod, "check_model_inputs"):
            def passthrough_decorator(func):
                """Replacement that skips the **kwargs signature check."""
                return func

            generic_mod.check_model_inputs = passthrough_decorator

            # Re-apply to Qwen3 classes that already used the decorator
            from transformers.models.qwen3 import modeling_qwen3
            import importlib
            importlib.reload(modeling_qwen3)
    except (ImportError, AttributeError, Exception):
        pass


# Apply patch at import time
_patch_check_model_inputs()


class DecoderHandle:
    """Small handle returned by load_decoder."""

    def __init__(self, model, tokenizer, num_layers: int, hidden_size: int, name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.name = name

    def __repr__(self):
        return (
            f"DecoderHandle(name={self.name!r}, "
            f"num_layers={self.num_layers}, hidden_size={self.hidden_size})"
        )


def _get_num_layers(model) -> int:
    """Detect number of transformer layers for a NNsight-wrapped model."""
    inner = model._model if hasattr(model, "_model") else model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return len(inner.model.layers)
    if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
        return len(inner.transformer.h)
    raise ValueError("Cannot detect number of layers.")


def _get_hidden_size(model) -> int:
    """Detect hidden dimension of a NNsight-wrapped model."""
    inner = model._model if hasattr(model, "_model") else model
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
    raise ValueError("Cannot detect hidden size from model config.")


def load_decoder(cfg: dict, name: str) -> DecoderHandle:
    """
    Load a decoder-only model by name using NNsight, returning a DecoderHandle.

    Parameters
    ----------
    cfg : dict
        Pipeline config. Reads cfg['model']['dtype'] and cfg['model']['device'].
    name : str
        HuggingFace model name/path to load (overrides any name in cfg).

    Returns
    -------
    DecoderHandle with .model, .tokenizer, .num_layers, .hidden_size, .name
    """
    from nnsight import LanguageModel

    dtype_str = cfg["model"].get("dtype", "float16")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float16)
    device = cfg["model"]["device"]

    model = LanguageModel(
        name,
        torch_dtype=dtype,
        device_map=device,
        dispatch=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = _get_num_layers(model)
    hidden_size = _get_hidden_size(model)

    return DecoderHandle(
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers,
        hidden_size=hidden_size,
        name=name,
    )
