"""
shared_utils/model.py
Model loading and layer detection for NNsight + Qwen3-8B.
"""

import functools
import inspect
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
            original_decorator = generic_mod.check_model_inputs

            def passthrough_decorator(func):
                """Replacement that skips the **kwargs signature check."""
                return func

            generic_mod.check_model_inputs = passthrough_decorator

            # Re-apply to Qwen3 classes that already used the decorator
            from transformers.models.qwen3 import modeling_qwen3
            # Force re-decoration by reloading the module
            import importlib
            importlib.reload(modeling_qwen3)
    except (ImportError, AttributeError, Exception):
        pass


# Apply patch at import time
_patch_check_model_inputs()


def get_model_and_tokenizer(cfg: dict):
    """Load model with NNsight wrapping. Returns (model, tokenizer)."""
    from nnsight import LanguageModel

    model_name = cfg["model"]["name"]
    dtype_str = cfg["model"].get("dtype", "float16")
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    device = cfg["model"]["device"]

    model = LanguageModel(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        dispatch=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_num_layers(model) -> int:
    """Detect number of transformer layers."""
    inner = model._model if hasattr(model, "_model") else model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return len(inner.model.layers)
    if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
        return len(inner.transformer.h)
    raise ValueError("Cannot detect number of layers.")


def get_layers_to_probe(cfg: dict, num_layers: int):
    """Return list of layer indices from config."""
    layer_spec = cfg["model"]["layers"]
    if layer_spec == "all":
        return list(range(num_layers))
    return [l for l in layer_spec if l < num_layers]
