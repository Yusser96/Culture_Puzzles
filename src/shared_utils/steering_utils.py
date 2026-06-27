"""
src/shared_utils/steering_utils.py
Activation-steering helpers for decoder-only models.
Ported from scripts/shared_utils/steering.py (steer_generate / add_direction logic).
"""

import gc
from typing import Optional

import numpy as np
import torch


def add_and_generate(
    handle,
    prompt: str,
    layer: int,
    vec: np.ndarray,
    alpha: float,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """
    Generate text from *prompt* with a steering vector added at *layer*.

    At every autoregressive step, `alpha * vec` is added to the residual
    stream at position [:, :, :] of the chosen layer before sampling.

    Parameters
    ----------
    handle        : DecoderHandle returned by models.load_decoder
    prompt        : str — input prompt
    layer         : int — transformer layer index to steer at
    vec           : np.ndarray, shape (hidden_dim,) — steering direction
    alpha         : float — steering strength (0 = no steering)
    max_new_tokens: int — maximum tokens to generate
    temperature   : float — sampling temperature (0 = greedy)

    Returns
    -------
    str — decoded generated text (excluding the prompt)
    """
    model = handle.model
    tokenizer = handle.tokenizer

    device = next(handle.model._model.parameters()).device
    dtype = next(handle.model._model.parameters()).dtype

    steer_tensor = torch.tensor(vec * alpha, dtype=dtype, device=device)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    generated_ids = inputs["input_ids"].to(device)

    for _step in range(max_new_tokens):
        with model.trace(generated_ids):
            if alpha != 0:
                hidden = model.model.layers[layer].output
                hidden[:, :, :] += steer_tensor.unsqueeze(0).unsqueeze(0)
            logits = model.lm_head.output.save()

        next_logits = logits[:, -1, :].float()
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    del logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text
