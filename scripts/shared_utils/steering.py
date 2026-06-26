"""
shared_utils/steering.py
Steering hooks: logit-based MCQ evaluation and autoregressive SAQ generation.
"""

import gc
from typing import Dict, List, Optional

import numpy as np
import torch


def steer_logits(
    model,
    tokenizer,
    prompt: str,
    steering_vector: np.ndarray,
    layer: int,
    alpha: float,
    target_token_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Run a single forward pass with steering and return logits/probabilities.

    Used for MCQ evaluation: inject steering vector, read out logits at
    the last position for specific tokens (e.g., A/B/C/D).

    Parameters
    ----------
    model : NNsight LanguageModel
    tokenizer : tokenizer
    prompt : str — full MCQ prompt
    steering_vector : np.ndarray — steering direction (hidden_dim,)
    layer : int — layer to steer at
    alpha : float — steering strength (0 = no steering)
    target_token_ids : list of token ids to extract logits for (e.g., tokens for A,B,C,D)

    Returns
    -------
    dict with:
      "logits": np.ndarray of shape (vocab_size,) or (len(target_token_ids),)
      "probs": np.ndarray — softmax probabilities for target tokens
      "full_logits": np.ndarray — all logits at last position
    """
    device = next(model._model.parameters()).device
    dtype = next(model._model.parameters()).dtype

    steer_tensor = torch.tensor(
        steering_vector * alpha, dtype=dtype, device=device
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with model.trace(input_ids, attention_mask=attention_mask):
        if alpha != 0:
            hidden = model.model.layers[layer].output
            hidden[:, :, :] += steer_tensor.unsqueeze(0).unsqueeze(0)
        logits_saved = model.lm_head.output.save()

    last_logits = logits_saved[0, -1, :].float().detach().cpu().numpy()

    if target_token_ids is not None:
        target_logits = last_logits[target_token_ids]
        # Clamp to avoid inf-inf=nan when steering at deep layers
        target_logits = np.clip(target_logits, -1e4, 1e4)
        exp_logits = np.exp(target_logits - target_logits.max())
        probs = exp_logits / exp_logits.sum()
        result = {"logits": target_logits, "probs": probs}
    else:
        result = {"full_logits": last_logits}
        exp_logits = np.exp(last_logits - last_logits.max())
        result["probs"] = exp_logits / exp_logits.sum()

    del logits_saved

    return result


def steer_logits_batch(
    model,
    tokenizer,
    prompts: List[str],
    steering_vector: np.ndarray,
    layer: int,
    alpha: float,
    target_token_ids: Optional[List[int]] = None,
    batch_size: int = 32,
) -> List[Dict[str, np.ndarray]]:
    """
    Batched version of steer_logits — processes multiple prompts sharing
    one steering vector in chunks of ``batch_size``.

    Returns a list of dicts (one per prompt), each with keys
    ``"logits"``, ``"probs"``, ``"full_logits"`` — identical schema to
    :func:`steer_logits`.
    """
    device = next(model._model.parameters()).device
    dtype = next(model._model.parameters()).dtype

    steer_tensor = torch.tensor(
        steering_vector * alpha, dtype=dtype, device=device
    )

    # Ensure left-padding so the last token is always at position -1.
    orig_side = tokenizer.padding_side
    orig_pad = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: List[Dict[str, np.ndarray]] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with model.trace(input_ids, attention_mask=attention_mask):
            if alpha != 0:
                hidden = model.model.layers[layer].output
                hidden[:, :, :] += steer_tensor.unsqueeze(0).unsqueeze(0)
            logits_saved = model.lm_head.output.save()

        # With left-padding, last real token is always at position -1.
        batch_last_logits = logits_saved[:, -1, :].float().detach().cpu().numpy()

        for i in range(len(chunk)):
            last_logits = batch_last_logits[i]
            if target_token_ids is not None:
                target_logits = last_logits[target_token_ids]
                # Clamp to avoid inf-inf=nan when steering at deep layers
                target_logits = np.clip(target_logits, -1e4, 1e4)
                exp_logits = np.exp(target_logits - target_logits.max())
                probs = exp_logits / exp_logits.sum()
                result = {"logits": target_logits, "probs": probs}
            else:
                result = {"full_logits": last_logits}
                exp_logits = np.exp(last_logits - last_logits.max())
                result["probs"] = exp_logits / exp_logits.sum()
            results.append(result)

        del logits_saved
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore tokenizer state.
    tokenizer.padding_side = orig_side
    tokenizer.pad_token_id = orig_pad

    return results


def steer_generate(
    model,
    tokenizer,
    prompt: str,
    steering_vector: np.ndarray,
    layer: int,
    alpha: float,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    num_generations: int = 1,
) -> List[str]:
    """
    Generate text from a prompt with steering vector applied at each step.

    Used for SAQ generation.
    """
    device = next(model._model.parameters()).device
    dtype = next(model._model.parameters()).dtype

    steer_tensor = torch.tensor(
        steering_vector * alpha, dtype=dtype, device=device
    )

    generations = []
    for _ in range(num_generations):
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

        # Decode only the generated portion
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generations.append(text)

        del logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generations


def activation_patch(
    model,
    tokenizer,
    clean_text: str,
    corrupt_text: str,
    layer: int,
    component: str,
    metric_fn,
    max_seq_len: int = 128,
) -> float:
    """
    Activation patching: run clean and corrupt, patch clean activation
    into corrupt forward pass at a specific component.

    Parameters
    ----------
    component : str — "residual", "attn", "mlp"
    metric_fn : callable(logits_tensor) -> float

    Returns
    -------
    Normalized effect: 0 = no recovery, 1 = full recovery.
    """
    device = next(model._model.parameters()).device

    clean_inputs = tokenizer(
        clean_text, return_tensors="pt", truncation=True,
        max_length=max_seq_len, padding="max_length",
    )
    corrupt_inputs = tokenizer(
        corrupt_text, return_tensors="pt", truncation=True,
        max_length=max_seq_len, padding="max_length",
    )

    clean_ids = clean_inputs["input_ids"].to(device)
    clean_mask = clean_inputs["attention_mask"].to(device)
    corrupt_ids = corrupt_inputs["input_ids"].to(device)
    corrupt_mask = corrupt_inputs["attention_mask"].to(device)

    # 1. Clean forward pass — save target component activation
    # Note: attn and residual outputs are tuples (hidden, ...), so save [0]
    with model.trace(clean_ids, attention_mask=clean_mask):
        if component == "residual":
            clean_act = model.model.layers[layer].output[0].save()
        elif component == "attn":
            clean_act = model.model.layers[layer].self_attn.output[0].save()
        elif component == "mlp":
            clean_act = model.model.layers[layer].mlp.output.save()
        else:
            raise ValueError(f"Unknown component: {component}")
        clean_logits = model.lm_head.output.save()

    clean_metric = metric_fn(clean_logits)

    # 2. Corrupt forward pass (no patching) — baseline
    with model.trace(corrupt_ids, attention_mask=corrupt_mask):
        corrupt_logits = model.lm_head.output.save()

    corrupt_metric = metric_fn(corrupt_logits)

    # 3. Corrupt forward pass with patching
    with model.trace(corrupt_ids, attention_mask=corrupt_mask):
        if component == "residual":
            model.model.layers[layer].output[0][:] = clean_act
        elif component == "attn":
            model.model.layers[layer].self_attn.output[0][:] = clean_act
        elif component == "mlp":
            model.model.layers[layer].mlp.output[:] = clean_act
        patched_logits = model.lm_head.output.save()

    patched_metric = metric_fn(patched_logits)

    denom = clean_metric - corrupt_metric
    if abs(denom) < 1e-10:
        return 0.0
    return float((patched_metric - corrupt_metric) / denom)
