"""
shared_utils/evaluation.py
Script detection, LLM judge (GPT-4o), and evaluation helpers.
"""

import json
from typing import Dict, List, Optional


# ── Script Detection ─────────────────────────────────────────────────────────

SCRIPT_RANGES = {
    "Arab": [("\u0600", "\u06FF"), ("\u0750", "\u077F"), ("\uFB50", "\uFDFF"), ("\uFE70", "\uFEFF")],
    "Hans": [("\u4e00", "\u9fff"), ("\u3400", "\u4dbf")],
    "Jpan": [("\u3040", "\u309f"), ("\u30a0", "\u30ff"), ("\u4e00", "\u9fff")],
    "Deva": [("\u0900", "\u097f")],
    "Hang": [("\uac00", "\ud7af"), ("\u1100", "\u11ff"), ("\u3130", "\u318f")],
    "Cyrl": [("\u0400", "\u04ff")],
    "Grek": [("\u0370", "\u03ff"), ("\u1f00", "\u1fff")],
    "Ethi": [("\u1200", "\u137f"), ("\u1380", "\u139f"), ("\u2d80", "\u2ddf")],
    "Beng": [("\u0980", "\u09ff")],
    "Latn": [("A", "Z"), ("a", "z"), ("\u00c0", "\u024f")],
}


def detect_script(text: str) -> Dict[str, float]:
    """
    Detect script composition of text. Returns {script_name: fraction}.
    """
    if not text:
        return {}

    counts = {s: 0 for s in SCRIPT_RANGES}
    total = 0

    for char in text:
        for script, ranges in SCRIPT_RANGES.items():
            for lo, hi in ranges:
                if lo <= char <= hi:
                    counts[script] += 1
                    total += 1
                    break

    if total == 0:
        return {}
    return {s: c / total for s, c in counts.items() if c > 0}


def evaluate_script_match(
    text: str,
    expected_script: str,
    threshold: float = 0.3,
) -> Dict[str, object]:
    """
    Evaluate whether generated text is in the expected script.

    Returns dict with:
      script_detected: bool — is >=threshold of chars in expected script?
      script_fraction: float — fraction of chars in expected script
      dominant_script: str — the script with highest fraction
      all_scripts: dict — full script breakdown
    """
    scripts = detect_script(text)
    fraction = scripts.get(expected_script, 0.0)
    dominant = max(scripts, key=scripts.get) if scripts else "unknown"

    return {
        "script_detected": fraction >= threshold,
        "script_fraction": fraction,
        "dominant_script": dominant,
        "all_scripts": scripts,
    }


# ── GPT-4o LLM Judge ────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a generated answer reflects the culture of {country}.

Question: {question}
Expected answer (gold, from native informants): {gold_answer}
Generated answer: {generated_answer}

Rate on a 1-5 scale:
1 = Completely unrelated to the country's culture
2 = Vaguely relevant but incorrect
3 = Partially correct, some cultural elements present
4 = Mostly correct cultural answer
5 = Perfectly matches the country's cultural perspective

Also assess:
- script_correct: Is the answer written in {expected_language}? (yes/no/mixed)
- cultural_match: Does the content match {country}'s culture, not another? (yes/no/partial)
- coherent: Is the answer coherent and well-formed? (yes/no)

Return JSON only: {{"score": <int>, "script_correct": "<yes|no|mixed>", "cultural_match": "<yes|no|partial>", "coherent": "<yes|no>", "reasoning": "<brief>"}}"""


def judge_with_gpt4o(
    question: str,
    gold_answer: str,
    generated_answer: str,
    country: str,
    expected_language: str,
    client=None,
    model: str = "gpt-4o",
) -> Dict:
    """
    Use GPT-4o to judge whether a generated answer is culturally appropriate.

    Parameters
    ----------
    client : openai.OpenAI instance (create externally to reuse connection)
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    prompt = JUDGE_PROMPT.format(
        country=country,
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
        expected_language=expected_language,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=300,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "script_correct": "error",
            "cultural_match": "error",
            "coherent": "error",
            "reasoning": response.choices[0].message.content,
        }


def batch_judge(
    generations: List[Dict],
    client=None,
    model: str = "gpt-4o",
) -> List[Dict]:
    """
    Judge a list of generations. Each item must have keys:
    question, gold_answer, generated_text, country, expected_language, id.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    results = []
    for gen in generations:
        judgment = judge_with_gpt4o(
            question=gen["question"],
            gold_answer=gen["gold_answer"],
            generated_answer=gen["generated_text"],
            country=gen["country"],
            expected_language=gen["expected_language"],
            client=client,
            model=model,
        )
        judgment["generation_id"] = gen.get("id", "")
        results.append(judgment)
    return results


# ── Language-to-script mapping ───────────────────────────────────────────────

LANG_TO_SCRIPT = {
    "en": "Latn", "fr": "Latn", "de": "Latn", "es": "Latn",
    "ar": "Arab", "fa": "Arab",
    "zh": "Hans", "ja": "Jpan",
    "hi": "Deva",
    "ko": "Hang",
    "ru": "Cyrl",
    "el": "Grek",
    "am": "Ethi",
    "as": "Beng",
    "id": "Latn", "az": "Latn", "ha": "Latn", "su": "Latn",
}

LANG_TO_COUNTRY = {
    "ar": "Algeria", "zh": "China", "es": "Spain", "ko": "South_Korea",
    "en": "US", "el": "Greece", "fa": "Iran", "id": "Indonesia",
    "am": "Ethiopia", "az": "Azerbaijan", "ha": "Northern_Nigeria",
    "su": "West_Java", "as": "Assam",
}

COUNTRY_TO_BLEND_SPLIT = {
    "Algeria": "DZ", "China": "CN", "Spain": "ES", "Mexico": "MX",
    "South_Korea": "KR", "North_Korea": "KP", "UK": "GB", "US": "US",
    "Greece": "GR", "Iran": "IR", "Indonesia": "ID", "Ethiopia": "ET",
    "Azerbaijan": "AZ", "Assam": "AS", "Northern_Nigeria": "NG",
    "West_Java": "JB",
}
