import re, unicodedata
from collections import Counter

def detect_script(text):
    c = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        tok = name.split(" ")[0]
        if tok in ("CJK", "IDEOGRAPHIC"):
            tok = "CJK"
        c[tok] += 1
    return c.most_common(1)[0][0] if c else "UNKNOWN"

def sentence_split(text):
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 20]

def content_token_offsets(tokenizer, text, answer=None):
    """Return a per-token bool mask (True=content). Marks special tokens and tokens
    overlapping the answer substring as False, using offset mapping."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]; specials = set(tokenizer.all_special_ids)
    ids = enc["input_ids"]
    ans_span = None
    if answer:
        i = text.find(answer)
        if i >= 0:
            ans_span = (i, i + len(answer))
    mask = []
    for tid, (a, b) in zip(ids, offsets):
        if tid in specials or (a == b == 0):
            mask.append(False); continue
        if ans_span and not (b <= ans_span[0] or a >= ans_span[1]):
            mask.append(False); continue
        mask.append(True)
    return mask
