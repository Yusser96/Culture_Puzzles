import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

_FACTOR = {"heldout_language": "language", "heldout_region": "region",
           "heldout_language_region": "language_region", "heldout_source": "source",
           "heldout_prompt": "prompt_template"}

def train_probe(X, y, kind, seed):
    y = np.asarray(y)
    if kind == "logistic":
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(X, y); return ("logistic", m)
    if kind == "svm":
        m = LinearSVC(random_state=seed).fit(X, y); return ("svm", m)
    if kind == "diffmean":
        classes = sorted(set(y)); normals = {}
        for c in classes:
            v = X[y == c].mean(0) - X[y != c].mean(0)
            n = np.linalg.norm(v); normals[c] = v / n if n > 1e-10 else v
        return ("diffmean", {"classes": classes, "normals": normals})
    raise ValueError(kind)

def _predict(fitted, X):
    kind, m = fitted
    if kind == "diffmean":
        classes = m["classes"]; S = np.stack([X @ m["normals"][c] for c in classes], 1)
        return np.array([classes[i] for i in S.argmax(1)])
    return m.predict(X)

def probe_score(fitted, X, y):
    y = np.asarray(y); pred = _predict(fitted, X)
    out = {"macro_f1": float(f1_score(y, pred, average="macro"))}
    kind, m = fitted
    try:
        classes = sorted(set(y))
        if kind == "logistic" and len(classes) == 2:
            out["auroc"] = float(roc_auc_score(y == classes[1], m.predict_proba(X)[:, 1]))
        else:
            out["auroc"] = None
    except Exception:
        out["auroc"] = None
    return out

def probe_normal(fitted, kind):
    k, m = fitted
    if k == "diffmean":
        cs = m["classes"]
        return m["normals"][cs[0]] if len(cs) == 2 else m["normals"]
    coef = m.coef_
    v = coef[0] if coef.shape[0] == 1 else coef
    if v.ndim == 1:
        n = np.linalg.norm(v); v = v / n if n > 1e-10 else v
    return v

def make_splits(meta_df, scheme, seed):
    rng = np.random.default_rng(seed); n = len(meta_df); idx = np.arange(n)
    if scheme == "random":
        rng.shuffle(idx); cut = int(0.8 * n)
        return [(idx[:cut], idx[cut:])]
    col = _FACTOR[scheme]; groups = sorted(meta_df[col].astype(str).unique())
    splits = []
    for g in groups:
        te = idx[meta_df[col].astype(str).values == g]; tr = idx[meta_df[col].astype(str).values != g]
        if len(te) and len(tr):
            splits.append((tr, te))
    return splits
