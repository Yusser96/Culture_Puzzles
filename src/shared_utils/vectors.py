from collections import defaultdict
import numpy as np

def diffmean(target, background, normalize=True):
    if target.size == 0 or background.size == 0:
        d = target.shape[-1] if target.ndim >= 2 else background.shape[-1]
        return np.zeros(d)
    v = target.mean(0) - background.mean(0)
    if normalize:
        n = np.linalg.norm(v)
        if n > 1e-10:
            v = v / n
    return v

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_matrix(vectors):
    labels = sorted(vectors); n = len(labels); m = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            m[i, j] = cosine(vectors[li], vectors[lj])
    return m, labels

def subspace_angle(a, b):
    c = np.clip(abs(cosine(a, b)), 0, 1)
    return float(np.degrees(np.arccos(c)))

def balanced_background(X, labels, target_label, groups, rng):
    """Rows where label != target_label, sampled to an equal count per distinct group."""
    idx_by_group = defaultdict(list)
    for i, (lab, g) in enumerate(zip(labels, groups)):
        if lab != target_label:
            idx_by_group[g].append(i)
    if not idx_by_group:
        return np.zeros((0, X.shape[1]))
    k = min(len(v) for v in idx_by_group.values())
    picked = []
    for g, idxs in idx_by_group.items():
        picked += list(rng.choice(idxs, size=k, replace=False))
    return X[np.array(sorted(picked))]
