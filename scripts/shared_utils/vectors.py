"""
shared_utils/vectors.py
Vector math: DiffMean, cosine similarity, subspace angles, I/O.
"""

import numpy as np
from typing import Dict, List, Tuple


def diffmean_vector(
    target_activations: np.ndarray,
    other_activations: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    DiffMean steering vector (AxBench style).
    v = normalize(mean(target) - mean(others))
    """
    if target_activations.size == 0 or other_activations.size == 0:
        dim = (
            target_activations.shape[-1]
            if target_activations.ndim >= 2
            else other_activations.shape[-1]
            if other_activations.ndim >= 2
            else 1
        )
        return np.zeros(dim, dtype=np.float64)

    mu_target = target_activations.mean(axis=0)
    mu_other = other_activations.mean(axis=0)
    vec = mu_target - mu_other
    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    vectors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(vectors.keys())
    n = len(labels)
    mat = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            mat[i, j] = cosine_similarity(vectors[li], vectors[lj])
    return mat, labels


def subspace_angle(a: np.ndarray, b: np.ndarray) -> float:
    cos_val = cosine_similarity(a, b)
    if cos_val == 0.0 and (
        not np.isfinite(a).all() or not np.isfinite(b).all()
    ):
        return 90.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_val))))


def save_vectors(vectors: dict, path: str) -> None:
    np.savez(path, **vectors)


def load_vectors(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def pairwise_distance_matrix(
    vectors: Dict[str, np.ndarray],
    metric: str = "cosine",
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute NxN pairwise distance matrix from named vectors.
    metric: "cosine" (1 - cos_sim) or "euclidean".
    Returns (distance_matrix, sorted_labels).
    """
    labels = sorted(vectors.keys())
    n = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "cosine":
                d = 1.0 - cosine_similarity(vectors[labels[i]], vectors[labels[j]])
            elif metric == "euclidean":
                d = float(np.linalg.norm(vectors[labels[i]] - vectors[labels[j]]))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            mat[i, j] = d
            mat[j, i] = d
    return mat, labels


def project_out_direction(
    vectors: Dict[str, np.ndarray],
    direction: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Project out a direction from all vectors: v' = v - (v . d_hat) * d_hat.
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return dict(vectors)
    d_hat = direction / norm
    result = {}
    for k, v in vectors.items():
        proj = np.dot(v, d_hat) * d_hat
        result[k] = v - proj
    return result
