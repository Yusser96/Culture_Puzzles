"""
shared_utils/clustering.py
Clustering utilities: hierarchical clustering, Mantel test, agreement scores.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def hierarchical_cluster(
    distance_matrix: np.ndarray,
    labels: List[str],
    method: str = "ward",
    n_clusters: Optional[int] = None,
) -> Dict:
    """
    Hierarchical clustering from a distance matrix.

    Returns dict with keys:
      - linkage_matrix: scipy linkage matrix
      - labels: input labels
      - cluster_labels: cluster assignment (if n_clusters given)
      - dendrogram_order: leaf order for plotting
    """
    n = distance_matrix.shape[0]
    # Ensure symmetric, zero diagonal
    dm = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(dm, 0)
    # Clamp negatives from float rounding
    dm = np.clip(dm, 0, None)

    condensed = squareform(dm, checks=False)
    Z = linkage(condensed, method=method)

    result = {
        "linkage_matrix": Z.tolist(),
        "labels": labels,
        "dendrogram_order": leaves_list(Z).tolist(),
    }

    if n_clusters is not None:
        result["cluster_labels"] = fcluster(Z, n_clusters, criterion="maxclust").tolist()

    return result


def mantel_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Mantel test: permutation-based correlation between two distance matrices.

    Returns {"r": Pearson r, "p_value": p-value}.
    """
    n = dist_a.shape[0]
    # Extract upper triangles
    idx = np.triu_indices(n, k=1)
    va = dist_a[idx]
    vb = dist_b[idx]

    # Handle constant vectors
    if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
        return {"r": 0.0, "p_value": 1.0}

    observed_r = float(np.corrcoef(va, vb)[0, 1])

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        dist_b_perm = dist_b[np.ix_(perm, perm)]
        vb_perm = dist_b_perm[idx]
        perm_r = np.corrcoef(va, vb_perm)[0, 1]
        if perm_r >= observed_r:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return {"r": observed_r, "p_value": p_value}


def cluster_agreement_scores(
    predicted_labels: List,
    true_labels: List,
) -> Dict[str, float]:
    """Compute ARI and NMI between two label assignments."""
    return {
        "ari": float(adjusted_rand_score(true_labels, predicted_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted_labels)),
    }
