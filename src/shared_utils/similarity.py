import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import subspace_angles as _sa

def _center(X):
    return X - X.mean(0)

def linear_cka(X, Y):
    Xc, Yc = _center(X), _center(Y)
    hsic = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(hsic / den) if den > 1e-12 else 0.0

def svcca(X, Y, k=10):
    from numpy.linalg import svd
    Xc, Yc = _center(X), _center(Y)
    Ux = svd(Xc, full_matrices=False)[0][:, :k]; Uy = svd(Yc, full_matrices=False)[0][:, :k]
    s = svd(Ux.T @ Uy, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))

def procrustes_disparity(X, Y):
    _, _, disparity = procrustes(X, Y)
    return float(disparity)

def rdm(X, metric="cosine"):
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return 1 - Xn @ Xn.T
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(X, metric="euclidean"))

def subspace_angles(A, B):
    return np.degrees(_sa(A, B))
