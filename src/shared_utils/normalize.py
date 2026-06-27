from collections import defaultdict
import numpy as np

def fit_stats(X_train):
    mu = X_train.mean(0); std = X_train.std(0) + 1e-6
    return mu, std

def standardize(X, mu, std):
    return (X - mu) / std

def center(X, group_ids):
    X = np.asarray(X, dtype=float); out = X.copy()
    idx = defaultdict(list)
    for i, g in enumerate(group_ids):
        idx[g].append(i)
    for g, rows in idx.items():
        rows = np.array(rows); out[rows] = X[rows] - X[rows].mean(0)
    return out
