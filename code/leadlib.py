"""Torch-free helpers for the lead-importance analysis (Phase 2A). Tested in test_leads.py."""
import numpy as np
from scipy.stats import rankdata

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]   # PTB-XL channel order


def fast_auroc(y, p):
    """Binary AUROC from ranks (ties get average rank). Raises if a class is absent."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC undefined: only one class present")
    r = rankdata(p)
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def class_aurocs(Y, P):
    return np.array([fast_auroc(Y[:, c], P[:, c]) for c in range(Y.shape[1])])


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def mean_pairwise_spearman(V):
    """V: (n_runs, n_items). Mean Spearman over all pairs of runs."""
    n = len(V)
    return float(np.nanmean([spearman(V[i], V[j]) for i in range(n) for j in range(i + 1, n)]))


def topk_overlap(V, k=3):
    """Mean Jaccard overlap of the top-k item sets over all pairs of runs."""
    sets = [set(np.argsort(-v)[:k].tolist()) for v in V]
    n = len(sets)
    return float(np.mean([len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                          for i in range(n) for j in range(i + 1, n)]))


def spearman_null(n_items=12, n_runs=5, n_sim=5000, seed=0):
    """Null distribution of the mean pairwise Spearman between runs with random rankings."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sim)
    for s in range(n_sim):
        V = np.array([rng.permutation(n_items) for _ in range(n_runs)], float)
        out[s] = mean_pairwise_spearman(V)
    return out


def importance_from_preds(Y, P_base, P_zero):
    """Per-class AUROC drop when each lead is removed. P_zero: (12, N, C). Returns (12, C+1); last column = macro."""
    base = class_aurocs(Y, P_base)
    drop = np.stack([base - class_aurocs(Y, P_zero[l]) for l in range(P_zero.shape[0])])
    return np.concatenate([drop, drop.mean(1, keepdims=True)], axis=1)


def random_orders(n_orders, n_leads=12, seed=2026):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_leads) for _ in range(n_orders)]


def zero_leads(X, leads):
    """Copy of X with the given leads set to 0 (= the train-set mean after z-scoring)."""
    Xp = X.copy()
    Xp[:, list(leads), :] = 0.0
    return Xp


def permute_lead(X, lead, rng):
    """Copy of X where one lead is taken from other records (keeps its marginal distribution)."""
    Xp = X.copy()
    Xp[:, lead, :] = X[rng.permutation(len(X)), lead, :]
    return Xp
