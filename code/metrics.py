"""Metrics and threshold selection (no torch dependency)."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             matthews_corrcoef, roc_auc_score)

from config import CLASSES


def _need_both_classes(y, name=""):
    """sklearn returns NaN (with a warning) in recent versions when a class has no
    positives or no negatives; we refuse instead of letting NaN propagate."""
    s = float(np.sum(y))
    if s == 0 or s == len(y):
        raise ValueError(f"class {name!r} has no positives or no negatives")


def macro_auroc(Y, P):
    vals = []
    for c in range(Y.shape[1]):
        _need_both_classes(Y[:, c], c)
        vals.append(roc_auc_score(Y[:, c], P[:, c]))
    return float(np.mean(vals))


def _f1(y, pred):
    tp = np.sum((y == 1) & (pred == 1))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def choose_thresholds(Y, P, grid):
    """Per-class F1-optimal threshold. Call with VALIDATION data only."""
    lo, hi, step = grid
    cand = np.arange(lo, hi + 1e-9, step)
    thr = np.zeros(Y.shape[1])
    for c in range(Y.shape[1]):
        scores = [_f1(Y[:, c], (P[:, c] >= t).astype(int)) for t in cand]
        thr[c] = cand[int(np.argmax(scores))]
    return thr


def evaluate(Y, P, thr):
    per = {}
    for c, name in enumerate(CLASSES):
        y, p = Y[:, c], P[:, c]
        _need_both_classes(y, name)
        pred = (p >= thr[c]).astype(int)
        tp = np.sum((y == 1) & (pred == 1)); fn = np.sum((y == 1) & (pred == 0))
        tn = np.sum((y == 0) & (pred == 0)); fp = np.sum((y == 0) & (pred == 1))
        per[name] = dict(
            auroc=float(roc_auc_score(y, p)),
            auprc=float(average_precision_score(y, p)),
            f1=float(_f1(y, pred)),
            sensitivity=float(tp / max(tp + fn, 1)),
            specificity=float(tn / max(tn + fp, 1)),
            mcc=float(matthews_corrcoef(y, pred)),
            brier=float(brier_score_loss(y, p)),
        )
    macro = {k: float(np.mean([per[n][k] for n in CLASSES])) for k in per[CLASSES[0]]}
    return dict(macro=macro, per_class=per)
