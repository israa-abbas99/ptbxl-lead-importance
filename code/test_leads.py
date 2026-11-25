"""Tests for the torch-free parts of the lead-importance analysis. Run: python test_leads.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

from config import CLASSES, MODELS, SEEDS
from leadlib import (class_aurocs, fast_auroc, importance_from_preds, mean_pairwise_spearman, permute_lead,
                     random_orders, spearman, spearman_null, topk_overlap, zero_leads)

rng = np.random.default_rng(0)

# 1) fast AUROC equals sklearn (with ties), and refuses a single-class input
y = (rng.random(500) < 0.3).astype(int); p = np.round(rng.random(500), 2)
assert abs(fast_auroc(y, p) - roc_auc_score(y, p)) < 1e-12
try:
    fast_auroc(np.zeros(10), rng.random(10)); raise SystemExit("FAIL: single-class AUROC accepted")
except ValueError:
    pass

# 2) Spearman / stability / top-k overlap
a = rng.random(12)
assert abs(spearman(a, a) - 1) < 1e-12 and abs(spearman(a, -a) + 1) < 1e-12
V = np.tile(a, (5, 1))
assert abs(mean_pairwise_spearman(V) - 1) < 1e-12 and abs(topk_overlap(V, 3) - 1) < 1e-12
null = spearman_null(12, 5, n_sim=2000)
assert abs(null.mean()) < 0.03 and 0.05 < np.percentile(null, 95) < 0.35, np.percentile(null, 95)

# 3) perturbations: zeroing touches only the chosen leads; permuting keeps the lead's values (as a set)
X = rng.normal(size=(40, 12, 50)).astype(np.float32)
Xz = zero_leads(X, [2, 7])
assert (Xz[:, [2, 7]] == 0).all() and np.array_equal(Xz[:, [0, 1, 3]], X[:, [0, 1, 3]]) and X[:, 2].any()
Xp = permute_lead(X, 4, np.random.default_rng(1))
assert np.array_equal(np.sort(Xp[:, 4].ravel()), np.sort(X[:, 4].ravel())) and np.array_equal(Xp[:, :4], X[:, :4])
ro = random_orders(3, 12, seed=5)
assert all(sorted(o) == list(range(12)) for o in ro)

# 4) importance_from_preds recovers a known structure: only leads 1 and 6 carry signal
N, C = 800, len(CLASSES)
Y = (rng.random((N, C)) < 0.3).astype(np.float32)
lead_w = np.zeros(12); lead_w[1], lead_w[6] = 1.0, 0.5
sig = 2.0 * (Y - 0.5)
def make(seed):
    r = np.random.default_rng(seed)
    noise = r.normal(0, 1.0, (N, C))
    base = 1 / (1 + np.exp(-(lead_w.sum() * sig * 1.2 + noise)))
    zero = np.stack([1 / (1 + np.exp(-((lead_w.sum() - lead_w[l]) * sig * 1.2 + noise))) for l in range(12)])
    return base.astype(np.float32), zero.astype(np.float32)
Pb, Pz = make(0)
imp = importance_from_preds(Y, Pb, Pz)
assert imp.shape == (12, C + 1) and np.argmax(imp[:, -1]) == 1 and np.argsort(-imp[:, -1])[1] == 6
assert np.allclose(imp[:, -1], imp[:, :-1].mean(1)) and np.abs(imp[[0, 2, 3, 4, 5], -1]).max() < 1e-12

# 5) end-to-end analyze_leads on fake files (same structure as lead_perturb output)
pid = np.repeat(np.arange(N // 2), 2)
with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, "summary"))
    np.save(os.path.join(d, "y_test.npy"), Y); np.save(os.path.join(d, "test_patient_ids.npy"), pid)
    json.dump(dict(winner="cnn_lstm"), open(os.path.join(d, "summary", "winner.json"), "w"))
    names = np.array(["occlusion_val_most_first", "occlusion_val_least_first", "ig_most_first"] + [f"random_{i}" for i in range(10)])
    for m in MODELS:
        for k, s in enumerate(SEEDS):
            Pb, Pz = make(100 + k)
            dd = os.path.join(d, "phase2a", m, f"seed{s}"); os.makedirs(dd)
            igv = np.tile(lead_w[:, None], (1, C)) + np.random.default_rng(k).normal(0, 0.01, (12, C))
            order = np.argsort(-lead_w)
            curves = np.tile(np.linspace(0.9, 0.5, 12), (len(names), 1))
            np.savez_compressed(os.path.join(dd, "lead_perturb.npz"), P_base_test=Pb, P_zero_test=Pz,
                                drop_zero_val=np.zeros((12, C)), drop_perm_test=np.tile(lead_w[None, :, None] * 0.01, (3, 1, C)),
                                ig=igv, ig_idx=np.arange(20), ig_completeness_err=0.02, del_curves=curves,
                                del_names=names, del_orders=np.tile(order, (len(names), 1)), ckpt_pred_gap=1e-5)
    r = subprocess.run([sys.executable, "analyze_leads.py", "--results", d, "--boot", "30"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    import pandas as pd
    S = pd.read_csv(os.path.join(d, "phase2a", "summary", "table_stability.csv"))
    assert S.shape[0] == 6 and float(S["stability_macro"][0].split()[0]) > 0.7, S["stability_macro"][0]
    for f in ("table_lead_importance_macro.csv", "table_agreement.csv", "table_deletion.csv", "table_class_similarity.csv",
              "table_lead_importance_by_class_cnn_lstm.csv", "phase2a_log.json"):
        assert os.path.exists(os.path.join(d, "phase2a", "summary", f)), f
    for f in ("fig_lead_heatmap.png", "fig_lead_stability.png", "fig_lead_deletion.png"):
        assert os.path.exists(os.path.join(d, "phase2a", "summary", f)), f
    # a missing run must stop the analysis
    os.remove(os.path.join(d, "phase2a", "cnn", "seed55", "lead_perturb.npz"))
    r2 = subprocess.run([sys.executable, "analyze_leads.py", "--results", d, "--boot", "10"], capture_output=True, text=True)
    assert r2.returncode != 0 and "missing" in (r2.stdout + r2.stderr)

print("test_leads: ALL PASSED")
