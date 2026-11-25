"""Revision analyses that need only saved outputs (no signals, no torch)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from aggregate import boot_contrasts, point_contrasts
from leadlib import mean_pairwise_spearman, spearman
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SEEDS = [11, 22, 33, 44, 55]
Y = np.load(f"{R}/y_test.npy"); pid = np.load(f"{R}/test_patient_ids.npy")
out = {}
# 1) M3 - M0, descriptive (same resamples as the pre-specified family: rng 2026, 5000)
PS = {m: np.stack([np.load(f"{R}/{m}/seed{s}/preds_test.npy") for s in SEEDS]) for m in ["cnn_lstm", "m1_early_se", "m2_separate", "m3_cs_se", "m4_mh_se"]}
contr = [{"m3_cs_se": 1, "cnn_lstm": -1}, {"m1_early_se": 1, "cnn_lstm": -1}, {"m2_separate": 1, "cnn_lstm": -1},
         {"m3_cs_se": 1, "m2_separate": -1}, {"m4_mh_se": 1, "m3_cs_se": -1}, {"m4_mh_se": 1, "m2_separate": -1}]
names = ["M3-M0", "M1-M0", "M2-M0", "M3-M2", "M4-M3", "M4-M2"]
pS, pE = point_contrasts(Y, PS, contr)
DS, DE, sk = boot_contrasts(Y, PS, contr, pid, 5000, np.random.default_rng(2026))
for i, n in enumerate(names):
    out[n] = dict(single=round(float(pS[i]), 5), ci95=[round(float(x), 5) for x in np.percentile(DS[:, i], [2.5, 97.5])],
                  ci99=[round(float(x), 5) for x in np.percentile(DS[:, i], [0.5, 99.5])],
                  ens=round(float(pE[i]), 5), ens_ci95=[round(float(x), 5) for x in np.percentile(DE[:, i], [2.5, 97.5])],
                  midpoint99=round(float(np.mean(np.percentile(DS[:, i], [0.5, 99.5]))), 5))
print(json.dumps(out, indent=1))
