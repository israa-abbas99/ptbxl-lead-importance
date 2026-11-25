"""Torch-free end-to-end test of aggregate_2b.py and analyze_native.py on fake results. Run: python test_2b_pipeline.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from config import CLASSES, SEEDS
from config_2b import BASELINE, MODELS_2B
from leadlib import class_aurocs
from metrics import evaluate

rng = np.random.default_rng(1)
N, C = 600, 5
pid = np.repeat(np.arange(300), 2)
Y = (rng.random((N, C)) < 0.35).astype(np.float32)
R = tempfile.mkdtemp()
np.save(f"{R}/y_test.npy", Y); np.save(f"{R}/test_patient_ids.npy", pid)
skill = {BASELINE: 1.0, "m1_early_se": 0.9, "m2_separate": 1.0, "m3_cs_se": 1.1, "m4_mh_se": 1.2}
for m, sk in skill.items():
    for s in SEEDS:
        P = 1 / (1 + np.exp(-(sk * (Y - .5) * 1.5 + rng.normal(size=Y.shape))))
        thr = np.full(C, 0.5)
        res = dict(model=m, seed=s, n_params=1000, epochs_run=20, train_seconds=60.0,
                   val=evaluate(Y, P, thr), test=evaluate(Y, P, thr))
        os.makedirs(f"{R}/{m}/seed{s}", exist_ok=True)
        json.dump(res, open(f"{R}/{m}/seed{s}/metrics.json", "w"))
        np.save(f"{R}/{m}/seed{s}/preds_test.npy", P.astype(np.float32))

r = subprocess.run([sys.executable, "aggregate_2b.py", "--results", R, "--boot", "60"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]
import pandas as pd
c = pd.read_csv(f"{R}/summary_2b/table_comparisons_2b.csv")
assert len(c) == 5 and "SINGLE_CI_bonf(99%)" in c.columns
d = c.set_index("comparison")["SINGLE_diff"].astype(float)
assert d.iloc[3] > 0 and d.iloc[0] < 0, d          # M4 > M3 and M1 < M0 by construction
print("aggregate_2b OK")

# a missing run must stop the aggregation
os.remove(f"{R}/m4_mh_se/seed55/metrics.json")
r = subprocess.run([sys.executable, "aggregate_2b.py", "--results", R, "--boot", "20"], capture_output=True, text=True)
assert r.returncode != 0 and "missing" in (r.stdout + r.stderr), "missing run was not detected"
print("missing-run stop OK")

# ---- native-weight analysis on fake lead_perturb files
true_lead = np.array([0.9, 0.8, 0.1, 0.3, 0.2, 0.1, 0.7, 0.2, 0.1, 0.3, 0.1, 0.2])
for m in MODELS_2B:
    for s in SEEDS:
        Pb = (1 / (1 + np.exp(-(1.5 * (Y - .5) + rng.normal(size=Y.shape))))).astype(np.float32)
        Pz = np.stack([np.clip(Pb - 0.3 * true_lead[l] * (Y - .5) + 0.001 * rng.normal(size=Y.shape), 0, 1) for l in range(12)]).astype(np.float32)
        nat = np.tile(0.6 + 0.3 * true_lead, (C, 1)) + 0.01 * rng.normal(size=(C, 12))
        if m == "m1_early_se":
            nat = np.tile(nat.mean(0), (C, 1))
        curves = np.zeros((14, 12)); curves[:, 0] = .9
        for k in range(1, 12):
            curves[:, k] = .9 - 0.05 * k
        curves[3, 1:] -= 0.04                                    # native_val_most_first falls faster than random
        names = ["occlusion_val_most_first", "occlusion_val_least_first", "ig_most_first", "native_val_most_first"] + [f"random_{i}" for i in range(10)]
        os.makedirs(f"{R}/phase2a/{m}/seed{s}", exist_ok=True)
        np.savez(f"{R}/phase2a/{m}/seed{s}/lead_perturb.npz", P_base_test=Pb, P_zero_test=Pz, ig=np.abs(true_lead[:, None] + 0.05 * rng.normal(size=(12, C))),
                 del_curves=curves, del_names=np.array(names), native_mean_test=nat, native_pos_test=nat, native_mean_val=nat,
                 ckpt_pred_gap=0.0, has_native=True)
os.makedirs(f"{R}/m4_mh_se/seed55", exist_ok=True)
r = subprocess.run([sys.executable, "analyze_native.py", "--results", R], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]
stab = pd.read_csv(f"{R}/phase2b/table_native_stability.csv")
assert len(stab) == 3 and float(stab["stability_class_avg"].iloc[0].split()[0]) > 0.8      # m2 is skipped
ag = pd.read_csv(f"{R}/phase2b/table_native_agreement.csv")
assert float(ag["native_vs_zeroout_spearman"].iloc[0].split()[0]) > 0.5
dl = pd.read_csv(f"{R}/phase2b/table_native_deletion.csv")
assert (dl["seeds_native_below_random"] == "5/5").all()
sp = pd.read_csv(f"{R}/phase2b/table_native_class_specificity.csv")
assert float(sp.loc[sp.model == "m1_early_se", "cross_class_SD"].iloc[0]) < 1e-9          # shared gate: no class variation
print("analyze_native OK")
print("ALL 2B PIPELINE TESTS PASSED")
