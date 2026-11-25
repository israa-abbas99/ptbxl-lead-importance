"""Exploratory (NOT pre-specified) analyses for the Phase 2B results. CPU only, no torch, no retraining.
  python extra_crossmodel.py --results /content/drive/MyDrive/ptbxl_results

Reads the existing <results>/phase2a/<model>/seed<k>/lead_perturb.npz files and writes <results>/extra/:
  A) table_extra_crossmodel.csv : agreement (Spearman over the 12 leads, per class) between
       - M4 gate weights and M4 zero-out importance (own model, as in the paper),
       - M4 gate weights and M0 (CNN-LSTM) zero-out importance (independent model),
       - M4 zero-out importance and M0 zero-out importance.
     Reported for matched seeds (mean +- SD over 5 seed pairs) and over all 25 seed pairs.
  B) table_extra_random_deletion.csv : test macro-AUROC after zeroing k = 0, 3, 6, 9 leads in random order
     (mean of the 10 fixed random orders per run, then mean +- SD over the 5 seeds), for M0-M4.
Nothing here is a hypothesis test; label any use of it in the paper as exploratory.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Standalone: same definitions as config.py / leadlib.py, copied so the script runs from any folder.
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = [11, 22, 33, 44, 55]


def fast_auroc(y, p):
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


def importance_from_preds(Y, P_base, P_zero):
    base = class_aurocs(Y, P_base)
    drop = np.stack([base - class_aurocs(Y, P_zero[l]) for l in range(P_zero.shape[0])])
    return np.concatenate([drop, drop.mean(1, keepdims=True)], axis=1)


def find_results():
    """Locate the results folder on Drive: the one that holds y_test.npy and phase2a/m4_mh_se."""
    for y in glob.glob("/content/drive/MyDrive/**/y_test.npy", recursive=True):
        r = os.path.dirname(y)
        if os.path.isdir(os.path.join(r, "phase2a", "m4_mh_se")):
            return r
    raise SystemExit("STOP: could not find a results folder with y_test.npy and phase2a/m4_mh_se on Drive.")

M0 = "cnn_lstm"
M4 = "m4_mh_se"
ALL = [("M0 CNN-LSTM", "cnn_lstm"), ("M1 Early-Fusion SE", "m1_early_se"), ("M2 Separate-Lead", "m2_separate"),
       ("M3 Channel-Specific SE", "m3_cs_se"), ("M4 Multi-Head SE", "m4_mh_se")]
KS = [0, 3, 6, 9]


def load(pdir, m):
    Z = []
    for s in SEEDS:
        p = os.path.join(pdir, m, f"seed{s}", "lead_perturb.npz")
        if not os.path.exists(p):
            raise SystemExit(f"STOP: missing {p}")
        z = dict(np.load(p))
        if float(z["ckpt_pred_gap"]) > 5e-3:
            raise SystemExit(f"STOP: {p}: checkpoint does not match saved predictions.")
        Z.append(z)
    return Z


def ms(v, d=2):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):+.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="results folder; found automatically on Drive if omitted")
    a = ap.parse_args()
    if not a.results or not os.path.exists(os.path.join(a.results, "y_test.npy")):
        a.results = find_results()
    print("results folder:", a.results)
    pdir = os.path.join(a.results, "phase2a")
    out = os.path.join(a.results, "extra"); os.makedirs(out, exist_ok=True)
    Y = np.load(os.path.join(a.results, "y_test.npy"))
    C = len(CLASSES)

    # ---------- A) cross-model agreement ----------
    Z0, Z4 = load(pdir, M0), load(pdir, M4)
    if not all(bool(z["has_native"]) for z in Z4):
        raise SystemExit("STOP: M4 files carry no gate weights.")
    G4 = np.stack([z["native_mean_test"] for z in Z4])                                    # (S, C, 12)
    I4 = np.stack([importance_from_preds(Y, z["P_base_test"], z["P_zero_test"])[:, :C] for z in Z4])  # (S, 12, C)
    I0 = np.stack([importance_from_preds(Y, z["P_base_test"], z["P_zero_test"])[:, :C] for z in Z0])
    S = len(SEEDS)
    rows = []
    for c, cname in enumerate(CLASSES):
        own = [spearman(G4[s, c], I4[s, :, c]) for s in range(S)]
        x_m = [spearman(G4[s, c], I0[s, :, c]) for s in range(S)]
        x_all = [spearman(G4[i, c], I0[j, :, c]) for i in range(S) for j in range(S)]
        zz_m = [spearman(I4[s, :, c], I0[s, :, c]) for s in range(S)]
        zz_all = [spearman(I4[i, :, c], I0[j, :, c]) for i in range(S) for j in range(S)]
        rows.append({"class": cname,
                     "M4_gate_vs_M4_zeroout (matched)": ms(own),
                     "M4_gate_vs_M0_zeroout (matched)": ms(x_m),
                     "M4_gate_vs_M0_zeroout (all 25 pairs)": f"{np.nanmean(x_all):+.2f}",
                     "M4_zeroout_vs_M0_zeroout (matched)": ms(zz_m),
                     "M4_zeroout_vs_M0_zeroout (all 25 pairs)": f"{np.nanmean(zz_all):+.2f}"})
    dis = [r for r in rows if r["class"] != "NORM"]
    rows.append({"class": "mean of 4 disease classes",
                 **{k: f"{np.mean([float(r[k].split()[0]) for r in dis]):+.2f}" for k in rows[0] if k != "class"}})
    dfA = pd.DataFrame(rows); dfA.to_csv(os.path.join(out, "table_extra_crossmodel.csv"), index=False)

    # ---------- B) robustness to random lead deletion ----------
    rowsB = []
    for label, m in ALL:
        Z = Z0 if m == M0 else (Z4 if m == M4 else load(pdir, m))
        per_seed = []
        for z in Z:
            names = [str(n) for n in z["del_names"]]
            rnd = [i for i, n in enumerate(names) if n.startswith("random_")]
            per_seed.append(z["del_curves"][rnd].mean(0))                                # (12,) mean over 10 orders
        P = np.stack(per_seed)                                                            # (S, 12)
        rowsB.append({"model": label, **{f"k={k}": f"{P[:, k].mean():.3f} ± {P[:, k].std(ddof=1):.3f}" for k in KS}})
    dfB = pd.DataFrame(rowsB); dfB.to_csv(os.path.join(out, "table_extra_random_deletion.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("\nA) Cross-model agreement (Spearman over 12 leads)\n", dfA.to_string(index=False))
    print("\nB) Test macro-AUROC after random lead deletion (k leads zeroed)\n", dfB.to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
