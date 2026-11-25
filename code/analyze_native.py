"""Phase 2B explanation analysis (CPU, no torch): built-in lead weights of M1-M4 against external measures.
  python analyze_native.py --results /content/drive/MyDrive/ptbxl_results
Needs <results>/phase2a/<model>/seed<k>/lead_perturb.npz for the four 2B models (written by lead_perturb.py --models ...).
Writes <results>/phase2b/: table_native_stability.csv, table_native_agreement.csv, table_native_saturation.csv,
table_native_class_specificity.csv, table_native_deletion.csv, table_native_weights_<model>.csv.
Endpoints are fixed in PROTOCOL_PHASE2B.md; nothing here is a hypothesis test.
"""
import argparse
import os

import numpy as np
import pandas as pd

from config import CLASSES, SEEDS
from config_2b import MODELS_2B
from leadlib import LEADS, importance_from_preds, mean_pairwise_spearman, spearman, spearman_null


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def load(pdir, m):
    Z = []
    for s in SEEDS:
        p = os.path.join(pdir, m, f"seed{s}", "lead_perturb.npz")
        if not os.path.exists(p):
            raise SystemExit(f"STOP: missing {p}. Run lead_perturb.py --models {m} first.")
        z = dict(np.load(p))
        if not bool(z["has_native"]):
            raise SystemExit(f"STOP: {p} has no built-in weights (was it produced for a Phase-1 model?).")
        if float(z["ckpt_pred_gap"]) > 5e-3:
            raise SystemExit(f"STOP: {p}: checkpoint does not match saved predictions.")
        Z.append(z)
    return Z


def pair_stats(V):
    n = len(V)
    pr = [spearman(V[i], V[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.nanmean(pr)), float(np.nanmin(pr)), float(np.nanmax(pr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    pdir = os.path.join(a.results, "phase2a"); out = os.path.join(a.results, "phase2b"); os.makedirs(out, exist_ok=True)
    Y = np.load(os.path.join(a.results, "y_test.npy"))
    thr = float(np.percentile(spearman_null(12, len(SEEDS)), 95))
    st, ag, sat, spec, dele = [], [], [], [], []
    for m in [k for k in MODELS_2B if k != "m2_separate"]:      # M2 has no gate
        Z = load(pdir, m)
        NAT = np.stack([z["native_mean_test"] for z in Z])                 # (S, C, 12) mean gate weight over test records
        v = NAT.mean(1)                                                    # class-averaged (S, 12)
        mean_, lo, hi = pair_stats(v)
        pc = [pair_stats(NAT[:, c])[0] for c in range(len(CLASSES))]
        st.append({"model": m, "stability_class_avg": f"{mean_:.3f} (pairs {lo:.3f} to {hi:.3f})",
                   "null95": f"{thr:.3f}", **{f"stab_{c}": f"{x:.3f}" for c, x in zip(CLASSES, pc)}})
        pd.DataFrame({"lead": LEADS, **{c: [ms(NAT[:, c_i, l]) for l in range(12)] for c_i, c in enumerate(CLASSES)}}
                     ).to_csv(os.path.join(out, f"table_native_weights_{m}.csv"), index=False)
        IMPZ = np.stack([importance_from_preds(Y, z["P_base_test"], z["P_zero_test"]) for z in Z])   # (S, 12, C+1)
        pd.DataFrame({"lead": LEADS, **{c: [ms(IMPZ[:, l, j], 4) for l in range(12)] for j, c in enumerate(CLASSES + ["macro"])}}
                     ).to_csv(os.path.join(out, f"table_native_zeroout_{m}.csv"), index=False)
        # agreement with external measures (per class, then mean over classes, then over seeds)
        a_z, a_ig = [], []
        for s_i, z in enumerate(Z):
            imp = importance_from_preds(Y, z["P_base_test"], z["P_zero_test"])[:, :len(CLASSES)]
            a_z.append(np.nanmean([spearman(NAT[s_i, c], imp[:, c]) for c in range(len(CLASSES))]))
            a_ig.append(np.nanmean([spearman(NAT[s_i, c], z["ig"][:, c]) for c in range(len(CLASSES))]))
        ag.append({"model": m, "native_vs_zeroout_spearman": ms(a_z), "native_vs_IG_spearman": ms(a_ig),
                   "seeds_positive_zeroout": f"{int((np.array(a_z) > 0).sum())}/{len(SEEDS)}"})
        # saturation: how far from 1 are the gates, and how much do leads differ within a record set
        sat.append({"model": m, "mean_weight": f"{NAT.mean():.3f}", "min_weight": f"{NAT.min():.3f}",
                    "max_weight": f"{NAT.max():.3f}",
                    "range_across_leads(max-min, class-avg, mean over seeds)": f"{(v.max(1) - v.min(1)).mean():.3f}"})
        # class specificity vs seed variability: same units (SD of one lead's weight)
        cross = NAT.std(1, ddof=1).mean()            # SD across classes (per seed, lead) averaged
        seedsd = NAT.std(0, ddof=1).mean()           # SD across seeds (per class, lead) averaged
        spec.append({"model": m, "cross_class_SD": f"{cross:.4f}", "cross_seed_SD": f"{seedsd:.4f}",
                     "ratio_class_over_seed": f"{cross / seedsd:.2f}" if seedsd > 0 else "nan"})
        # faithfulness: delete leads in order of VALIDATION native weights (largest first)
        names = [str(n) for n in Z[0]["del_names"]]
        rnd = [i for i, n in enumerate(names) if n.startswith("random")]
        D = np.stack([z["del_curves"] for z in Z])
        for k in (3, 6):
            r = {"model": m, "k_removed": k}
            for nme in ("native_val_most_first", "occlusion_val_most_first", "ig_most_first"):
                r[nme] = ms(D[:, names.index(nme), k], 4)
            r["random_mean"] = ms(D[:, rnd, k].mean(1), 4)
            r["seeds_native_below_random"] = f"{int((D[:, names.index('native_val_most_first'), k] < D[:, rnd, k].mean(1)).sum())}/{len(SEEDS)}"
            dele.append(r)
    for nme, rows in (("stability", st), ("agreement", ag), ("saturation", sat), ("class_specificity", spec), ("deletion", dele)):
        df = pd.DataFrame(rows); df.to_csv(os.path.join(out, f"table_native_{nme}.csv"), index=False)
        print(f"\n[{nme}]\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
