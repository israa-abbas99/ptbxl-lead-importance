"""Sensitivity analysis for Integrated Gradients (requested in review; NOT part of the frozen protocol).

Answers two questions without any retraining:
  (1) Does the IG result change when the integral is approximated more finely?   -> 128 steps instead of 32
  (2) Does the IG deletion curve change when the IG ranking comes from the VALIDATION fold instead of test inputs?

For every (model, seed) it loads the saved checkpoint, checks that it reproduces the saved test predictions,
and computes
  * IG with --steps steps on the SAME 500 test records used in Phase 2A  -> completeness error, P2 agreement, gate agreement
  * IG with --steps steps on 500 fixed VALIDATION records               -> lead ranking for the deletion curve
  * a deletion curve on the test fold in that validation IG order (k = 0..11)
It reads the existing Phase-2A file <results>/phase2a/<model>/seed<k>/lead_perturb.npz for the 32-step values,
the zero-out drops, the random-order curves and the built-in gate weights. Nothing existing is overwritten.

Output: <results>/ig_sens/<model>/seed<k>.npz (resumable) and <results>/ig_sens/table_ig_sensitivity.csv

Colab:
  python ig_sensitivity.py --root /content/ptbxl_data --cache /content/drive/MyDrive/ptbxl_cache \
                           --results /content/drive/MyDrive/ptbxl_results
  # only the summary table again (CPU, seconds):
  python ig_sensitivity.py --results /content/drive/MyDrive/ptbxl_results --summary-only
"""
import argparse
import os
import time

import numpy as np

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = [11, 22, 33, 44, 55]
DEFAULT_MODELS = ["cnn_lstm", "m1_early_se", "m2_separate", "m3_cs_se", "m4_mh_se"]
GATED = {"m1_early_se", "m3_cs_se", "m4_mh_se"}
VAL_SUBSET_SEED = 2027          # differs from the test-subset seed (2026) on purpose; fixed before running
KS = (3, 6)


# ---------------------------------------------------------------- torch-free helpers (summary)
def _spearman(a, b):
    from leadlib import spearman
    return spearman(a, b)


def _class_aurocs(Y, P):
    from leadlib import class_aurocs
    return class_aurocs(Y, P)


def zero_drop_test(Y, z):
    """(12, C) drop in per-class test AUROC when each lead is zeroed, from the saved Phase-2A predictions."""
    base = _class_aurocs(Y, z["P_base_test"])
    return np.stack([base - _class_aurocs(Y, z["P_zero_test"][l]) for l in range(z["P_zero_test"].shape[0])])


def agree(A, B):
    """Mean over classes of the Spearman correlation between two (12, C) importance maps."""
    return float(np.nanmean([_spearman(A[:, c], B[:, c]) for c in range(A.shape[1])]))


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


def summarise(results, models):
    import pandas as pd
    Y = np.load(os.path.join(results, "y_test.npy"))
    rows = []
    for m in models:
        R = {k: [] for k in ("err32", "errN", "p2_32", "p2_N", "ig32_vs_igN", "gate_ig32", "gate_igN",
                             "valig_vs_testig")}
        for k in KS:
            R[f"testIG32_k{k}"] = []; R[f"valIGN_k{k}"] = []; R[f"rand_k{k}"] = []; R[f"below_k{k}"] = []
        for s in SEEDS:
            p = os.path.join(results, "ig_sens", m, f"seed{s}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"STOP: missing {p}; run the GPU part first.")
            g = dict(np.load(p))
            z = dict(np.load(os.path.join(results, "phase2a", m, f"seed{s}", "lead_perturb.npz")))
            drop = zero_drop_test(Y, z)
            R["err32"].append(float(z["ig_completeness_err"])); R["errN"].append(float(g["err_test"]))
            R["p2_32"].append(agree(drop, z["ig"])); R["p2_N"].append(agree(drop, g["ig_test"]))
            R["ig32_vs_igN"].append(agree(z["ig"], g["ig_test"]))
            R["valig_vs_testig"].append(agree(g["ig_val"], g["ig_test"]))
            if m in GATED:
                W = z["native_mean_test"].T                       # (12, C)
                R["gate_ig32"].append(agree(W, z["ig"])); R["gate_igN"].append(agree(W, g["ig_test"]))
            names = [str(n) for n in z["del_names"]]
            rnd = [i for i, n in enumerate(names) if n.startswith("random_")]
            for k in KS:
                rmean = z["del_curves"][rnd, k].mean()
                R[f"testIG32_k{k}"].append(z["del_curves"][names.index("ig_most_first"), k])
                R[f"valIGN_k{k}"].append(g["curve_val_ig"][k])
                R[f"rand_k{k}"].append(rmean)
                R[f"below_k{k}"].append(int(g["curve_val_ig"][k] < rmean))
        row = {"model": m,
               "IG completeness error, 32 steps (median)": ms(R["err32"]),
               f"IG completeness error, {int(g['steps'])} steps (median)": ms(R["errN"]),
               f"Spearman IG32 vs IG{int(g['steps'])} (same test records)": ms(R["ig32_vs_igN"]),
               "P2 agreement zero-out vs IG32": ms(R["p2_32"]),
               f"P2 agreement zero-out vs IG{int(g['steps'])}": ms(R["p2_N"]),
               "Spearman validation IG vs test IG": ms(R["valig_vs_testig"])}
        if m in GATED:
            row["gate weights vs IG32"] = ms(R["gate_ig32"])
            row[f"gate weights vs IG{int(g['steps'])}"] = ms(R["gate_igN"])
        for k in KS:
            row[f"k={k}: test-IG32 order"] = ms(R[f"testIG32_k{k}"], 4)
            row[f"k={k}: validation-IG{int(g['steps'])} order"] = ms(R[f"valIGN_k{k}"], 4)
            row[f"k={k}: random (mean of 10)"] = ms(R[f"rand_k{k}"], 4)
            row[f"k={k}: seeds val-IG order below random"] = f"{sum(R[f'below_k{k}'])}/{len(SEEDS)}"
        rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(results, "ig_sens", "table_ig_sensitivity.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
    print(df.T.to_string(header=False))
    print("\nwrote", out)


# ---------------------------------------------------------------- GPU part
def run_gpu(a, models):
    import torch
    from data import load_ptbxl, split_and_standardise
    from lead_perturb import SUBSET_SEED, integrated_gradients
    from leadlib import zero_leads
    from train import build, predict

    data = split_and_standardise(load_ptbxl(a.root, a.cache))
    Xva, Yva = data["val"]; Xte, Yte = data["test"]
    if not (np.array_equal(np.load(os.path.join(a.results, "y_test.npy")), Yte)
            and np.array_equal(np.load(os.path.join(a.results, "test_patient_ids.npy")), data["test_patient_id"])):
        raise SystemExit("STOP: test labels / patient ids differ from the results folder (different data or cache).")
    te_idx = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(Xte), size=min(a.ig_n, len(Xte)), replace=False))
    va_idx = np.sort(np.random.default_rng(VAL_SUBSET_SEED).choice(len(Xva), size=min(a.ig_n, len(Xva)), replace=False))
    print(f"device {a.device} | IG steps {a.steps} | test records {len(te_idx)} | validation records {len(va_idx)}", flush=True)
    print(f"  test subset: {len(np.unique(data['test_patient_id'][te_idx]))} unique patients", flush=True)
    for nm, Ys in (("test", Yte[te_idx]), ("validation", Yva[va_idx])):
        print(f"  {nm} subset positives per class: " +
              ", ".join(f"{c} {int(Ys[:, i].sum())}" for i, c in enumerate(CLASSES)), flush=True)

    for m in models:
        for s in SEEDS:
            dst = os.path.join(a.results, "ig_sens", m, f"seed{s}.npz")
            if os.path.exists(dst):
                print("skip (done):", m, s); continue
            z = dict(np.load(os.path.join(a.results, "phase2a", m, f"seed{s}", "lead_perturb.npz")))
            if not np.array_equal(z["ig_idx"], te_idx):
                raise SystemExit("STOP: the IG test subset differs from the one used in Phase 2A.")
            t0 = time.time()
            d = os.path.join(a.results, m, f"seed{s}")
            model = build(m).to(a.device)
            model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=a.device))
            model.eval()
            P = predict(model, Xte, a.device)
            gap = float(np.abs(P - np.load(os.path.join(d, "preds_test.npy"))).max())
            if gap > 5e-3:
                raise SystemExit(f"STOP: {m} seed {s}: checkpoint does not reproduce saved predictions ({gap:.1e}).")
            ig_te, err_te = integrated_gradients(model, Xte[te_idx], a.device, a.steps)
            ig_va, err_va = integrated_gradients(model, Xva[va_idx], a.device, a.steps)
            order = np.argsort(-(ig_va / ig_va.sum(0, keepdims=True)).mean(1))       # same macro rule as Phase 2A
            curve_cls = np.zeros((Xte.shape[1], len(CLASSES)))          # per-class AUROC at each k (descriptive)
            curve_cls[0] = _class_aurocs(Yte, P)
            for k in range(1, Xte.shape[1]):
                curve_cls[k] = _class_aurocs(Yte, predict(model, zero_leads(Xte, order[:k]), a.device))
            curve = curve_cls.mean(1)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, ig_test=ig_te, err_test=err_te, ig_val=ig_va, err_val=err_va,
                                order_val_ig=order, curve_val_ig=curve, curve_val_ig_per_class=curve_cls, steps=a.steps, va_idx=va_idx, ckpt_gap=gap)
            print(f"DONE {m} seed {s}: IG err test {err_te:.3f} (was {float(z['ig_completeness_err']):.3f}), "
                  f"val {err_va:.3f}, {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--cache", default="./cache")
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--ig-n", type=int, default=500)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not a.summary_only:
        import torch
        a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if a.root is None:
            raise SystemExit("STOP: --root is required for the GPU part.")
        run_gpu(a, a.models)
    summarise(a.results, a.models)


if __name__ == "__main__":
    main()
