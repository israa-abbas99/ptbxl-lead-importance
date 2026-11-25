"""Phase 2A, step 2 (CPU, no torch): tables and figures from the files written by lead_perturb.py.

  python analyze_leads.py --results /content/drive/MyDrive/ptbxl_results --boot 1000

Writes to <results>/phase2a/summary/. Endpoints and rules are fixed in PROTOCOL_PHASE2A.md.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import CLASSES, MODELS, SEEDS
from leadlib import (LEADS, class_aurocs, importance_from_preds, mean_pairwise_spearman, spearman,
                     spearman_null, topk_overlap)

COLS = CLASSES + ["macro"]


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def load_all(pdir, models):
    Z = {}
    for m in models:
        Z[m] = []
        for s in SEEDS:
            p = os.path.join(pdir, m, f"seed{s}", "lead_perturb.npz")
            if not os.path.exists(p):
                raise SystemExit(f"STOP: missing {p}. Run lead_perturb.py for all models and seeds first.")
            Z[m].append(dict(np.load(p)))
            if float(Z[m][-1]["ckpt_pred_gap"]) > 5e-3:
                raise SystemExit(f"STOP: {p} was produced by a checkpoint that does not match its predictions.")
    return Z


def stability_all(imp):
    """imp: (n_seeds, 12, 6) -> dict of stability per column (macro and classes) and top-3 overlap of macro."""
    st = {c: mean_pairwise_spearman(imp[:, :, j]) for j, c in enumerate(COLS)}
    st["top3_macro"] = topk_overlap(imp[:, :, -1], 3)
    return st


def boot_stability(Y, pid, Z, models, B, rng):
    uniq, inv = np.unique(pid, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    res = {m: [] for m in models}
    skipped = 0
    while len(res[models[0]]) < B:
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))])
        Yb = Y[idx]; pos = Yb.sum(0)
        if (pos == 0).any() or (pos == len(Yb)).any():
            skipped += 1
            if skipped > 0.01 * B:
                raise RuntimeError(f"too many unusable bootstrap resamples ({skipped} of {B})")
            continue
        for m in models:
            imp = np.stack([importance_from_preds(Yb, z["P_base_test"][idx], z["P_zero_test"][:, idx]) for z in Z[m]])
            res[m].append([stability_all(imp)[c] for c in COLS])
    return {m: np.array(v) for m, v in res.items()}, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    pdir = os.path.join(a.results, "phase2a")
    out = os.path.join(pdir, "summary")
    os.makedirs(out, exist_ok=True)
    Y = np.load(os.path.join(a.results, "y_test.npy"))
    pid = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    winner = json.load(open(os.path.join(a.results, "summary", "winner.json")))["winner"]
    models = list(MODELS)
    Z = load_all(pdir, models)

    # point estimates: importance = AUROC drop when one lead is zeroed (test fold)
    IMP = {m: np.stack([importance_from_preds(Y, z["P_base_test"], z["P_zero_test"]) for z in Z[m]]) for m in models}

    # 1. importance tables
    rows = [{"model": m, **{L: ms(IMP[m][:, l, -1]) for l, L in enumerate(LEADS)}} for m in models]
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_lead_importance_macro.csv"), index=False)
    W = pd.DataFrame({c: [ms(IMP[winner][:, l, j]) for l in range(12)] for j, c in enumerate(COLS)}, index=LEADS)
    W.to_csv(os.path.join(out, f"table_lead_importance_by_class_{winner}.csv"))

    # 2. seed stability with patient-level bootstrap CI, and the null for random rankings
    null = spearman_null(12, len(SEEDS))
    thr = float(np.percentile(null, 95))
    BS, skipped = boot_stability(Y, pid, Z, models, a.boot, np.random.default_rng(2026))
    rows = []
    for m in models:
        pt = stability_all(IMP[m])
        row = {"model": m, "top3_overlap_macro": f"{pt['top3_macro']:.3f}"}
        for j, c in enumerate(COLS):
            lo, hi = np.percentile(BS[m][:, j], [2.5, 97.5])
            row[f"stability_{c}"] = f"{pt[c]:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "table_stability.csv"), index=False)
    print(f"null 95th percentile of the mean pairwise Spearman (random rankings): {thr:.3f}")
    print(T.to_string(index=False))

    # 3. agreement between importance measures (descriptive; IG uses a fixed subset of test records)
    rows = []
    for m in models:
        a_ig, a_pm = [], []
        for z, imp in zip(Z[m], IMP[m]):
            a_ig.append(np.mean([spearman(imp[:, j], z["ig"][:, j]) for j in range(len(CLASSES))]))
            a_pm.append(np.mean([spearman(imp[:, j], z["drop_perm_test"].mean(0)[:, j]) for j in range(len(CLASSES))]))
        rows.append({"model": m, "zero_vs_IG_spearman": ms(a_ig, 3), "zero_vs_permutation_spearman": ms(a_pm, 3),
                     "IG_completeness_err": f"{np.mean([float(z['ig_completeness_err']) for z in Z[m]]):.3f}"})
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(out, "table_agreement.csv"), index=False)
    print(); print(A.to_string(index=False))

    # 4. do classes use different leads? (mean Spearman between class importance vectors, seed-averaged)
    rows = []
    for m in models:
        pairs = [np.mean([spearman(imp[:, i], imp[:, j]) for imp in IMP[m]])
                 for i in range(5) for j in range(i + 1, 5)]
        rows.append({"model": m, "mean_between_class_spearman": f"{np.mean(pairs):.3f}"})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(out, "table_class_similarity.csv"), index=False)

    # 5. deletion curves (orders from validation occlusion / IG; random orders are the reference)
    names = [str(n) for n in Z[models[0]][0]["del_names"]]
    rnd = [i for i, n in enumerate(names) if n.startswith("random")]
    rows, curves = [], {}
    for m in models:
        D = np.stack([z["del_curves"] for z in Z[m]])                # (seeds, orders, 12)
        curves[m] = D
        for k in (3, 6):
            r = {"model": m, "k_leads_removed": k}
            for nme in ("occlusion_val_most_first", "occlusion_val_least_first", "ig_most_first"):
                r[nme] = ms(D[:, names.index(nme), k], 4)
            r["random_mean"] = ms(D[:, rnd, k].mean(1), 4)
            r["seeds_most_first_below_random"] = f"{int((D[:, names.index('occlusion_val_most_first'), k] < D[:, rnd, k].mean(1)).sum())}/{len(SEEDS)}"
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_deletion.csv"), index=False)

    json.dump(dict(winner=winner, n_bootstrap=a.boot, discarded_resamples=skipped,
                   null95_mean_pairwise_spearman=thr, ig_subset_size=int(len(Z[winner][0]["ig_idx"]))),
              open(os.path.join(out, "phase2a_log.json"), "w"), indent=1)
    print("\nwinner:", winner); print(W.to_string())
    make_figs(out, IMP, BS, thr, curves, names, rnd, winner, models)


def make_figs(out, IMP, BS, thr, curves, names, rnd, winner, models):
    import matplotlib.pyplot as plt
    from make_figures import LABEL, save, setup, style_of
    setup()
    # heatmap: importance of each lead for each class (winner, mean over seeds)
    M = IMP[winner].mean(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(COLS); ax.set_yticks(range(12)); ax.set_yticklabels(LEADS)
    ax.grid(False)
    for i in range(12):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.6 * M.max() else "#0b0b0b")
    fig.colorbar(im, label="AUROC drop when the lead is zeroed")
    ax.set_title(f"{LABEL[winner]}: lead importance (mean of 5 seeds)", fontsize=9.5, loc="left")
    save(fig, out, "fig_lead_heatmap")
    # stability with CI, against the null for random rankings
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, m in enumerate(models):
        col, ms_ = style_of(m)
        pt = mean_pairwise_spearman(IMP[m][:, :, -1])
        lo, hi = np.percentile(BS[m][:, -1], [2.5, 97.5])
        ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="s" if ms_ else "o", color=col, capsize=3, lw=1.6)
    ax.axvline(thr, color="#52514e", ls="--", lw=1)
    ax.text(thr, -0.8, " null (random rankings), 95th pct", fontsize=7.5, color="#52514e")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(None, 1.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seed stability of lead importance (mean pairwise Spearman, macro; 95% CI)")
    save(fig, out, "fig_lead_stability")
    # deletion curves
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    D = curves[winner]; k = np.arange(12)
    for nme, col, ls in (("occlusion_val_most_first", "#e34948", "-"), ("ig_most_first", "#eb6834", "--"),
                         ("occlusion_val_least_first", "#2a78d6", "-")):
        ax.plot(k, D[:, names.index(nme)].mean(0), color=col, ls=ls, lw=1.8, label=nme.replace("_", " "))
    R = D[:, rnd].mean(1)
    ax.plot(k, R.mean(0), color="#52514e", ls=":", lw=1.8, label="random order (mean of 10)")
    ax.set_xlabel("Number of leads removed (zeroed)"); ax.set_ylabel("Test macro-AUROC")
    ax.legend(fontsize=7.5)
    ax.set_title(f"{LABEL[winner]}: deletion curves (mean of 5 seeds)", fontsize=9.5, loc="left")
    save(fig, out, "fig_lead_deletion")


if __name__ == "__main__":
    main()
