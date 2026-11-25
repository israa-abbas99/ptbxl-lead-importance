"""Phase 2B performance tables and the 5 pre-specified comparisons (no torch).
  python aggregate_2b.py --results /content/drive/MyDrive/ptbxl_results --boot 5000
Writes to <results>/summary_2b/: table_models_2b.csv/.md, table_per_class_2b.csv/.md, table_comparisons_2b.csv,
bootstrap_log_2b.json. Same estimands as Phase 1 (SINGLE primary, ENSEMBLE secondary; patient-level paired bootstrap;
Bonferroni family = 5 comparisons -> 99%). Stops if any of the 25 runs is missing.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from aggregate import boot_contrasts, fmt, ms, point_contrasts
from config import CLASSES, SEEDS
from config_2b import BASELINE, COMPARISONS_2B, LABEL_2B, MODELS_2B
from metrics import macro_auroc

ORDER = [BASELINE] + list(MODELS_2B)


def load(results):
    R = {}
    for m in ORDER:
        R[m] = {}
        for s in SEEDS:
            p = os.path.join(results, m, f"seed{s}", "metrics.json")
            if not os.path.exists(p):
                raise SystemExit(f"STOP: missing {p}. Finish all Phase-1 (cnn_lstm) and Phase-2B runs first.")
            r = json.load(open(p))
            assert r["model"] == m and r["seed"] == s, p
            R[m][s] = r
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    out = os.path.join(a.results, "summary_2b"); os.makedirs(out, exist_ok=True)
    R = load(a.results)

    rows, crow = [], []
    for m in ORDER:
        r = list(R[m].values())
        t = lambda k: [x["test"]["macro"][k] for x in r]
        rows.append({"model": LABEL_2B[m], "key": m, "params": r[0]["n_params"],
                     "val_macro_auroc": ms([x["val"]["macro"]["auroc"] for x in r]),
                     "test_macro_auroc": ms(t("auroc")), "test_macro_auprc": ms(t("auprc")),
                     "test_macro_f1": ms(t("f1")), "test_macro_mcc": ms(t("mcc")), "test_macro_brier": ms(t("brier")),
                     "epochs_run": f"{np.mean([x['epochs_run'] for x in r]):.0f}",
                     "wall_min_per_run": f"{np.mean([x['train_seconds'] for x in r]) / 60:.1f}"})
        c = {"model": LABEL_2B[m]}
        for k in CLASSES:
            c[f"{k} AUROC"] = ms([x["test"]["per_class"][k]["auroc"] for x in r])
            c[f"{k} AUPRC"] = ms([x["test"]["per_class"][k]["auprc"] for x in r])
        crow.append(c)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "table_models_2b.csv"), index=False)
    open(os.path.join(out, "table_models_2b.md"), "w").write(T.to_markdown(index=False))
    P = pd.DataFrame(crow); P.to_csv(os.path.join(out, "table_per_class_2b.csv"), index=False)
    open(os.path.join(out, "table_per_class_2b.md"), "w").write(P.to_markdown(index=False))
    print(T.to_string(index=False))

    Y = np.load(os.path.join(a.results, "y_test.npy")); pid = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    PS = {m: np.stack([np.load(os.path.join(a.results, m, f"seed{s}", "preds_test.npy")) for s in SEEDS]) for m in ORDER}
    for m in ORDER:
        assert PS[m].shape == (len(SEEDS),) + Y.shape, (m, PS[m].shape)
        for k, sd in enumerate(SEEDS):
            gap = abs(macro_auroc(Y, PS[m][k]) - R[m][sd]["test"]["macro"]["auroc"])
            assert gap < 1e-6, f"{m} seed {sd}: preds_test.npy does not match metrics.json (gap {gap:.2e})"
    contrasts = [{A: 1, B: -1} for _, A, B in COMPARISONS_2B]
    PE_S, PE_E = point_contrasts(Y, PS, contrasts)
    DS, DE, skipped = boot_contrasts(Y, PS, contrasts, pid, a.boot, np.random.default_rng(2026))
    json.dump(dict(n_bootstrap=a.boot, discarded_resamples=skipped, family_size=len(COMPARISONS_2B),
                   n_test_patients=int(len(np.unique(pid))), n_test_records=int(len(pid))),
              open(os.path.join(out, "bootstrap_log_2b.json"), "w"), indent=1)
    bonf = 1 - 0.05 / len(COMPARISONS_2B)
    rows = []
    for j, (label, A, B) in enumerate(COMPARISONS_2B):
        per_seed = np.array([R[A][s]["test"]["macro"]["auroc"] - R[B][s]["test"]["macro"]["auroc"] for s in SEEDS])
        sm, sc, sb, s0, s0b = fmt(PE_S[j], DS[:, j], bonf)
        em, ec, eb, e0, e0b = fmt(PE_E[j], DE[:, j], bonf)
        rows.append({"comparison": label, "seed_diff_mean±sd": ms(per_seed),
                     "seeds_positive": f"{int((per_seed > 0).sum())}/{len(SEEDS)}",
                     "SINGLE_diff": sm, "SINGLE_CI95": sc, f"SINGLE_CI_bonf({bonf*100:.0f}%)": sb,
                     "SINGLE_excl0_unadj": s0, "SINGLE_excl0_bonf": s0b,
                     "ENSEMBLE_diff": em, "ENSEMBLE_CI95": ec, f"ENSEMBLE_CI_bonf({bonf*100:.0f}%)": eb,
                     "ENSEMBLE_excl0_bonf": e0b})
    C = pd.DataFrame(rows); C.to_csv(os.path.join(out, "table_comparisons_2b.csv"), index=False)
    print(f"\nbootstrap {a.boot}, discarded {skipped}\n"); print(C.to_string(index=False))


if __name__ == "__main__":
    main()
