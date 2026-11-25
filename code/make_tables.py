"""Report tables built directly from the saved files (nothing is retyped by hand).

  python make_tables.py --results /content/drive/MyDrive/ptbxl_results

Writes to <results>/summary/:
  table_per_class_full.csv / .md  per model x class: AUROC, AUPRC, F1, sensitivity, specificity (mean +- SD over seeds)
  table_inference.csv / .md       the 7 comparisons and 3 interactions in compact form (SINGLE primary; ENSEMBLE secondary)
Per-class numbers are descriptive: no per-class hypothesis tests were pre-specified.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import CLASSES, MODELS, SEEDS

METRICS = ["auroc", "auprc", "f1", "sensitivity", "specificity"]


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    summ = os.path.join(a.results, "summary")

    rows = []
    for m in MODELS:
        R = [json.load(open(os.path.join(a.results, m, f"seed{s}", "metrics.json"))) for s in SEEDS]
        assert all(r["model"] == m for r in R) and [r["seed"] for r in R] == list(SEEDS)
        for c in CLASSES:
            rows.append(dict(model=m, cls=c, **{k: ms([r["test"]["per_class"][c][k] for r in R]) for k in METRICS}))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(summ, "table_per_class_full.csv"), index=False)
    open(os.path.join(summ, "table_per_class_full.md"), "w").write(P.to_markdown(index=False))

    parts = []
    for fam in ("comparisons", "interactions"):
        T = pd.read_csv(os.path.join(summ, f"table_{fam}.csv"))
        bs = [c for c in T.columns if c.startswith("SINGLE_CI_bonf")][0]
        be = [c for c in T.columns if c.startswith("ENSEMBLE_CI_bonf")][0]
        parts.append(pd.DataFrame({
            "family": fam, "contrast": T["contrast"], "seeds_positive": T["seeds_positive"],
            "SINGLE diff": T["SINGLE_diff"], "SINGLE 95% CI": T["SINGLE_CI95"], "SINGLE Bonf CI": T[bs],
            "SINGLE sig (unadj / Bonf)": T["SINGLE_excl0_unadj"].astype(str) + " / " + T["SINGLE_excl0_bonf"].astype(str),
            "ENSEMBLE diff": T["ENSEMBLE_diff"], "ENSEMBLE 95% CI": T["ENSEMBLE_CI95"], "ENSEMBLE Bonf CI": T[be],
            "ENSEMBLE sig (unadj / Bonf)": T["ENSEMBLE_excl0_unadj"].astype(str) + " / " + T["ENSEMBLE_excl0_bonf"].astype(str)}))
    I = pd.concat(parts, ignore_index=True)
    I.to_csv(os.path.join(summ, "table_inference.csv"), index=False)
    open(os.path.join(summ, "table_inference.md"), "w").write(I.to_markdown(index=False))
    print(P.to_string(index=False)); print(); print(I.to_string(index=False))


if __name__ == "__main__":
    main()
