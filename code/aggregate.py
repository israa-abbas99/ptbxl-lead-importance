"""Aggregate Phase-1 results (no torch needed).

Outputs in <results>/summary/:
  table_models.csv / .md   mean +- sd over seeds (test fold), params, times
  table_per_class.csv      per-class AUROC mean +- sd
  table_comparisons.csv    7 pre-specified comparisons
  table_interactions.csv   3 contrasts: does the MHSA effect depend on the back-end?
  winner.json              chosen by mean VALIDATION macro-AUROC only

Two estimands, both with patient-level bootstrap (all records of a patient
resampled together; the SAME resampled patients are used for every model, so
every difference is paired on patients):
  SINGLE  primary   : mean over the 5 seeds of the single-model macro-AUROC
                      difference (matches the mean +- SD in table_models).
  ENSEMBLE secondary: difference of macro-AUROC of the seed-averaged
                      probabilities (describes the 5-seed ensemble of each model).
CIs: 95% and Bonferroni-adjusted within each family (7 comparisons; 3 interactions).
Point estimates are computed on the ORIGINAL test set; the bootstrap supplies percentile CIs only.
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from config import CFG, CLASSES, COMPARISONS, INTERACTIONS, MODELS, SEEDS
from metrics import macro_auroc


def load(results):
    rows = {}
    for name in MODELS:
        rows[name] = {}
        for s in SEEDS:
            p = os.path.join(results, name, f"seed{s}", "metrics.json")
            if os.path.exists(p):
                r = json.load(open(p))
                assert r.get("model") == name and r.get("seed") == s, f"{p} belongs to {r.get('model')}/{r.get('seed')}"
                rows[name][s] = r
    return rows


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"


def boot_contrasts(Y, PS, contrasts, pid, B, rng):
    """PS[model] -> (S, N, C) test probabilities. Returns (single, ensemble),
    each (B', n_contrasts), from the same patient resamples."""
    uniq, inv = np.unique(pid, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    need = sorted({m for c in contrasts for m in c})
    ens = {m: PS[m].mean(0) for m in need}
    single, ensemble, skipped = [], [], 0
    while len(single) < B:
        samp = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[k] for k in samp])
        Yb = Y[idx]
        # Policy (fixed): a resample in which any class has no positives or no
        # negatives is discarded and redrawn; the count is reported, and the
        # run aborts if more than 1% of the requested draws are discarded.
        # The check is explicit, so any other error is raised and never counted here.
        pos = Yb.sum(0)
        if (pos == 0).any() or (pos == len(Yb)).any():
            skipped += 1
            if skipped > 0.01 * B:
                raise RuntimeError(f"too many unusable bootstrap resamples ({skipped} of {B})")
            continue
        aS = {m: np.mean([macro_auroc(Yb, PS[m][s][idx]) for s in range(PS[m].shape[0])]) for m in need}
        aE = {m: macro_auroc(Yb, ens[m][idx]) for m in need}
        single.append([sum(w * aS[m] for m, w in c.items()) for c in contrasts])
        ensemble.append([sum(w * aE[m] for m, w in c.items()) for c in contrasts])
    return np.array(single), np.array(ensemble), skipped


def choose_winner(R):
    """Winner for Phase 2: highest mean VALIDATION (fold 9) macro-AUROC over the five seeds.

    Uses validation metrics only. Refuses to choose unless every model has exactly the
    pre-specified seeds and every value is finite, so a partial or mixed results folder
    can never produce a winner.
    """
    val_means = {}
    for n in MODELS:
        runs = R.get(n, {})
        if set(runs) != set(SEEDS):
            raise ValueError(f"model {n!r}: completed seeds {sorted(runs)} != pre-specified {sorted(SEEDS)}")
        v = np.array([runs[s]["val"]["macro"]["auroc"] for s in SEEDS], float)
        if not np.isfinite(v).all():
            raise ValueError(f"model {n!r}: non-finite validation AUROC")
        val_means[n] = float(v.mean())
    win = max(val_means, key=val_means.get)
    return dict(winner=win, rule="highest mean validation (fold 9) macro-AUROC over seeds",
                val_macro_auroc_means=val_means, models_considered=list(MODELS),
                seeds_used=list(SEEDS),
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))


def point_contrasts(Y, PS, contrasts):
    """Estimates on the ORIGINAL test set (no resampling). Returns (SINGLE, ENSEMBLE) arrays."""
    need = sorted({m for c in contrasts for m in c})
    aS = {m: np.mean([macro_auroc(Y, PS[m][k]) for k in range(PS[m].shape[0])]) for m in need}
    aE = {m: macro_auroc(Y, PS[m].mean(0)) for m in need}
    return (np.array([sum(w * aS[m] for m, w in c.items()) for c in contrasts]),
            np.array([sum(w * aE[m] for m, w in c.items()) for c in contrasts]))


def ci(d, level):
    a = (1 - level) / 2 * 100
    return np.percentile(d, a), np.percentile(d, 100 - a)


def fmt(point, d, bonf):
    """point: estimate on the original test set; d: bootstrap distribution (percentile CIs)."""
    lo, hi = ci(d, 0.95)
    blo, bhi = ci(d, bonf)
    return (f"{point:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", f"[{blo:+.4f}, {bhi:+.4f}]",
            bool(lo > 0 or hi < 0), bool(blo > 0 or bhi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--allow-partial", action="store_true",
                    help="development only: compute inferential tables even if some runs are missing")
    a = ap.parse_args()
    out = os.path.join(a.results, "summary")
    os.makedirs(out, exist_ok=True)
    R = load(a.results)

    # ---- per-model table
    trows, crows = [], []
    for name, runs in R.items():
        if not runs:
            continue
        r = list(runs.values())
        t = lambda k: [x["test"]["macro"][k] for x in r]
        trows.append({
            "model": name, "n_seeds": len(r), "params": r[0]["n_params"],
            "val_macro_auroc": ms([x["val"]["macro"]["auroc"] for x in r]),
            "test_macro_auroc": ms(t("auroc")), "test_macro_auprc": ms(t("auprc")),
            "test_macro_f1": ms(t("f1")), "test_macro_mcc": ms(t("mcc")),
            "test_macro_brier": ms(t("brier")),
            "epochs_run": f"{np.mean([x['epochs_run'] for x in r]):.0f}",
            "runs_at_epoch_cap": f"{sum(x['epochs_run'] >= CFG['max_epochs'] for x in r)}/{len(r)}",
            "wall_min_per_run": f"{np.mean([x['train_seconds'] for x in r]) / 60:.1f}",
            "infer_ms_per_record": f"{np.mean([x.get('infer_ms_per_record', np.nan) for x in r]):.3f}",
        })
        crow = {"model": name}
        for c in CLASSES:
            crow[c] = ms([x["test"]["per_class"][c]["auroc"] for x in r])
        crows.append(crow)
    T = pd.DataFrame(trows)
    T.to_csv(os.path.join(out, "table_models.csv"), index=False)
    open(os.path.join(out, "table_models.md"), "w").write(T.to_markdown(index=False))
    pd.DataFrame(crows).to_csv(os.path.join(out, "table_per_class.csv"), index=False)
    print(T.to_string(index=False))

    # ---- winner: validation only
    try:
        W = choose_winner(R)
        json.dump(W, open(os.path.join(out, "winner.json"), "w"), indent=1)
        print("\nWinner (validation only):", W["winner"])
    except ValueError as e:
        print("\nWinner NOT written:", e)

    # ---- contrasts (test fold)
    items = [(l, k, {A: 1, B: -1}, "comparisons") for l, A, B, k in COMPARISONS] + \
            [(l, "interaction", w, "interactions") for l, w in INTERACTIONS]
    items = [it for it in items if all(len(R[m]) == len(SEEDS) for m in it[2])]
    if not items:
        return
    Y = np.load(os.path.join(a.results, "y_test.npy"))
    pid = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    need = sorted({m for it in items for m in it[2]})
    PS = {m: np.stack([np.load(os.path.join(a.results, m, f"seed{s}", "preds_test.npy")) for s in SEEDS])
          for m in need}
    assert len(pid) == len(Y), "patient ids and labels differ in length"
    for m in need:   # every model must be scored on exactly the same test records, in the same order
        assert PS[m].shape == (len(SEEDS),) + Y.shape, (m, PS[m].shape, Y.shape)
    for m in need:   # the saved predictions must be the ones behind the saved metrics
        for k, sd in enumerate(SEEDS):
            gap = abs(macro_auroc(Y, PS[m][k]) - R[m][sd]["test"]["macro"]["auroc"])
            assert gap < 1e-6, f"{m} seed {sd}: preds_test.npy does not match metrics.json (gap {gap:.2e})"
    n_defined = len(COMPARISONS) + len(INTERACTIONS)
    complete = len(items) == n_defined
    if not complete:
        msg = (f"only {len(items)} of {n_defined} pre-specified contrasts can be computed (incomplete runs)")
        if not a.allow_partial:
            print(f"\nSTOP: {msg}. Finish all 30 runs, or use --allow-partial for a development preview.")
            return
        print(f"\nWARNING (development preview): {msg}. Bonferroni still uses the full pre-specified family sizes.")
    rng = np.random.default_rng(2026)
    PE_S, PE_E = point_contrasts(Y, PS, [it[2] for it in items])
    DS, DE, skipped = boot_contrasts(Y, PS, [it[2] for it in items], pid, a.boot, rng)
    json.dump(dict(n_bootstrap=a.boot, discarded_resamples=skipped,
                   n_test_patients=int(len(np.unique(pid))), n_test_records=int(len(pid)),
                   all_prespecified_contrasts_computed=complete),
              open(os.path.join(out, "bootstrap_log.json"), "w"), indent=1)
    print(f"\nbootstrap: {a.boot} resamples used, {skipped} discarded; "
          f"{len(np.unique(pid))} test patients / {len(pid)} records")

    fam_n = {"comparisons": len(COMPARISONS), "interactions": len(INTERACTIONS)}   # fixed in config, not data-dependent
    tables = {"comparisons": [], "interactions": []}
    for j, (label, kind, w, fam) in enumerate(items):
        bonf = 1 - 0.05 / fam_n[fam]
        per_seed = np.array([sum(wt * R[m][s]["test"]["macro"]["auroc"] for m, wt in w.items())
                             for s in SEEDS])
        sm, sc, sb, s0, s0b = fmt(PE_S[j], DS[:, j], bonf)
        em, ec, eb, e0, e0b = fmt(PE_E[j], DE[:, j], bonf)
        tables[fam].append({
            "contrast": label, "type": kind,
            "seed_diff_mean±sd": ms(per_seed), "seeds_positive": f"{int((per_seed > 0).sum())}/{len(SEEDS)}",
            "SINGLE_diff": sm, "SINGLE_CI95": sc, f"SINGLE_CI_bonf({bonf*100:.2f}%)": sb,
            "SINGLE_excl0_unadj": s0, "SINGLE_excl0_bonf": s0b,
            "ENSEMBLE_diff": em, "ENSEMBLE_CI95": ec, f"ENSEMBLE_CI_bonf({bonf*100:.2f}%)": eb,
            "ENSEMBLE_excl0_unadj": e0, "ENSEMBLE_excl0_bonf": e0b,
        })
    for fam, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"table_{fam}.csv"), index=False)
        if len(df):
            print(f"\n[{fam}]\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
