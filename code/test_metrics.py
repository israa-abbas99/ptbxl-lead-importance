"""Edge-case tests for metrics/thresholds/bootstrap (no torch). Run: python test_metrics.py"""
import numpy as np

from aggregate import boot_contrasts
from config import CLASSES
from metrics import choose_thresholds, evaluate, macro_auroc

rng = np.random.default_rng(0)
C = len(CLASSES)
n = 400
Y = (rng.random((n, C)) < 0.3).astype(np.float32)
Y[:5] = 1  # ensure positives

# 1) perfect scores -> AUROC = 1; constant scores -> 0.5
assert abs(macro_auroc(Y, Y) - 1.0) < 1e-9
assert abs(macro_auroc(Y, np.full_like(Y, 0.5)) - 0.5) < 1e-9

# 2) inverted scores -> AUROC = 0
assert abs(macro_auroc(Y, 1 - Y)) < 1e-9

# 3) a class with no positives must raise (never silently averaged)
Yz = Y.copy(); Yz[:, 2] = 0
try:
    macro_auroc(Yz, rng.random((n, C))); raise SystemExit("FAIL: no-positive class not detected")
except ValueError:
    pass

# 4) threshold search recovers a known optimum
P = np.where(Y == 1, 0.8, 0.2) + rng.normal(0, 0.01, Y.shape)
thr = choose_thresholds(Y, np.clip(P, 0, 1), (0.05, 0.95, 0.01))
assert np.all((thr > 0.2) & (thr < 0.8)), thr

# 5) evaluate() on perfect predictions
r = evaluate(Y, Y, np.full(C, 0.5))
assert abs(r["macro"]["auroc"] - 1) < 1e-9 and abs(r["macro"]["f1"] - 1) < 1e-9
assert abs(r["macro"]["brier"]) < 1e-9 and abs(r["macro"]["mcc"] - 1) < 1e-9

# 6) bootstrap: identical models -> contrast exactly 0; better model -> positive
pid = np.repeat(np.arange(n // 2), 2)          # 2 records per patient
good = np.clip(np.where(Y == 1, 0.7, 0.3) + rng.normal(0, 0.2, Y.shape), 0, 1)
bad = rng.random(Y.shape)
PS = {"a": np.stack([good, good]), "b": np.stack([good, good]), "c": np.stack([bad, bad])}
S, E, sk = boot_contrasts(Y, PS, [{"a": 1, "b": -1}, {"a": 1, "c": -1}], pid, 50, np.random.default_rng(1))
assert np.allclose(S[:, 0], 0) and np.allclose(E[:, 0], 0)
assert (S[:, 1] > 0).all() and (E[:, 1] > 0).all()

# 7) winner uses VALIDATION only, and refuses incomplete / wrong-seed / non-finite results
from aggregate import choose_winner
from config import MODELS, SEEDS


def fake_R(val_of, test_of):
    return {m: {s: {"val": {"macro": {"auroc": val_of(m)}}, "test": {"macro": {"auroc": test_of(m)}}}
                for s in SEEDS} for m in MODELS}


names = list(MODELS)
R = fake_R(lambda m: 0.90 + 0.01 * names.index(m), lambda m: 0.99 - 0.01 * names.index(m))
assert choose_winner(R)["winner"] == names[-1], "winner must follow validation, not test"

R_missing = fake_R(lambda m: 0.9, lambda m: 0.9)
del R_missing[names[0]][SEEDS[-1]]
R_wrong = fake_R(lambda m: 0.9, lambda m: 0.9)
R_wrong[names[0]][7] = R_wrong[names[0]].pop(SEEDS[-1])          # a different seed in its place
R_nan = fake_R(lambda m: 0.9, lambda m: 0.9)
R_nan[names[1]][SEEDS[0]]["val"]["macro"]["auroc"] = float("nan")
for bad_R in (R_missing, R_wrong, R_nan):
    try:
        choose_winner(bad_R); raise SystemExit("FAIL: invalid results accepted")
    except ValueError:
        pass

# 8) too many unusable bootstrap resamples -> abort (class with a single positive)
Yr = np.zeros((400, C), dtype=np.float32); Yr[::3] = 1; Yr[:, 1] = 0; Yr[0, 1] = 1
pr = np.stack([rng.random(Yr.shape)] * 2)
try:
    boot_contrasts(Yr, {"a": pr, "b": pr}, [{"a": 1, "b": -1}], np.repeat(np.arange(200), 2), 100,
                   np.random.default_rng(3)); raise SystemExit("FAIL: abort rule not triggered")
except RuntimeError:
    pass

# 9) point estimates come from the ORIGINAL test set; identical models -> exactly 0
from aggregate import point_contrasts
pe_s, pe_e = point_contrasts(Y, PS, [{"a": 1, "b": -1}, {"a": 1, "c": -1}])
assert abs(pe_s[0]) < 1e-12 and abs(pe_e[0]) < 1e-12 and pe_s[1] > 0 and pe_e[1] > 0
assert abs(pe_s[1] - (macro_auroc(Y, good) - macro_auroc(Y, bad))) < 1e-12

# 10) run-folder guards: a results folder is never silently reused with other data or settings
import os, tempfile
from guards import check_or_write_array, check_or_write_json, validate_cache_ids
with tempfile.TemporaryDirectory() as d:
    pj, pa = os.path.join(d, "config_used.json"), os.path.join(d, "y_test.npy")
    assert check_or_write_json(pj, dict(cfg={"lr": 1e-3, "g": (1, 2)}, seeds=[1]), keys=("cfg",))
    assert not check_or_write_json(pj, dict(cfg={"lr": 1e-3, "g": (1, 2)}, seeds=[2]), keys=("cfg",))  # seeds may differ
    for bad_cfg in (dict(cfg={"lr": 2e-3, "g": (1, 2)}, seeds=[1]),):
        try:
            check_or_write_json(pj, bad_cfg, keys=("cfg",)); raise AssertionError("FAIL: changed config accepted")
        except SystemExit:
            pass
    assert check_or_write_array(pa, Y)
    assert not check_or_write_array(pa, Y.copy())
    try:
        check_or_write_array(pa, Y[::-1]); raise AssertionError("FAIL: different labels accepted")
    except SystemExit:
        pass
validate_cache_ids(np.arange(5), np.arange(5))
validate_cache_ids(None, np.arange(5))                       # legacy cache: warning only
for bad_ids in (np.arange(5)[::-1], np.arange(6)):
    try:
        validate_cache_ids(bad_ids, np.arange(5)); raise AssertionError("FAIL: cache mismatch accepted")
    except SystemExit:
        pass

print("test_metrics: ALL PASSED (discarded resamples in test 6:", sk, ")")
