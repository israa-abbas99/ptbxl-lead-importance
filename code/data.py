"""PTB-XL loading, superclass labels, standardisation, and a synthetic generator.

Labels: 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label,
derived from scp_codes via scp_statements.csv (diagnostic == 1). Records with
no superclass label are removed (same convention as the PTB-XL benchmark).
Split: strat_fold 1-8 train, 9 validation, 10 test (official recommendation).
"""
import ast
import os

import numpy as np
import pandas as pd

from config import CFG, CLASSES
from guards import validate_cache_ids


def _superclass_matrix(df, scp_path):
    agg = pd.read_csv(scp_path, index_col=0)
    agg = agg[agg.diagnostic == 1]
    mapping = agg["diagnostic_class"].to_dict()
    Y = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for i, codes in enumerate(df["scp_codes"]):
        for code in codes:
            cls = mapping.get(code)
            if cls in CLASSES:
                Y[i, CLASSES.index(cls)] = 1.0
    return Y


def load_ptbxl(root, cache_dir):
    """root: folder containing ptbxl_database.csv, scp_statements.csv, records100/."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "ptbxl_100hz_cache.npz")
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    Y = _superclass_matrix(df, os.path.join(root, "scp_statements.csv"))

    if os.path.exists(cache):
        z = np.load(cache)
        validate_cache_ids(z["ecg_id"] if "ecg_id" in z.files else None, df.index.values)
        X = z["X"]
    else:
        import wfdb  # pip install wfdb
        X = np.zeros((len(df), CFG["n_leads"], CFG["n_samples"]), dtype=np.float32)
        for i, f in enumerate(df["filename_lr"]):
            sig, _ = wfdb.rdsamp(os.path.join(root, f))
            X[i] = sig.T.astype(np.float32)
            if i % 2000 == 0:
                print(f"  loaded {i}/{len(df)}", flush=True)
        np.savez_compressed(cache, X=X, ecg_id=df.index.values)

    keep = Y.sum(1) > 0
    return dict(
        X=X[keep], Y=Y[keep],
        fold=df["strat_fold"].values[keep],
        patient_id=df["patient_id"].values[keep],
        ecg_id=df.index.values[keep],
        n_total=int(len(df)), n_excluded_no_superclass=int((~keep).sum()),
    )


def make_synthetic(n=800, seed=0):
    """Learnable toy data with the PTB-XL shape (for smoke tests only)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, CFG["n_samples"], dtype=np.float32)
    Y = (rng.random((n, len(CLASSES))) < 0.3).astype(np.float32)
    Y[Y.sum(1) == 0, 0] = 1.0
    X = rng.normal(0, 0.5, (n, CFG["n_leads"], CFG["n_samples"])).astype(np.float32)
    for c in range(len(CLASSES)):
        wave = np.sin(2 * np.pi * (1.5 + c) * t).astype(np.float32)
        X[Y[:, c] == 1, c:c + 3, :] += wave
    return dict(
        X=X, Y=Y,
        fold=rng.integers(1, 11, n),
        patient_id=np.arange(n),
        ecg_id=np.arange(n),
        n_total=n, n_excluded_no_superclass=0,
    )


def dataset_report(data):
    """Counts per split + hard check that no patient appears in two splits."""
    f, pid = data["fold"], data["patient_id"]
    masks = dict(train=np.isin(f, CFG["train_folds"]), val=f == CFG["val_fold"], test=f == CFG["test_fold"])
    pats = {k: set(pid[m].tolist()) for k, m in masks.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (pats[a] & pats[b]), f"PATIENT LEAKAGE between {a} and {b}"
    rep = dict(n_records_total=data["n_total"],
               n_excluded_no_superclass=data["n_excluded_no_superclass"],
               n_records_used=int(len(f)), splits={})
    for k, m in masks.items():
        rep["splits"][k] = dict(
            records=int(m.sum()), patients=len(pats[k]),
            positives={c: int(data["Y"][m][:, i].sum()) for i, c in enumerate(CLASSES)},
            mean_labels_per_record=float(data["Y"][m].sum(1).mean()))
    return rep


def bandpass_filter(X, fs=None, low_hz=None, high_hz=None, order=None, chunk=2000):
    """Zero-phase Butterworth band-pass along the time axis (last axis).

    X: (N, leads, time). Applied record by record, so no information moves between
    records or splits. Second-order sections are used for numerical stability; the
    filter specification (order, band edges, zero-phase) is the one in config.py.
    Processed in chunks to limit memory; output is float32.
    """
    from scipy.signal import butter, sosfiltfilt
    fs = fs or CFG["sampling_rate"]
    low_hz = low_hz if low_hz is not None else CFG["bandpass"]["low_hz"]
    high_hz = high_hz if high_hz is not None else CFG["bandpass"]["high_hz"]
    order = order or CFG["bandpass"]["order"]
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = sosfiltfilt(sos, X[i:i + chunk].astype(np.float64), axis=-1)
    return out


def split_and_standardise(data):
    """Band-pass filter, then split. Per-lead mean/std come from TRAIN only (filtered signals)."""
    assert np.isfinite(data["X"]).all(), "NaN/inf found in signals; handle before training"
    X = bandpass_filter(data["X"])
    assert np.isfinite(X).all(), "NaN/inf after filtering"
    f = data["fold"]
    tr = np.isin(f, CFG["train_folds"])
    va = f == CFG["val_fold"]
    te = f == CFG["test_fold"]
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8

    def prep(m):
        return ((X[m] - mean) / std).astype(np.float32), data["Y"][m]

    return dict(
        train=prep(tr), val=prep(va), test=prep(te),
        test_patient_id=data["patient_id"][te],
        n=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
    )
