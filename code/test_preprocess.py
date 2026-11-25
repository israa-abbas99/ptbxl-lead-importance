"""Tests for the band-pass filter and standardisation (no torch). Run: python test_preprocess.py"""
import numpy as np

from config import CFG
from data import bandpass_filter, make_synthetic, split_and_standardise

fs = CFG["sampling_rate"]
t = np.arange(CFG["n_samples"]) / fs


def gain(freq):
    """Amplitude gain of the filter at one frequency (middle of the record, away from edges)."""
    x = np.sin(2 * np.pi * freq * t)[None, None, :].astype(np.float32)
    y = bandpass_filter(x)
    mid = slice(300, 700)
    return float(np.abs(y[0, 0, mid]).max() / np.abs(x[0, 0, mid]).max())


# 1) pass-band is kept; low and high frequencies are attenuated
assert gain(10.0) > 0.95, gain(10.0)
assert gain(0.1) < 0.1, gain(0.1)
assert gain(48.0) < 0.1, gain(48.0)

# 2) zero phase: a pulse inside the pass-band is not shifted
x = np.zeros((1, 1, CFG["n_samples"]), dtype=np.float32)
x[0, 0, 500] = 1.0
y = bandpass_filter(x)
assert int(np.argmax(np.abs(y[0, 0]))) == 500, "filter introduced a time shift"

# 3) shape, dtype, finiteness; each record is filtered independently
a = np.random.default_rng(0).normal(size=(5, 12, CFG["n_samples"])).astype(np.float32)
ya = bandpass_filter(a)
assert ya.shape == a.shape and ya.dtype == np.float32 and np.isfinite(ya).all()
assert np.allclose(bandpass_filter(a[:2]), ya[:2], atol=1e-5), "records are not independent"

# 4) standardisation uses train folds only and gives ~0 mean / ~1 SD per lead on train
d = make_synthetic(n=400, seed=1)
out = split_and_standardise(d)
Xtr = out["train"][0]
assert np.allclose(Xtr.mean(axis=(0, 2)), 0, atol=1e-3)
assert np.allclose(Xtr.std(axis=(0, 2)), 1, atol=1e-2)
assert np.isfinite(out["val"][0]).all() and np.isfinite(out["test"][0]).all()

print("test_preprocess: ALL PASSED")
