"""Figures for Phase 2B (no torch). Reads only the summary tables.
  python make_figures_2b.py --results /content/drive/MyDrive/ptbxl_results
Writes PNG (300 dpi) + PDF to <results>/figures_2b/:
  fig2b_forest      the 5 pre-specified comparisons (SINGLE estimand; 95% and Bonferroni 99% intervals)
  fig2b_weights     built-in lead weights (mean over test records and 5 seeds) of the three gated models
  fig2b_stability   seed stability of the built-in weights (mean pairwise Spearman) against the random reference
  fig2b_vs_zeroout  (needs table_native_zeroout_*.csv from analyze_native.py) built-in weights vs zero-out importance, M4
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASSES
from config_2b import LABEL_2B
from leadlib import LEADS, spearman
from make_figures import GRID, INK, INK2, save, setup

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def mean_of(cell):
    return float(str(cell).split("±")[0])


def sd_of(cell):
    return float(str(cell).split("±")[1])


def weights_matrix(path):
    d = pd.read_csv(path, index_col=0)
    return d.map(mean_of).values, d.map(sd_of).values


def forest(res, out):
    d = pd.read_csv(os.path.join(res, "summary_2b", "table_comparisons_2b.csv"))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, r in d.iterrows():
        pt = float(r["SINGLE_diff"])
        lo, hi = map(float, re.findall(r"[-+]?\d*\.\d+", r["SINGLE_CI95"]))
        blo, bhi = map(float, re.findall(r"[-+]?\d*\.\d+", r["SINGLE_CI_bonf(99%)"]))
        sig = bool(r["SINGLE_excl0_bonf"])
        col = "#d62728" if sig else INK2
        ax.plot([blo, bhi], [i, i], color=col, lw=1.2, alpha=0.5)
        ax.plot([lo, hi], [i, i], color=col, lw=3)
        ax.plot(pt, i, "o", color=col, ms=6)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(["Early-Fusion SE vs CNN-LSTM", "Separate-Lead vs CNN-LSTM", "Channel-Specific SE vs Separate-Lead",
                        "Multi-Head SE vs Channel-Specific SE", "Multi-Head SE vs Separate-Lead"], fontsize=8.5)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Difference in test macro-AUROC (mean over 5 seeds; thick = 95% CI, thin = Bonferroni 99% CI)")
    ax.set_title("Phase 2B: effect on performance (red: Bonferroni interval excludes 0)", fontsize=9.5, loc="left")
    save(fig, out, "fig2b_forest")


def weights_fig(res, out):
    pdir = os.path.join(res, "phase2b")
    W1, _ = weights_matrix(os.path.join(pdir, "table_native_weights_m1_early_se.csv"))
    W3, _ = weights_matrix(os.path.join(pdir, "table_native_weights_m3_cs_se.csv"))
    W4, S4 = weights_matrix(os.path.join(pdir, "table_native_weights_m4_mh_se.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1, 5]})
    for ax, M, title, cls in ((axs[0], W1[:, :1], "Early-Fusion SE\n(per lead)", [""]),
                              (axs[1], W3[:, :1], "Channel-Specific\nSE (per lead)", [""]),
                              (axs[2], W4, "Multi-Head SE (one gate per class)", CLASSES)):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.grid(False); ax.set_title(title, fontsize=9, loc="center" if len(cls)==1 else "left")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls)
        ax.set_yticks(range(12)); ax.set_yticklabels(LEADS if ax is axs[0] else [""] * 12)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=axs, label="Gate weight (0 = suppress, 1 = pass)", shrink=0.8, pad=0.02)
    fig.suptitle("Built-in lead weights (mean over test records and 5 seeds)", fontsize=10, y=1.04, x=0.01, ha="left")
    save(fig, out, "fig2b_weights")


def stability_fig(res, out):
    T = pd.read_csv(os.path.join(res, "phase2b", "table_native_stability.csv")).set_index("model")
    ref = pd.read_csv(os.path.join(res, "phase2a", "summary", "table_stability.csv")).set_index("model").loc["cnn_lstm", "stability_macro"]
    ref_v = float(ref.split()[0]); null = float(T["null95"].iloc[0])
    rows = []
    for m, lab in (("m1_early_se", "Early-Fusion SE"), ("m3_cs_se", "Channel-Specific SE"), ("m4_mh_se", "Multi-Head SE (class average)")):
        s = T.loc[m, "stability_class_avg"]
        v, lo, hi = map(float, re.findall(r"[-+]?\d*\.\d+", s))
        rows.append((lab, v, lo, hi, ORANGE))
    for c in CLASSES:
        rows.append((f"Multi-Head SE: {c}", float(T.loc["m4_mh_se", f"stab_{c}"]), None, None, GREEN))
    rows.append(("M0 CNN-LSTM zero-out (Phase 2A, reference)", ref_v, None, None, INK2))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for i, (lab, v, lo, hi, col) in enumerate(rows):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=col, lw=2, alpha=0.5)
        ax.plot(v, i, "o", color=col, ms=7)
    ax.axvline(null, color=INK2, ls="--", lw=1); ax.text(null, -0.9, " random rankings, 95th pct", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seed stability: mean pairwise Spearman over 5 seeds (line = range of the 10 pairs)")
    ax.set_title("How reproducible are the built-in lead weights?", fontsize=9.5, loc="left")
    save(fig, out, "fig2b_stability")


def vs_zero_fig(res, out):
    pdir = os.path.join(res, "phase2b")
    zp = os.path.join(pdir, "table_native_zeroout_m4_mh_se.csv")
    if not os.path.exists(zp):
        print("skip fig2b_vs_zeroout: rerun analyze_native.py to create table_native_zeroout_*.csv"); return
    W4, _ = weights_matrix(os.path.join(pdir, "table_native_weights_m4_mh_se.csv"))
    Z = pd.read_csv(zp, index_col=0)[CLASSES].map(mean_of).values
    rho = [spearman(W4[:, j], Z[:, j]) for j in range(5)]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, M, t, cmap, vmax in ((axs[0], W4, "Built-in gate weight", "Blues", 1.0),
                                 (axs[1], Z, "AUROC drop when the lead is zeroed", "Oranges", float(Z.max()))):
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{c}\nrho={r:+.2f}" for c, r in zip(CLASSES, rho)] if ax is axs[0] else CLASSES, fontsize=8)
        ax.set_yticks(range(12)); ax.set_yticklabels(LEADS if ax is axs[0] else [""] * 12)
        ax.set_title(t, fontsize=9, loc="left")
        for i in range(12):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}" if ax is axs[0] else f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 * vmax else INK)
    fig.suptitle("Multi-Head SE: what the gate says vs what zeroing a lead does (mean of 5 seeds; rho = Spearman per class)", fontsize=9.5, x=0.01, ha="left")
    save(fig, out, "fig2b_vs_zeroout")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args()
    out = os.path.join(a.results, "figures_2b"); os.makedirs(out, exist_ok=True)
    setup(); forest(a.results, out); weights_fig(a.results, out); stability_fig(a.results, out); vs_zero_fig(a.results, out)
