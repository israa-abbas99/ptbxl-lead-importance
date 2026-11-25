"""Figures for the Phase-1 results (no torch needed). Reads <results>/ and <results>/summary/ only.

  python make_figures.py --results /content/drive/MyDrive/ptbxl_results

Writes PNG (300 dpi) and PDF to <results>/figures/:
  fig_forest.png       the 7 pre-specified comparisons (SINGLE estimand) with 95% and Bonferroni CIs
  fig_per_class.png    per-class test AUROC, mean +- SD over 5 seeds, six models
  fig_curves.png       validation macro-AUROC per epoch (mean over seeds), six models
  fig_roc_pr.png       ROC and precision-recall curves of the winner (seed-averaged probabilities)

Colour = temporal module (none / LSTM / BiLSTM); line style or hatch = MHSA (no / yes).
Everything is also distinguishable in greyscale (markers, dashes, hatch).
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from config import CLASSES, MODELS, SEEDS

COL = {"none": "#2a78d6", "lstm": "#eb6834", "bilstm": "#1baf7a"}     # categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
LABEL = {"cnn": "CNN", "cnn_mhsa": "CNN + MHSA", "cnn_lstm": "CNN + LSTM", "cnn_lstm_mhsa": "CNN + LSTM + MHSA",
         "cnn_bilstm": "CNN + BiLSTM", "cnn_bilstm_mhsa": "CNN + BiLSTM + MHSA"}
MARK = {False: "o", True: "s"}


def style_of(m):
    t = "bilstm" if "bilstm" in m else "lstm" if "lstm" in m else "none"
    return COL[t], m.endswith("mhsa")


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight"})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"))
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def parse_ci(s):
    lo, hi = re.findall(r"[-+]?\d*\.\d+", s)
    return float(lo), float(hi)


def forest(summ, out):
    T = pd.read_csv(os.path.join(summ, "table_comparisons.csv"))
    bcol = [c for c in T.columns if c.startswith("SINGLE_CI_bonf")][0]
    n = len(T)
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * n + 1.3))
    y = np.arange(n)[::-1]
    for yi, (_, r) in zip(y, T.iterrows()):
        d = float(r["SINGLE_diff"])
        lo, hi = parse_ci(r["SINGLE_CI95"])
        blo, bhi = parse_ci(r[bcol])
        ax.plot([blo, bhi], [yi, yi], color=INK2, lw=1.0, alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.plot([lo, hi], [yi, yi], color=INK, lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot(d, yi, "o", ms=6, mfc=COL["none"], mec="white", mew=1.2, zorder=4)
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(T["contrast"], fontsize=8.5)
    ax.set_xlabel("Difference in test macro-AUROC (first model minus second)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], color=INK, lw=2.4, label="95% CI")
    ax.plot([], [], color=INK2, lw=1.0, alpha=0.55, label=f"Bonferroni CI ({bcol.split('(')[1][:-1]})")
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    ax.set_title("Pre-specified comparisons (mean over 5 seeds; patient-level bootstrap)", fontsize=9.5, loc="left")
    save(fig, out, "fig_forest")


def per_class(summ, out):
    T = pd.read_csv(os.path.join(summ, "table_per_class.csv")).set_index("model")
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    w = 0.13
    x = np.arange(len(CLASSES))
    for k, m in enumerate(MODELS):
        col, ms = style_of(m)
        mean = [float(T.loc[m, c].split("±")[0]) for c in CLASSES]
        sd = [float(T.loc[m, c].split("±")[1]) for c in CLASSES]
        ax.bar(x + (k - 2.5) * w, mean, w * 0.88, yerr=sd, color=col, edgecolor="white" if not ms else INK,
               linewidth=0.8, hatch="////" if ms else None, error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2),
               label=LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0.88, 0.96)
    ax.set_ylabel("Test AUROC (mean ± SD, 5 seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.set_title("Per-class AUROC (y-axis truncated at 0.88)", fontsize=9.5, loc="left")
    save(fig, out, "fig_per_class")


def curves(results, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for m in MODELS:
        logs = []
        for s in SEEDS:
            p = os.path.join(results, m, f"seed{s}", "metrics.json")
            if os.path.exists(p):
                logs.append([e["val_macro_auroc"] for e in json.load(open(p))["log"]])
        if not logs or not all(logs):
            continue
        L = max(map(len, logs))
        A = np.full((len(logs), L), np.nan)
        for i, l in enumerate(logs):
            A[i, :len(l)] = l
        cnt = (~np.isnan(A)).sum(0)
        mean = np.nanmean(A, 0)
        keep = cnt >= max(1, len(logs) // 2)          # draw only epochs reached by at least half of the runs
        col, ms = style_of(m)
        ax.plot(np.arange(1, L + 1)[keep], mean[keep], color=col, lw=1.8, ls="--" if ms else "-",
                label=LABEL[m])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-AUROC")
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title("Validation macro-AUROC per epoch (mean over seeds)", fontsize=9.5, loc="left")
    save(fig, out, "fig_curves")


def roc_pr(results, summ, out):
    win = json.load(open(os.path.join(summ, "winner.json")))["winner"]
    Y = np.load(os.path.join(results, "y_test.npy"))
    P = np.mean([np.load(os.path.join(results, win, f"seed{s}", "preds_test.npy")) for s in SEEDS], 0)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
    dashes = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
        pr, rc, _ = precision_recall_curve(Y[:, i], P[:, i])
        ax[0].plot(fpr, tpr, color=palette[i], ls=dashes[i], lw=1.6, label=c)
        ax[1].plot(rc, pr, color=palette[i], ls=dashes[i], lw=1.6, label=c)
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.7)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for a in ax:
        a.set_xlim(0, 1); a.set_ylim(0, 1.01)
    ax[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{LABEL[win]} on the test fold (probabilities averaged over 5 seeds)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_roc_pr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    summ = os.path.join(a.results, "summary")
    out = os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    setup()
    forest(summ, out)
    per_class(summ, out)
    curves(a.results, out)
    roc_pr(a.results, summ, out)


if __name__ == "__main__":
    main()
