"""Phase 2A, step 1 (GPU): lead-level perturbation and Integrated Gradients on the saved Phase-1 checkpoints.

No training. For every (model, seed) it loads results/<model>/seed<k>/model.pt, checks that the checkpoint
reproduces the saved test predictions, then computes (see PROTOCOL_PHASE2A.md):
  * zero-out of one lead at a time (validation + test predictions),
  * permutation of one lead across records (test, 3 repeats; sensitivity analysis),
  * Integrated Gradients per lead and class (fixed random subset of test records),
  * deletion curves (leads zeroed cumulatively in several orders; orders from validation or from IG, never from test labels).
Output: <out>/<model>/seed<k>/lead_perturb.npz   (resumable)

Real:   python lead_perturb.py --root /content/ptbxl_data/... --cache /content/drive/MyDrive/ptbxl_cache --results /content/drive/MyDrive/ptbxl_results
Smoke:  python lead_perturb.py --synthetic --results ./results_smoke --models cnn_lstm --seeds 11 --ig-n 20 --ig-steps 4
"""
import argparse
import os
import time

import numpy as np
import torch

from config import CFG, CLASSES, MODELS, SEEDS
from data import load_ptbxl, make_synthetic, split_and_standardise
from leadlib import (class_aurocs, permute_lead, random_orders, zero_leads)
from train import build, predict

N_RANDOM_ORDERS = 10
N_PERM_REPEATS = 3
SUBSET_SEED = 2026


def macro(Y, P):
    return float(class_aurocs(Y, P).mean())


@torch.no_grad()
def native_weights(model, X, Y, device, bs=256):
    """Built-in gate weights of Phase-2B models: mean over records (C,12) and, per class, mean over the records
    positive for that class (C,12). Returns None for models without gates (Phase 1)."""
    if not hasattr(model, "lead_weights"):
        return None
    A = np.concatenate([model.lead_weights(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy()
                        for i in range(0, len(X), bs)])                  # (N, C, 12)
    pos = np.stack([A[Y[:, c] > 0.5, c].mean(0) for c in range(A.shape[1])]) if Y is not None else None
    return A.mean(0), pos


def integrated_gradients(model, X, device, steps, bs=100):
    """Midpoint-rule IG on the logits, baseline = 0 (the train mean after z-scoring).
    Returns importance (12, C) = mean over records of sum_t |IG|, and the mean relative completeness error."""
    model.eval()
    C = len(CLASSES)
    imp = np.zeros((X.shape[1], C))
    rel_err = []
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    with torch.backends.cudnn.flags(enabled=False):          # cuDNN RNN backward needs train mode otherwise
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).to(device)
            grads = [torch.zeros_like(x) for _ in range(C)]
            for a in alphas:
                xa = (a.item() * x).requires_grad_(True)
                logits = model(xa)
                for c in range(C):
                    g, = torch.autograd.grad(logits[:, c].sum(), xa, retain_graph=(c < C - 1))
                    grads[c] += g
            with torch.no_grad():
                l1 = model(x); l0 = model(torch.zeros_like(x))
                for c in range(C):
                    ig = x * grads[c] / steps                                  # (b, 12, T)
                    imp[:, c] += ig.abs().sum(-1).sum(0).cpu().numpy()
                    gap = (ig.sum((1, 2)) - (l1[:, c] - l0[:, c])).abs()
                    scale = (l1[:, c] - l0[:, c]).abs().clamp_min(1e-3)
                    rel_err.append((gap / scale).cpu().numpy())
    return imp / len(X), float(np.median(np.concatenate(rel_err)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--cache", default="./cache")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="default: <results>/phase2a")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--ig-n", type=int, default=500)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "phase2a")
    os.makedirs(out, exist_ok=True)

    raw = make_synthetic() if a.synthetic else load_ptbxl(a.root, a.cache)
    data = split_and_standardise(raw)
    Xva, Yva = data["val"]; Xte, Yte = data["test"]
    # the data used here must be exactly the data of the Phase-1 runs
    y_saved = np.load(os.path.join(a.results, "y_test.npy"))
    p_saved = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    if not (y_saved.shape == Yte.shape and np.array_equal(y_saved, Yte) and np.array_equal(p_saved, data["test_patient_id"])):
        raise SystemExit("STOP: test labels / patient ids differ from the Phase-1 results folder (different data or cache).")
    n_leads = Xte.shape[1]
    ig_idx = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(Xte), size=min(a.ig_n, len(Xte)), replace=False))
    rand_orders = random_orders(N_RANDOM_ORDERS, n_leads, seed=SUBSET_SEED)
    print("device:", a.device, "| test records:", len(Xte), "| IG records:", len(ig_idx), flush=True)

    for name in a.models:
        for seed in a.seeds:
            d = os.path.join(a.results, name, f"seed{seed}")
            dst = os.path.join(out, name, f"seed{seed}", "lead_perturb.npz")
            if os.path.exists(dst):
                print("skip (done):", name, seed); continue
            t0 = time.time()
            model = build(name).to(a.device)
            model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=a.device))
            model.eval()
            P_base = predict(model, Xte, a.device)
            gap = float(np.abs(P_base - np.load(os.path.join(d, "preds_test.npy"))).max())
            if gap > 5e-3:
                raise SystemExit(f"STOP: {name} seed {seed}: checkpoint does not reproduce saved predictions (max gap {gap:.2e}).")
            Pb_va = predict(model, Xva, a.device)

            Pz_te = np.zeros((n_leads,) + P_base.shape, np.float32)
            drop_va = np.zeros((n_leads, len(CLASSES)))
            base_va = class_aurocs(Yva, Pb_va)
            for l in range(n_leads):
                Pz_te[l] = predict(model, zero_leads(Xte, [l]), a.device)
                drop_va[l] = base_va - class_aurocs(Yva, predict(model, zero_leads(Xva, [l]), a.device))

            base_te = class_aurocs(Yte, P_base)
            drop_perm = np.zeros((N_PERM_REPEATS, n_leads, len(CLASSES)))
            for r in range(N_PERM_REPEATS):
                rng = np.random.default_rng(SUBSET_SEED + 100 * r)
                for l in range(n_leads):
                    drop_perm[r, l] = base_te - class_aurocs(Yte, predict(model, permute_lead(Xte, l, rng), a.device))

            nat_te = native_weights(model, Xte, Yte, a.device)
            nat_va = native_weights(model, Xva, Yva, a.device)
            ig, ig_err = integrated_gradients(model, Xte[ig_idx], a.device, a.ig_steps)
            if ig_err > 0.1:
                print(f"  WARNING: IG completeness error {ig_err:.3f} (> 0.1); consider more steps", flush=True)

            # deletion curves: orders come from validation occlusion or from IG, never from test labels
            occ_val = drop_va.mean(1)
            ig_macro = (ig / ig.sum(0, keepdims=True)).mean(1)
            orders = {"occlusion_val_most_first": np.argsort(-occ_val),
                      "occlusion_val_least_first": np.argsort(occ_val),
                      "ig_most_first": np.argsort(-ig_macro)}
            if nat_va is not None:                       # built-in weights, ranked on the VALIDATION fold
                orders["native_val_most_first"] = np.argsort(-nat_va[0].mean(0))
            for i, o in enumerate(rand_orders):
                orders[f"random_{i}"] = o
            curves = np.zeros((len(orders), n_leads))
            for j, o in enumerate(orders.values()):
                curves[j, 0] = base_te.mean()
                for k in range(1, n_leads):
                    curves[j, k] = macro(Yte, predict(model, zero_leads(Xte, o[:k]), a.device))

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, P_base_test=P_base.astype(np.float32), P_zero_test=Pz_te, drop_zero_val=drop_va,
                                drop_perm_test=drop_perm, ig=ig, ig_idx=ig_idx, ig_completeness_err=ig_err,
                                del_curves=curves, del_names=np.array(list(orders)), del_orders=np.array(list(orders.values())),
                                ckpt_pred_gap=gap, has_native=nat_te is not None,
                                native_mean_test=nat_te[0] if nat_te else np.zeros((len(CLASSES), n_leads)),
                                native_pos_test=nat_te[1] if nat_te else np.zeros((len(CLASSES), n_leads)),
                                native_mean_val=nat_va[0] if nat_va else np.zeros((len(CLASSES), n_leads)))
            print(f"DONE {name} seed {seed}: gap {gap:.1e}, IG err {ig_err:.3f}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
