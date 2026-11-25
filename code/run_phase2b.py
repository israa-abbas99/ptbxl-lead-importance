"""Run Phase 2B: 4 new models x 5 seeds (resumable), into the SAME results folder as Phase 1.

The Phase-1 folder must already contain the five cnn_lstm runs (M0); they are not re-trained.
Real:   python run_phase2b.py --root ... --cache /content/drive/MyDrive/ptbxl_cache --out /content/drive/MyDrive/ptbxl_results
Smoke:  python smoke_test.py   (makes ./results_smoke)   then
        python run_phase2b.py --synthetic --out ./results_smoke --epochs 2 --seeds 11 --models m4_mh_se
"""
import argparse
import json
import os

import torch

from config import CFG, SEEDS
from config_2b import ARCH, BASELINE, MODELS_2B
from data import dataset_report, load_ptbxl, make_synthetic, split_and_standardise
from guards import check_or_write_array, check_or_write_json
from train import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", required=True, help="the Phase-1 results folder")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS_2B))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke test only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.out, "y_test.npy")):
        raise SystemExit("STOP: no Phase-1 results (y_test.npy) in --out. Phase 2B must share the Phase-1 folder.")
    raw = make_synthetic() if a.synthetic else load_ptbxl(a.root, a.cache)
    rep = dataset_report(raw)
    check_or_write_json(os.path.join(a.out, "dataset_report.json"), rep)           # same data as Phase 1, or STOP
    data = split_and_standardise(raw)
    check_or_write_array(os.path.join(a.out, "test_patient_ids.npy"), data["test_patient_id"])
    check_or_write_array(os.path.join(a.out, "y_test.npy"), data["test"][1])
    check_or_write_json(os.path.join(a.out, "config_used_2b.json"),
                        dict(cfg=CFG, arch=ARCH, models=MODELS_2B, seeds=SEEDS, baseline=BASELINE,
                             synthetic=a.synthetic, epochs_override=a.epochs),
                        keys=("cfg", "arch", "models", "synthetic", "epochs_override"))
    base = [os.path.join(a.out, BASELINE, f"seed{s}", "metrics.json") for s in a.seeds]
    if not all(os.path.exists(p) for p in base):
        print("WARNING: Phase-1 baseline", BASELINE, "runs are missing for some seeds; comparisons need all five.")
    print("device:", a.device, "| split sizes:", data["n"], flush=True)

    for name in a.models:
        for seed in a.seeds:
            d = os.path.join(a.out, name, f"seed{seed}")
            if os.path.exists(os.path.join(d, "metrics.json")):
                print("skip (done):", name, seed); continue
            r = train_one(name, seed, data, d, a.device, max_epochs=a.epochs)
            print(f"DONE {name} seed {seed}: val {r['val']['macro']['auroc']:.4f} params {r['n_params']:,} "
                  f"({r['train_seconds']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
