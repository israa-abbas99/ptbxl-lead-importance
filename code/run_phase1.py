"""Run Phase 1: 6 models x 5 seeds (resumable).

Real run:   python run_phase1.py --root /path/to/ptbxl --cache ./cache --out ./results
Smoke test: python run_phase1.py --synthetic --out ./results_smoke --epochs 2 --seeds 11
"""
import argparse
import json
import os

import numpy as np
import torch

from config import CFG, MODELS, SEEDS
from guards import check_or_write_array, check_or_write_json
from data import dataset_report, load_ptbxl, make_synthetic, split_and_standardise
from train import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="PTB-XL folder (ptbxl_database.csv, scp_statements.csv, records100/)")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke test only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    raw = make_synthetic() if a.synthetic else load_ptbxl(a.root, a.cache)
    rep = dataset_report(raw)          # also asserts: no patient in two splits
    check_or_write_json(os.path.join(a.out, "dataset_report.json"), rep)   # stops if data differ from earlier runs here
    print(json.dumps(rep, indent=1))
    data = split_and_standardise(raw)
    print("split sizes:", data["n"], "| device:", a.device)

    check_or_write_array(os.path.join(a.out, "test_patient_ids.npy"), data["test_patient_id"])
    check_or_write_array(os.path.join(a.out, "y_test.npy"), data["test"][1])
    check_or_write_json(os.path.join(a.out, "config_used.json"),
                        dict(cfg=CFG, models=MODELS, seeds=a.seeds, synthetic=a.synthetic,
                             epochs_override=a.epochs),
                        keys=("cfg", "models", "synthetic", "epochs_override"))

    for name in a.models:
        for seed in a.seeds:
            d = os.path.join(a.out, name, f"seed{seed}")
            if os.path.exists(os.path.join(d, "metrics.json")):
                print("skip (done):", name, seed)
                continue
            r = train_one(name, seed, data, d, a.device, max_epochs=a.epochs)
            print(f"DONE {name} seed {seed}: val {r['val']['macro']['auroc']:.4f} "
                  f"params {r['n_params']:,} ({r['train_seconds']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
