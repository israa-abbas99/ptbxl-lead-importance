#!/usr/bin/env bash
# Phase 2 (2A + 2B) in one go. Resumable: finished runs are skipped, so you can re-run after a Colab disconnect.
# Usage (Colab):  bash run_phase2_all.sh            # full run
#                 bash run_phase2_all.sh smoke      # quick synthetic check only (a few minutes)
# Paths come from the environment (ROOT, CACHE, RES) or the defaults below.
set -euo pipefail

ROOT="${ROOT:-/content/ptbxl_data}"                     # folder with ptbxl_database.csv, scp_statements.csv, records100/
CACHE="${CACHE:-/content/drive/MyDrive/ptbxl_cache}"
RES="${RES:-/content/drive/MyDrive/ptbxl_results}"      # the Phase-1 results folder (must contain cnn_lstm runs + summary/winner.json)
export ROOT CACHE RES

cd "$(dirname "$0")"
pip -q install tabulate wfdb >/dev/null 2>&1 || true

echo "== 1. tests =="
python test_metrics.py
python test_preprocess.py
python test_leads.py
python test_2b_pipeline.py
python test_models_2b.py

if [ "${1:-}" = "smoke" ]; then
  echo "== smoke (synthetic data, ./results_smoke) =="
  python smoke_test.py
  python lead_perturb.py --synthetic --results ./results_smoke --models cnn_lstm --seeds 11 --ig-n 20 --ig-steps 4
  python run_phase2b.py --synthetic --out ./results_smoke --epochs 2 --seeds 11 --models m1_early_se m2_separate m3_cs_se m4_mh_se
  python lead_perturb.py --synthetic --results ./results_smoke --models m1_early_se m2_separate m3_cs_se m4_mh_se --seeds 11 --ig-n 20 --ig-steps 4
  echo "SMOKE OK"; exit 0
fi

echo "== 2A. lead perturbation of the 30 Phase-1 models =="
python lead_perturb.py --root "$ROOT" --cache "$CACHE" --results "$RES"
python analyze_leads.py --results "$RES" --boot 1000

echo "== 2B. train the 4 new models (20 runs) =="
python run_phase2b.py --root "$ROOT" --cache "$CACHE" --out "$RES"
python aggregate_2b.py --results "$RES" --boot 5000

echo "== 2B. built-in lead weights and explanation analysis =="
python lead_perturb.py --root "$ROOT" --cache "$CACHE" --results "$RES" --models m1_early_se m2_separate m3_cs_se m4_mh_se
python analyze_native.py --results "$RES"

echo "DONE. Send me: $RES/phase2a/summary, $RES/summary_2b, $RES/phase2b"
