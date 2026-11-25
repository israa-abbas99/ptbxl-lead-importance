# ptbxl-lead-importance

Code, trained checkpoints and per-run outputs for the manuscript

> I. D. Abbas and A. N. Hashim, "Do Built-In Lead Weights Explain Multi-Label 12-Lead ECG Classifiers? A Pre-Specified Multi-Seed Study on PTB-XL", *International Journal of Intelligent Engineering and Systems* (under review, manuscript 20267841).

Everything needed to rerun the experiments is here: the versioned protocols, the data pipeline, the model definitions, the training and analysis scripts, the 60 trained networks, and the files every table and figure was built from.

---

## 1. Requirements

- Python 3.10 or newer (the revision analyses ran on Python 3.13.16)
- PyTorch 2.x with CUDA for training and for any script that loads a checkpoint
- Everything else: `pip install -r requirements.txt`

Exact versions of the revision session are stored in `results_summary/normalisation_and_versions.json`.

## 2. Data

PTB-XL v1.0.3 is not redistributed here. Download it from PhysioNet:
<https://physionet.org/content/ptb-xl/1.0.3/>

The scripts need the folder that holds `ptbxl_database.csv`, `scp_statements.csv` and `records100/` (100 Hz signals). Pass it as `--root`. The first run reads the signals once and stores them in `--cache`.

Fixed choices (see `code/config.py` and `code/data.py`):

| Item | Setting |
| --- | --- |
| Split | strat_fold 1–8 train, 9 validation, 10 test (2,158 test records) |
| Labels | five diagnostic superclasses: NORM, MI, STTC, CD, HYP |
| Filter | zero-phase Butterworth band-pass, 0.5–40 Hz, order 4 |
| Normalisation | z-score per lead with training-fold mean and SD (values in `results_summary/normalisation_and_versions.json`) |
| Seeds | 11, 22, 33, 44, 55 |
| Bootstrap | patients resampled together, generator seed 2026 |

## 3. Repository layout

```
code/               all scripts and the three frozen protocols (PROTOCOL_PHASE1/2A/2B.md)
revision_extra/     small CPU scripts added during the first review (read saved outputs only)
results/            trained checkpoints and every per-run output (see section 6)
results_summary/    per-seed results table, normalisation constants, software versions, model selection file
```

Run every command from inside `code/`.

## 4. Quick check (no data, a few minutes)

```bash
cd code
python test_metrics.py && python test_preprocess.py && python test_leads.py && python test_2b_pipeline.py
python test_models_2b.py      # needs torch
python smoke_test.py          # synthetic signals, ends with SMOKE TEST PASSED
```

## 5. Reproducing the study

Set three paths first:

```bash
ROOT=/path/to/ptb-xl-1.0.3      # PTB-XL folder
CACHE=./cache                   # created on first use
RES=../results                  # results folder of this repository
```

All training and perturbation scripts are resumable: rerun the same command after an interruption and finished runs are skipped. Before any analysis, each script that loads a checkpoint checks that it reproduces the saved test predictions and stops otherwise.

### Phase 1: six temporal architectures (30 runs)

```bash
python run_phase1.py --root $ROOT --cache $CACHE --out $RES
python aggregate.py  --results $RES --boot 5000     # tables, 7 comparisons, winner.json (validation only)
python make_tables.py  --results $RES
python make_figures.py --results $RES
```

### Phase 2A: lead importance of the 30 Phase 1 networks (no training)

```bash
python lead_perturb.py  --root $ROOT --cache $CACHE --results $RES
python analyze_leads.py --results $RES --boot 1000
python ig_sensitivity.py --root $ROOT --cache $CACHE --results $RES   # 128-step IG and validation-derived IG order
```

### Phase 2B: four lead-aware models (20 runs) and their built-in weights

```bash
python run_phase2b.py   --root $ROOT --cache $CACHE --out $RES
python aggregate_2b.py  --results $RES --boot 5000
python lead_perturb.py  --root $ROOT --cache $CACHE --results $RES --models m1_early_se m2_separate m3_cs_se m4_mh_se
python analyze_native.py   --results $RES
python extra_crossmodel.py --results $RES
python make_figures_2b.py  --results $RES
```

`run_phase2_all.sh` runs Phases 2A and 2B in one go.

### Revision analyses (first review)

None of these was pre-specified; the manuscript labels them as such. They run in full 32-bit arithmetic (TF32 disabled).

```bash
python revision_r1.py constants --root $ROOT --cache $CACHE --results $RES   # normalisation constants, versions
python revision_r1.py tied      --root $ROOT --cache $CACHE --results $RES   # M4-tied control (5 new runs)
python revision_r1.py baseline  --root $ROOT --cache $CACHE --results $RES   # resnet1d_wang (5 new runs) + Phase 2A analysis
python revision_r1.py subsets   --root $ROOT --cache $CACHE --results $RES   # all 220 / 924 lead subsets at k = 3 / 6
python revision_r1.py groups    --root $ROOT --cache $CACHE --results $RES   # lead-group zeroing and joint replacement
python revision_r1.py mi        --root $ROOT --cache $CACHE --results $RES   # MI split into AMI and IMI
python revision_r1.py summary   --results $RES                               # prints all revision tables
```

Then, from `revision_extra/` (CPU only, reads `../results` or the folder in the `RESULTS` environment variable):

| Script | Output |
| --- | --- |
| `rev_local.py` | descriptive M3 vs M0 contrast, paired seed SDs |
| `b1_check.py` | bootstrap distribution of B1 stability (explains the M3 interval) |
| `norm_flat.py` | per-class spread of lead importance (NORM profile) |
| `fig3_redraw.py` | Fig. 3 with the validation-derived IG order |
| `fig8.py` | Fig. 8, per-record M4 gate weights |
| `gate_dist.py` | larger version of the per-record gate plot |

## 6. Where each result comes from

| Manuscript item | Script | File(s) in `results/` |
| --- | --- | --- |
| Table I (split counts) | `run_phase1.py` | `dataset_report.json` |
| Tables III–IV (Phase 1) | `aggregate.py`, `make_tables.py` | `summary/` |
| Model selection | `aggregate.py` | `summary/winner.json` |
| Fig. 2, Table V, Fig. 3 (Phase 2A) | `analyze_leads.py`, `ig_sensitivity.py`, `revision_extra/fig3_redraw.py` | `phase2a/summary/`, `ig_sens/` |
| Tables VI–VIII, Fig. 4 (Phase 2B accuracy) | `aggregate_2b.py`, `make_figures_2b.py` | `summary_2b/`, `figures_2b/` |
| Figs. 5–7, Tables IX–XI (built-in weights) | `analyze_native.py`, `extra_crossmodel.py`, `make_figures_2b.py` | `phase2b/`, `extra/`, `figures_2b/` |
| B1 intervals, Fig. 8 | `revision_extra/b1_check.py`, `revision_extra/fig8.py` | `b1_ci/` |
| Exhaustive deletion, lead groups, M4-tied, ResNet, AMI/IMI | `revision_r1.py` | `revision/` |

Each run folder `results/<model>/seed<k>/` holds `model.pt`, the saved validation and test predictions (`preds_val.npy`, `preds_test.npy`) and `metrics.json`. The configurations used are in `results/config_used.json` and `results/config_used_2b.json`; test labels and patient identifiers in `y_test.npy` and `test_patient_ids.npy`. `results_summary/per_seed_results.csv` lists the test macro-AUROC and per-class AUROC of every run.

Model names in the code: `cnn`, `cnn_lstm`, `cnn_bilstm`, `cnn_mhsa`, `cnn_lstm_mhsa`, `cnn_bilstm_mhsa` (Phase 1); `m1_early_se`, `m2_separate`, `m3_cs_se`, `m4_mh_se` (Phase 2B); `m4t_tied`, `resnet1d_wang` (revision).

## 7. Notes

- Model selection used the validation fold only. The full-precision validation means are in `winner.json` (CNN-LSTM 0.932947, CNN-BiLSTM 0.932942).
- Do not reuse one results folder for runs with different settings; the scripts stop with a `STOP` message when a saved configuration differs.
- Results on other GPUs can differ in the last digits. Checkpoints trained under TF32 (the ResNet baseline) reproduce their saved predictions to within about 7e-4; all others to within 6e-6.
- PTB-XL is public and pseudonymised. Saved predictions are indexed by PTB-XL record identifiers only.

## 8. Citation

If you use this code, please cite the manuscript above. The citation will be updated with volume and pages on publication.

## 9. License

Code: MIT (see `LICENSE`). PTB-XL is distributed by PhysioNet under its own license (CC BY 4.0).
