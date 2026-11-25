# Phase 1 Protocol (v5, to be frozen before the first real run)

Status: **DRAFT until Esraa and the supervisor confirm the items marked `CONFIRM`.**
After confirmation `config.py` is frozen. Any later change is logged in section 13 and all affected runs are repeated.

## 1. Question
Does Multi-Head Self-Attention (MHSA) improve multi-label ECG classification, and does its effect depend on the temporal back-end (none, LSTM, BiLSTM)? The selected configuration becomes the baseline of Phase 2.

## 2. Design (2 x 3 factorial, 6 models, 5 seeds = 30 runs)
| # | Model | Temporal | MHSA |
|---|---|---|---|
| 1 | cnn | none | no |
| 2 | cnn_mhsa | none | yes |
| 3 | cnn_lstm | LSTM | no |
| 4 | cnn_lstm_mhsa | LSTM | yes |
| 5 | cnn_bilstm | BiLSTM | no |
| 6 | cnn_bilstm_mhsa | BiLSTM | yes |

Seeds: 11, 22, 33, 44, 55 for every model. Sequence of every model: ECG -> CNN -> [LSTM|BiLSTM] -> [MHSA] -> mean pooling -> classifier. MHSA always comes after the temporal module and before pooling.

## 3. Pre-specified comparisons (the only ones reported as findings)
Ablations (one factor changes): MHSA on CNN, on CNN+LSTM, on CNN+BiLSTM; LSTM added without MHSA and with MHSA.
Controlled architecture comparisons (not pure ablation): BiLSTM vs LSTM without and with MHSA.
Interaction analysis (is the MHSA effect different across back-ends?): three contrasts of MHSA effects (LSTM-CNN, BiLSTM-CNN, BiLSTM-LSTM).
Ensembling ("voting") is a separate secondary analysis and never used for selection. `CONFIRM`

## 4. Data
PTB-XL v1.0.3, 100 Hz, 12 leads, 10 s (1000 samples). Five diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label, derived from `scp_codes` with `scp_statements.csv` (`diagnostic == 1`, column `diagnostic_class`); a record receives every superclass of any of its codes. Records without any of the five labels are excluded; their number is reported, and absence of a label is not interpreted as "normal".
Split: `strat_fold` 1-8 train, 9 validation, 10 test. Fold 10 is used for no decision.
The code writes `dataset_report.json` (records and patients per split, positives per class, excluded records) and **asserts that no patient appears in two splits**.

## 5. Preprocessing
1. Each record is band-pass filtered with a fourth-order zero-phase Butterworth filter (0.5-40 Hz; `sosfiltfilt`, second-order sections) along the time axis. The cut-offs follow common practice in ECG analysis and the pipeline of the earlier experiments; they are fixed before training and are never tuned on validation or test data. Filtering is applied record by record, so nothing moves between records or splits.
2. Per-lead z-score on the filtered signals: z[l,t] = (x[l,t] - mu_l) / sigma_l, where mu_l and sigma_l are the mean and SD of lead l over all time points of all records in training folds 1-8; the same constants are applied to folds 9 and 10.
3. No augmentation, no resampling beyond the 100 Hz release (same for all models). `CONFIRM`
Limitation to report: the 0.5 Hz lower cut-off may attenuate low-frequency components near the ST segment; the effect of filtering on the model comparison is not evaluated in this study, so conclusions hold under this preprocessing pipeline.

## 6. Architecture
CNN: Conv1d(12,64,k7)-BN-ReLU-MaxPool2, Conv1d(64,128,k5)-BN-ReLU-MaxPool2, Conv1d(128,128,k5)-BN-ReLU-MaxPool2 -> 125 steps x 128. `CONFIRM`
Temporal: LSTM(128, hidden 128) or BiLSTM(128, hidden 64 per direction): output width 128 in all variants (width matched, no parameter matching; counts reported).
MHSA: 4 heads, d = 128, dropout 0.1, sinusoidal positional encoding, residual connection + LayerNorm, identical in models 2, 4, 6.
Head: mean pooling over time -> Dropout 0.2 -> Linear(128, 5), sigmoid outputs.

## 7. Training
BCEWithLogitsLoss, no class weights (for any model) `CONFIRM`; AdamW (lr 1e-3, weight decay 1e-4); batch 128; **max 100 epochs** with early stopping (patience 8) and ReduceLROnPlateau on validation macro-AUROC (factor 0.5, patience 3); gradient clipping 1.0. Identical for all models, no per-model tuning.
Pilot: one seed (not in the final list, e.g. 7) of `cnn` and `cnn_bilstm_mhsa` to check run time and that early stopping ends before epoch 100. Decisions from the pilot use training and validation curves only; if the cap is hit, it is raised and logged in section 13.

## 8. Evaluation
Primary: **macro-AUROC** (AUROC per class, then unweighted mean of the five classes). Secondary: macro-AUPRC (average precision per class as computed by `average_precision_score`, then the unweighted mean of the five classes; called average precision in the paper), macro-F1, MCC, Brier, per-class AUROC/AUPRC/sensitivity/specificity, trainable parameters, wall-clock time per run (start of training to the end of the final evaluation; includes the per-epoch validation passes), inference time per record (measured separately).
Thresholds: per class, F1-optimal on fold 9 only, then frozen. Reported checkpoint: best validation macro-AUROC. Final performance and confidence intervals are on fold 10, evaluated once per run.

## 9. Selection rule for Phase 2
Highest mean validation (fold 9) macro-AUROC over five seeds, written to `summary/winner.json`; fixed before fold 10 results are inspected and never revised afterwards. No parsimony rule is used for selection (the SD across seeds measures training variability, not uncertainty of a performance difference). Parameters, training time and inference time are reported and discussed next to the result, and a simpler model that performs equally well is an acceptable and reportable outcome. `CONFIRM`

## 10. Statistics
Mean +/- SD over seeds describes training randomness. Test-sample uncertainty is assessed with a patient-level bootstrap (1000 resamples; all records of a patient are resampled together, and the same resampled patients are used for every model, so each difference is paired on patients). Two estimands are reported for every contrast:
- **Primary (SINGLE):** mean over the five seeds of the single-model macro-AUROC difference. This matches the mean +/- SD in the model table and describes an individual trained network.
- **Secondary (ENSEMBLE):** difference in macro-AUROC of the seed-averaged probabilities; it describes a different quantity (the 5-seed ensemble of each model), is not used for model selection, and its interval width is not assumed to be narrower or wider. `CONFIRM`
Each is given with a 95% CI and a Bonferroni-adjusted CI within its family (7 comparisons; 3 interaction contrasts); adjusted versus unadjusted is always stated. The pre-specified families are fixed in `config.py`. Seeds are repeated trainings, not independent patient samples.
Bootstrap size: 1000 resamples during development, **5000 for the final analysis**; the number of test patients and records is reported. A resample in which any class has no positives or no negatives is discarded and redrawn; the number discarded is reported (`bootstrap_log.json`) and the analysis aborts if more than 1% of draws are discarded. Metric code is covered by `test_metrics.py` (perfect/constant/inverted scores, missing class, thresholds, paired bootstrap), which must pass before any real run.

## 11. Reproducibility
Seeds fixed for Python, NumPy, PyTorch; cuDNN deterministic. Bitwise identity on GPU is not guaranteed. Software versions and GPU are recorded in the run log.

## 12. Reporting
All six models are reported whatever the outcome, including per-class results and any model where MHSA or the recurrent module does not help.

## 13. Deviations log
(empty)
Note (analysis and bookkeeping code, before any real result was aggregated; model, data-processing and training code unchanged): `aggregate.py` was hardened: explicit check for unusable bootstrap resamples (abort above 1% of requested draws); Bonferroni family sizes fixed from `config.py` (7 and 3) instead of counted from available results; point estimates are computed on the original test set (the bootstrap supplies percentile CIs only); the inferential tables are refused unless all 30 runs exist; each saved prediction file is checked against its metrics.json; the winner is written only when every model has exactly the pre-specified seeds with finite validation values (winner.json also records models, seeds and creation time). `run_phase1.py` refuses to reuse a results folder whose dataset report, test labels, patient ids or settings differ from the current run; the signal cache stores record ids and is rejected if they differ from the metadata file.
Note: the pilot run (seed 7, models cnn and cnn_bilstm_mhsa) was done before the band-pass filter was adopted (no filtering). It was a pipeline and run-time check only; its results are not used anywhere.
