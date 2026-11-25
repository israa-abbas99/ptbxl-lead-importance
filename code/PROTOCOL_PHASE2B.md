# Phase 2B Protocol: lead-aware models and their built-in lead weights (v1, fixed before any 2B result is seen)

Status: **new models are trained once under the frozen Phase-1 protocol; nothing is tuned.** The old Keras models and numbers of the original manuscript are not used. The four architectures below are defined here and implemented again in PyTorch (`models_lead.py`), so results from this phase replace the earlier ones.

## 1. Questions
1. **Performance.** Does separating the 12 leads, and does adding SE gates (shared or class-specific), change test macro-AUROC relative to the Phase-1 winner (M0, CNN+LSTM)?
2. **Stability.** Are the built-in lead weights of the same model similar across five training seeds?
3. **Faithfulness.** Do the built-in lead weights agree with what happens when a lead is removed (zero-out), and with Integrated Gradients? Does deleting the leads they rank highest hurt more than deleting random leads?
4. **Class specificity.** Do class-specific gates (M4) differ between classes by more than they differ between seeds?

## 2. Models (everything not stated here is identical to Phase 1: data, losses, optimiser, early stopping, thresholds)
Notation: x is a (12, 1000) z-scored record; D = 32; sigma = sigmoid.

- **SE gate** `SE(s)`: s in R^12, a = sigma(W2 ReLU(W1 s + b1) + b2) in (0,1)^12, W1: 12->3, W2: 3->12 (default PyTorch initialisation).
- **Lead encoder** (M2-M4): 12 fully independent encoders, none sharing weights. Each is Conv(1->16,k7)-BN-ReLU-MaxPool2, Conv(16->32,k5)-..., Conv(32->32,k5)-..., then its own LSTM(32,32); r_l = mean over time of the LSTM output. Implemented with grouped convolutions (groups = 12) and one LSTM per lead. r = (r_1..r_12) in R^(12 x 32).

| Model | Definition |
|---|---|
| **M0** CNN+LSTM (Phase 1, 5 existing runs) | CNN(12->64->128->128) -> LSTM(128) -> mean over time -> Dropout 0.2 -> Linear(128,5) |
| **M1** Early-fusion SE | s_l = RMS of lead l over time; a = SE(s); x' = a * x (lead-wise); then the M0 network on x'. One weight per lead, shared by all classes. |
| **M2** Separate leads (control, no gate) | r from the lead encoder; z = (1/12) sum_l r_l; logits = Linear(32,5)(Dropout 0.2(z)). |
| **M3** Channel-specific SE | As M2, with s_l = mean over the 32 features of r_l; a = SE(s); z = (1/12) sum_l a_l r_l; same head as M2. One weight per lead, shared by all five classes. |
| **M4** Multi-head SE | As M3, but five independent gates a^c = SE_c(s) and five heads: z^c = (1/12) sum_l a^c_l r_l; logit_c = Linear(32,1)(Dropout 0.2(z^c)). |

Design logic: M3 - M2 isolates the gate; M4 - M3 isolates class-specific gates and heads (same head capacity per class); M2 - M0 isolates separating the leads; M1 - M0 isolates the early gate on the same backbone (M1 keeps no temporal attention, unlike the original manuscript, where Early-Fusion SE also kept the attention layer).

**Choices made by the analyst and not derived from the data (stated limitations).** Squeeze by mean over features (M3, M4) and by RMS over time (M1); width D = 32 and encoder widths 16/32/32; bottleneck 3; mean pooling over leads. Parameter counts are not matched; they are reported. These settings are fixed a priori and were not tuned; a different setting could change the numbers.

## 3. Runs
4 new models x seeds {11,22,33,44,55} = 20 runs, same folds (1-8 train, 9 validation, 10 test), same hyper-parameters as Phase 1. M0 uses the five existing `cnn_lstm` runs and is not retrained. Fold 10 is evaluated once per run.

## 4. Outcome group A: classification performance (`aggregate_2b.py`)
Primary estimand: SINGLE (mean over seeds of the single-model test macro-AUROC difference); secondary: ENSEMBLE. Patient-level paired bootstrap, 5000 resamples, same redraw/abort rule as Phase 1.
**Five pre-specified comparisons** (Bonferroni family = 5, level 99%): M1-M0, M2-M0, M3-M2, M4-M3, M4-M2. Per-class AUROC/AUPRC, F1, MCC, Brier and parameter counts are descriptive; no per-class tests. Primary decisions use the Bonferroni interval; unadjusted intervals are reported and labelled exploratory.

## 5. Outcome group B: explanation quality (`lead_perturb.py` then `analyze_native.py`; M1, M3, M4 only, M2 has no gate)
Built-in weight of lead l and class c = mean gate value over the test records. Primary endpoints:
- **B1 Stability.** Mean pairwise Spearman of the class-averaged 12-lead weight vector across the five seeds (10 pairs; the range of the pairs is also reported). Reference: 95th percentile of the same statistic for random rankings (about 0.17). There is no patient-level interval for this statistic, because per-record weights are not stored; only the spread of the pairs is shown.
- **B2 Agreement.** Spearman between built-in weights and (i) zero-out importance, (ii) Integrated Gradients, per class, averaged over classes, mean +- SD over seeds, and the number of seeds with positive correlation.
- **B3 Faithfulness.** Test macro-AUROC after zeroing the k = 3 and k = 6 leads with the largest built-in weight (ranking from the **validation** fold), against the mean of 10 random orders; reported as the number of seeds (of 5) below random.
Descriptive: weight saturation (min, max, range across leads), class specificity (SD across classes divided by SD across seeds, same lead), per-class tables. No comparisons between models are tested; they are described.

## 6. Reporting rules
- The built-in weights describe where the model's gates put weight, not where the heart disease is. Agreement with textbook lead-disease associations is a plausibility note only.
- Zeroing a lead gives out-of-distribution input; permutation importance (computed in Phase 2A for M0 and, optionally, here) is the sensitivity analysis.
- Gate weights near 1 for all leads (saturation) are reported as such and not interpreted as importance differences.
- If M3 or M4 do not exceed M2, or if the weights are unstable or unfaithful, the result is reported as such.
- PTB-XL labels are annotations; one dataset only; 5 seeds; Phase-1 results (Phase 1 aggregate) are not affected by this phase.

## 7. Integrity checks
Same data guards as Phase 1 (`dataset_report.json`, test labels and patient ids must be identical); `aggregate_2b.py` stops if any of the 25 runs is missing; `analyze_native.py` stops if a perturbation file is missing or its checkpoint does not reproduce the saved predictions (gap > 5e-3).

## 8. Order of execution
1. `python test_2b_pipeline.py` (torch-free) and `python test_models_2b.py` (Colab).
2. Smoke: `python smoke_test.py`, then `python run_phase2b.py --synthetic --out ./results_smoke --epochs 2 --seeds 11 --models m4_mh_se`, then `python lead_perturb.py --synthetic --results ./results_smoke --models m4_mh_se --seeds 11 --ig-n 20 --ig-steps 4`.
3. Real: `run_phase2b.py`, `aggregate_2b.py --boot 5000`, `lead_perturb.py --models m1_early_se m2_separate m3_cs_se m4_mh_se`, `analyze_native.py`.

## 9. Deviations log
- `train.build` was extended to build the Phase-2B models; Phase-1 models and checkpoints are unaffected.
