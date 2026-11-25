# Phase 2A Protocol: lead-level importance of the Phase-1 models (v1, fixed before any Phase-2A result is seen)

Status: **analysis only, no training.** It uses the 30 saved Phase-1 checkpoints (6 models x 5 seeds, `model.pt`) and the unchanged Phase-1 data pipeline. Phase 2B (new models with learned lead weights, M1-M4) is reserved and is not part of this protocol.

## 1. Question
Do the leads a trained multi-label ECG classifier depends on form a stable and credible pattern? Specifically:
1. **Stability:** is the ranking of the 12 leads reproduced across training seeds?
2. **Agreement:** do independent importance measures (lead removal vs. Integrated Gradients) rank the leads alike?
3. **Faithfulness:** does removing the leads that the ranking calls important hurt performance more than removing random leads?
4. **Class dependence:** do NORM, MI, STTC, CD and HYP rely on different leads?

## 2. What is measured
All numbers come from the Phase-1 test fold (Fold 10). No model is selected, tuned or re-trained using these results.

| Quantity | Definition |
|---|---|
| Lead importance (primary) | drop in per-class AUROC when one lead is set to 0 in every test record (0 is the train-set mean of the lead after z-scoring, i.e. a missing lead). "macro" = mean over the five classes. |
| Permutation importance (sensitivity) | the same drop when the lead is replaced by the same lead of randomly chosen other records (3 repeats, fixed seeds); keeps the lead's marginal distribution. |
| Integrated Gradients (independent method) | on logits, baseline 0, 32 midpoint steps, mean over a fixed random subset of 500 test records of the sum over time of the absolute attribution, per lead and class. Completeness error is reported. |
| Deletion curves | leads are zeroed cumulatively (k = 0..11) in four kinds of order: most important first and least important first, both ranked by **validation-fold** zero-out importance; most important first by Integrated Gradients; and 10 fixed random orders (reference). Test macro-AUROC is recorded at each k. Rankings never use test labels. |

## 3. Primary endpoints (for the Phase-1 winner, `cnn_lstm`; all six models are reported)
- **P1 Stability.** Mean pairwise Spearman correlation of the 12-lead importance vector (macro, zero-out) over the 5 seeds, with a 95% patient-level bootstrap CI (1000 resamples; resamples lacking positives or negatives are discarded and redrawn; abort if more than 1% are discarded). Reference: the 95th percentile of the same statistic for random rankings (simulated).
- **P2 Agreement.** Spearman correlation between zero-out importance and Integrated Gradients importance per class, averaged over the five classes and then over seeds (mean +- SD over seeds).
- **P3 Faithfulness.** At k = 3 and k = 6 removed leads: test macro-AUROC after removing the validation-ranked most important leads, compared with the mean of the 10 random orders; reported as the number of seeds (of 5) in which the ranked curve is lower.

**Scope of inference.** The primary analysis (P1-P3) concerns the five `cnn_lstm` runs, which differ only in the random seed. Seed stability is always computed within one architecture across its own five seeds, so architecture and seed are never confounded. Comparisons across the six architectures are exploratory and descriptive: a difference in importance between two architectures may reflect the architecture itself and is not evidence of (in)stability. The Phase-1 models contain no SE weights, so only external measures (zero-out, permutation, Integrated Gradients) can be applied to them; comparison with built-in SE weights belongs to Phase 2B. Rankings, not raw scores, are compared across measures, so the measures need not share a scale.

Everything else (permutation importance, class similarity, per-class tables, comparisons between the six models, Integrated Gradients deletion order) is **descriptive**. No per-class or between-model hypothesis tests are performed.

## 4. How results will be reported
- No claim of "most important lead clinically". Removal measures what the model **depends on**, not clinical relevance.
- Agreement with textbook lead-disease associations (for example inferior leads for MI) may be shown as a plausibility check only.
- Perturbed inputs are out of distribution for the network (a flat lead); this is a limitation, and the permutation analysis is the sensitivity check for it.
- Integrated Gradients uses a subset of test records; this is stated with its size.
- A null or negative result (low stability or low agreement) is reported as such.
- The labels of PTB-XL are annotations, not clinical ground truth.

## 5. Integrity checks built into the code
- Data and cache must reproduce the Phase-1 test labels and patient ids exactly, otherwise the run stops.
- Each checkpoint must reproduce the saved test predictions (max gap 5e-3), otherwise the run stops.
- The analysis stops if any (model, seed) file is missing.
- Fixed seeds for subsets, random orders and permutations.

## 6. Deviations log
(empty)
