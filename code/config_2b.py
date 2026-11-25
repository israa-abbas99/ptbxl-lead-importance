"""Frozen configuration for Phase 2B (new lead-aware models). Do not edit after the first real 2B run.
All training settings are those of Phase 1 (config.CFG). Only the architectures below are new."""
from config import SEEDS  # noqa: F401  (same five seeds as Phase 1)

BASELINE = "cnn_lstm"        # M0: the Phase-1 model; its 5 existing runs are re-used, not re-trained

# per-lead encoder widths and gate bottleneck, fixed a priori (not tuned)
ARCH = dict(c1=16, c2=32, c3=32, k=(7, 5, 5), gate_bottleneck=3, dropout=0.2)

MODELS_2B = {
    "m1_early_se":  dict(kind="m1"),   # SE gate on the raw 12 leads -> M0 backbone
    "m2_separate":  dict(kind="m2"),   # 12 independent lead encoders, uniform pooling, no gate
    "m3_cs_se":     dict(kind="m3"),   # M2 + one shared SE gate (same lead weights for all classes)
    "m4_mh_se":     dict(kind="m4"),   # M2 + five class-specific SE gates and heads
}
LABEL_2B = {"cnn_lstm": "M0 CNN+LSTM", "m1_early_se": "M1 Early-fusion SE", "m2_separate": "M2 Separate leads",
            "m3_cs_se": "M3 Channel-specific SE", "m4_mh_se": "M4 Multi-head SE"}

# Pre-specified family of 5 comparisons (effect = a - b); Bonferroni level 1 - 0.05/5 = 99%
COMPARISONS_2B = [
    ("M1 vs M0: early SE gate",                         "m1_early_se", "cnn_lstm"),
    ("M2 vs M0: separating the leads",                  "m2_separate", "cnn_lstm"),
    ("M3 vs M2: shared SE gate",                        "m3_cs_se",    "m2_separate"),
    ("M4 vs M3: class-specific gates and heads",        "m4_mh_se",    "m3_cs_se"),
    ("M4 vs M2: total effect of class-specific gating", "m4_mh_se",    "m2_separate"),
]
