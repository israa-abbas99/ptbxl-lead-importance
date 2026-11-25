"""Frozen configuration for Phase 1 (PTB-XL, 6-model 2x3 design).

Do NOT edit after the first real run. Any change must be logged in the
deviations section of PROTOCOL_PHASE1.md and all affected runs repeated.
"""

SEEDS = [11, 22, 33, 44, 55]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# temporal module {none, LSTM, BiLSTM} x attention {none, MHSA}
MODELS = {
    "cnn":             dict(temporal="none",   mhsa=False),
    "cnn_mhsa":        dict(temporal="none",   mhsa=True),
    "cnn_lstm":        dict(temporal="lstm",   mhsa=False),
    "cnn_lstm_mhsa":   dict(temporal="lstm",   mhsa=True),
    "cnn_bilstm":      dict(temporal="bilstm", mhsa=False),
    "cnn_bilstm_mhsa": dict(temporal="bilstm", mhsa=True),
}

# Pre-specified comparisons: (label, model_a, model_b, kind); effect = a - b
COMPARISONS = [
    ("MHSA effect on CNN",          "cnn_mhsa",        "cnn",         "ablation"),
    ("MHSA effect on CNN+LSTM",     "cnn_lstm_mhsa",   "cnn_lstm",    "ablation"),
    ("MHSA effect on CNN+BiLSTM",   "cnn_bilstm_mhsa", "cnn_bilstm",  "ablation"),
    ("LSTM effect (no MHSA)",       "cnn_lstm",        "cnn",         "ablation"),
    ("LSTM effect (with MHSA)",     "cnn_lstm_mhsa",   "cnn_mhsa",    "ablation"),
    ("BiLSTM vs LSTM (no MHSA)",    "cnn_bilstm",      "cnn_lstm",    "controlled comparison"),
    ("BiLSTM vs LSTM (with MHSA)",  "cnn_bilstm_mhsa", "cnn_lstm_mhsa", "controlled comparison"),
]

# Interaction contrasts (difference of MHSA effects between temporal back-ends)
INTERACTIONS = [
    ("MHSA effect: CNN+LSTM minus CNN",
     {"cnn_lstm_mhsa": 1, "cnn_lstm": -1, "cnn_mhsa": -1, "cnn": 1}),
    ("MHSA effect: CNN+BiLSTM minus CNN",
     {"cnn_bilstm_mhsa": 1, "cnn_bilstm": -1, "cnn_mhsa": -1, "cnn": 1}),
    ("MHSA effect: CNN+BiLSTM minus CNN+LSTM",
     {"cnn_bilstm_mhsa": 1, "cnn_bilstm": -1, "cnn_lstm_mhsa": -1, "cnn_lstm": 1}),
]

CFG = dict(
    sampling_rate=100,
    n_leads=12,
    n_samples=1000,
    # Fixed by convention before any training (not tuned on validation or test data).
    # Zero-phase Butterworth band-pass, applied per record, then per-lead z-score (train folds only).
    bandpass=dict(low_hz=0.5, high_hz=40.0, order=4),
    train_folds=list(range(1, 9)),
    val_fold=9,
    test_fold=10,
    d_model=128,
    heads=4,
    dropout=0.2,
    attn_dropout=0.1,
    batch_size=128,
    max_epochs=100,                # early stopping (patience 8) ends runs sooner; pilot on val only
    patience=8,
    lr=1e-3,
    weight_decay=1e-4,
    sched_factor=0.5,
    sched_patience=3,
    grad_clip=1.0,
    thr_grid=(0.05, 0.95, 0.01),   # per-class F1-optimal threshold, tuned on fold 9 only
    selection_metric="val_macro_auroc",
)
