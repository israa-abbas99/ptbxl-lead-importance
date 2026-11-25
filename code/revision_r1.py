"""Analyses requested in the first IJIES review (2026-11). None of them was pre-specified; all are reported as such.
Uses the saved checkpoints; only task `tied` trains new networks. Run from the folder with the other .py files.

  python revision_r1.py constants --root R --cache C --results RES   # normalisation constants + software versions
  python revision_r1.py subsets   --root R --cache C --results RES   # R2-2: exhaustive k=3 / k=6 deletion reference
  python revision_r1.py groups    --root R --cache C --results RES   # R2-1: lead-group perturbations
  python revision_r1.py tied      --root R --cache C --results RES   # R2-4: M4 with one gate tied across classes
  python revision_r1.py baseline  --root R --cache C --results RES   # R2-3: benchmark ResNet (resnet1d_wang), same
                                                                     #   protocol, then the Phase-2A lead analysis on it
  python revision_r1.py mi        --root R --cache C --results RES   # R2-5: MI split by infarct location (AMI / IMI)
  python revision_r1.py summary   --results RES                      # tables from the three tasks above (CPU)
Every task is resumable (finished runs are skipped). Outputs: RES/revision/...
"""
import argparse
import itertools
import json
import os
import platform
import sys
import time

import numpy as np

SEEDS = [11, 22, 33, 44, 55]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GATED = ["m1_early_se", "m3_cs_se", "m4_mh_se"]
SUBSET_MODELS = ["cnn_lstm", "m2_separate"] + GATED      # M2: gate-free control of the lead-separate design
GROUPS = {"limb (I, II, III, aVR, aVL, aVF)": [0, 1, 2, 3, 4, 5], "V1-V2": [6, 7], "V3-V4": [8, 9], "V5-V6": [10, 11]}
N_REF = 100            # random subsets used as the bootstrap reference (the point estimate uses all subsets)
N_BOOT = 1000
BOOT_SEED = 2026
TIED = "m4t_tied"
BASE = "resnet1d_wang"


# ---------------------------------------------------------------- benchmark ResNet (Strodthoff et al. 2021)
def build_resnet_wang(n_leads=12, n_classes=5):
    """resnet1d_wang as defined in github.com/helme/ecg_ptbxl_benchmarking (code/models/resnet1d.py):
    stem Conv1d(k7, stride 1, 128) + BN + ReLU, no pooling; three basic residual blocks with 128 channels,
    kernel sizes (5, 3), strides 1, 2, 2 (1x1 projection when the stride is 2); head = concat(max, avg) pooling,
    BatchNorm, Dropout 0.5, Linear(256, 5)."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c, stride):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, 5, stride=stride, padding=2, bias=False); self.b1 = nn.BatchNorm1d(c)
            self.c2 = nn.Conv1d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm1d(c)
            self.down = None if stride == 1 else nn.Sequential(nn.Conv1d(c, c, 1, stride=stride, bias=False),
                                                                 nn.BatchNorm1d(c))

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            h = torch.relu(self.b1(self.c1(x)))
            return torch.relu(self.b2(self.c2(h)) + r)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(n_leads, 128, 7, padding=3, bias=False), nn.BatchNorm1d(128), nn.ReLU())
            self.blocks = nn.Sequential(Block(128, 1), Block(128, 2), Block(128, 2))
            self.head = nn.Sequential(nn.BatchNorm1d(256), nn.Dropout(0.5), nn.Linear(256, n_classes))

        def forward(self, x):
            h = self.blocks(self.stem(x))
            return self.head(torch.cat([h.amax(-1), h.mean(-1)], 1))
    return Net()


def patch_builds():
    """Make train.build (and the modules that imported it) know the two models added in revision."""
    import train as T
    orig = getattr(T, "_orig_build", T.build)
    T._orig_build = orig
    T.build = lambda name: build_tied() if name == TIED else build_resnet_wang() if name == BASE else orig(name)
    import lead_perturb
    lead_perturb.build = T.build
    return T


# ---------------------------------------------------------------- shared helpers
def load_data(a):
    from data import load_ptbxl, split_and_standardise
    data = split_and_standardise(load_ptbxl(a.root, a.cache))
    y_saved = np.load(os.path.join(a.results, "y_test.npy"))
    p_saved = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    if not (np.array_equal(y_saved, data["test"][1]) and np.array_equal(p_saved, data["test_patient_id"])):
        raise SystemExit("STOP: test labels / patient ids differ from the results folder.")
    return data


def strict_fp32():
    """Ampere-class GPUs (L4, A100) use TF32 for convolutions and matrix products by default, which moves
    sigmoid outputs by up to ~1e-2 relative to the original runs. All revision inference is done in full FP32."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(name, seed, a, dev):
    import torch
    strict_fp32()
    T = patch_builds()
    m = T.build(name).to(dev)
    m.load_state_dict(torch.load(os.path.join(a.results, name, f"seed{seed}", "model.pt"), map_location=dev))
    m.eval()
    return m


def predict_masked(model, Xg, mask, dev, bs=512):
    """Xg: test signals already on the device; mask (12,) of 0/1. Returns sigmoid outputs (N, 5)."""
    import torch
    out = []
    m = torch.as_tensor(mask, dtype=Xg.dtype, device=dev).view(1, -1, 1)
    with torch.no_grad():
        for i in range(0, len(Xg), bs):
            out.append(torch.sigmoid(model(Xg[i:i + bs] * m)).float().cpu().numpy())
    return np.concatenate(out)


def macro_auc(Y, P):
    from leadlib import class_aurocs
    return float(class_aurocs(Y, P).mean())


def patient_weights(pid, Y, B, seed):
    """Patient-level bootstrap as record weights (B, N); resamples with a class lacking positives or negatives
    are redrawn, as in the main analysis."""
    uniq, inv = np.unique(pid, return_inverse=True)
    rng = np.random.default_rng(seed)
    W, red = [], 0
    while len(W) < B:
        c = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(np.float64)
        pos = (c[:, None] * Y).sum(0)
        if (pos == 0).any() or (pos == c.sum()).any():
            red += 1
            continue
        W.append(c)
    return np.array(W), red


def weighted_macro_auc(Y, P, W):
    """Macro AUROC for every row of record weights W (B, N); ties between scores counted as one half."""
    out = np.zeros(len(W))
    for c in range(Y.shape[1]):
        o = np.argsort(P[:, c], kind="mergesort")
        s, y = P[o, c], Y[o, c] > 0.5
        Ws = W[:, o]
        # group equal scores
        starts = np.r_[0, np.where(np.diff(s) != 0)[0] + 1]
        neg = Ws * (~y)
        pos = Ws * y
        negg = np.add.reduceat(neg, starts, axis=1)
        posg = np.add.reduceat(pos, starts, axis=1)
        below = np.cumsum(negg, 1) - negg
        num = (posg * (below + 0.5 * negg)).sum(1)
        out += num / (pos.sum(1) * neg.sum(1))
    return out / Y.shape[1]


# ---------------------------------------------------------------- constants (R1-4)
def task_constants(a):
    from data import bandpass_filter, load_ptbxl
    from config import CFG
    import torch, scipy, pandas
    raw = load_ptbxl(a.root, a.cache)
    X = bandpass_filter(raw["X"])
    tr = np.isin(raw["fold"], CFG["train_folds"])
    mean = X[tr].mean(axis=(0, 2)); std = X[tr].std(axis=(0, 2)) + 1e-8
    d = os.path.join(a.results, "revision"); os.makedirs(d, exist_ok=True)
    json.dump({"leads": LEADS, "train_mean_after_filter": mean.tolist(), "train_std_after_filter": std.tolist(),
               "filter": CFG["bandpass"], "folds": dict(train=CFG["train_folds"], val=CFG["val_fold"], test=CFG["test_fold"]),
               "software": dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                scipy=scipy.__version__, pandas=pandas.__version__,
                                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
               "note": "Software versions are those of the session that ran this script; see the manuscript for the runs."},
              open(os.path.join(d, "normalisation_and_versions.json"), "w"), indent=1)
    print("saved", os.path.join(d, "normalisation_and_versions.json"))


# ---------------------------------------------------------------- R2-2 exhaustive deletion reference
def ranked_orders(a, name, seed):
    z = np.load(os.path.join(a.results, "phase2a", name, f"seed{seed}", "lead_perturb.npz"))
    names = list(z["del_names"])
    o = {"removal (validation)": z["del_orders"][names.index("occlusion_val_most_first")]}
    ig = os.path.join(a.results, "ig_sens", name, f"seed{seed}.npz")
    o["IG (validation, 128 steps)"] = np.load(ig)["order_val_ig"]
    if name in GATED and "native_val_most_first" in names:      # M2's "weights" are all 1: no meaningful order
        o["built-in weights (validation)"] = z["del_orders"][names.index("native_val_most_first")]
    return o


def subset_models(a):
    return SUBSET_MODELS + ([BASE] if os.path.exists(os.path.join(a.results, "ig_sens", BASE)) else [])


def task_subsets(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(a)
    Xte, Yte = data["test"]; pid = data["test_patient_id"]
    Xg = torch.from_numpy(Xte).to(dev)
    out = os.path.join(a.results, "revision", "subsets"); os.makedirs(out, exist_ok=True)
    W, red = patient_weights(pid, Yte, N_BOOT, BOOT_SEED)
    print(f"bootstrap weights: {len(W)} resamples, {red} redrawn", flush=True)
    for name in subset_models(a):
        for seed in SEEDS:
            f = os.path.join(out, f"{name}_seed{seed}.npz")
            if os.path.exists(f):
                print("skip", name, seed); continue
            t0 = time.time()
            model = load_model(name, seed, a, dev)
            full = predict_masked(model, Xg, np.ones(12), dev)
            gap = float(np.abs(full - np.load(os.path.join(a.results, name, f"seed{seed}", "preds_test.npy"))).max())
            if gap > 5e-3:
                raise SystemExit(f"STOP: {name} {seed}: checkpoint does not reproduce saved predictions ({gap:.1e})")
            res = {"ckpt_gap": gap}
            orders = ranked_orders(a, name, seed)
            for k in (3, 6):
                subs = list(itertools.combinations(range(12), k))
                auc = np.zeros(len(subs))
                P = {}
                rng = np.random.default_rng([BOOT_SEED, k])
                ref_idx = np.sort(rng.choice(len(subs), N_REF, replace=False))
                keep = set(ref_idx.tolist())
                for i, s in enumerate(subs):
                    mask = np.ones(12); mask[list(s)] = 0
                    p = predict_masked(model, Xg, mask, dev)
                    auc[i] = macro_auc(Yte, p)
                    if i in keep:
                        P[i] = p
                ref_boot = np.mean([weighted_macro_auc(Yte, P[i], W) for i in ref_idx], 0)        # (B,)
                res[f"k{k}_subsets"] = np.array(subs)
                res[f"k{k}_auc"] = auc
                res[f"k{k}_ref_idx"] = ref_idx
                res[f"k{k}_ref_boot"] = ref_boot
                for oname, o in orders.items():
                    key = oname.split(" ")[0]
                    mask = np.ones(12); mask[list(o[:k])] = 0
                    p = predict_masked(model, Xg, mask, dev)
                    res[f"k{k}_{key}_auc"] = macro_auc(Yte, p)
                    res[f"k{k}_{key}_boot"] = weighted_macro_auc(Yte, p, W)
                    res[f"k{k}_{key}_set"] = np.sort(o[:k])
            np.savez_compressed(f, **res, order_names=np.array(list(orders)))
            print(f"DONE {name} seed {seed} (gap {gap:.1e}, {(time.time() - t0) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- R2-1 lead-group perturbations
def task_groups(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(a)
    Xte, Yte = data["test"]
    out = os.path.join(a.results, "revision", "groups"); os.makedirs(out, exist_ok=True)
    from leadlib import class_aurocs
    from train import predict
    for name in subset_models(a):
        for seed in SEEDS:
            f = os.path.join(out, f"{name}_seed{seed}.npz")
            if os.path.exists(f):
                print("skip", name, seed); continue
            model = load_model(name, seed, a, dev)
            P0 = predict(model, Xte, dev)
            gap = float(np.abs(P0 - np.load(os.path.join(a.results, name, f"seed{seed}", "preds_test.npy"))).max())
            if gap > 5e-3:
                raise SystemExit(f"STOP: {name} {seed}: checkpoint does not reproduce saved predictions ({gap:.1e})")
            base = class_aurocs(Yte, P0)
            zero = np.zeros((len(GROUPS), 5)); perm = np.zeros((3, len(GROUPS), 5))
            for g, leads in enumerate(GROUPS.values()):
                Xp = Xte.copy(); Xp[:, leads, :] = 0.0
                zero[g] = base - class_aurocs(Yte, predict(model, Xp, dev))
                for r in range(3):
                    donor = np.random.default_rng(BOOT_SEED + 100 * r).permutation(len(Xte))
                    Xp = Xte.copy(); Xp[:, leads, :] = Xte[donor][:, leads, :]     # whole group from ONE donor record
                    perm[r, g] = base - class_aurocs(Yte, predict(model, Xp, dev))
            np.savez_compressed(f, zero=zero, perm=perm, groups=np.array(list(GROUPS)), ckpt_gap=gap)
            print(f"DONE {name} {seed} (gap {gap:.1e})", flush=True)


# ---------------------------------------------------------------- R2-4 tied-gate control
def build_tied():
    import torch.nn as nn
    from models_lead import LeadNet, SEGate
    net = LeadNet("m4", n_classes=5, n_leads=12)
    g = SEGate(12)
    net.gates = nn.ModuleList([g] * 5)          # one gate, shared by the five class-specific readouts
    return net


def task_tied(a):
    import torch
    T = patch_builds()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(a)
    for seed in SEEDS:
        d = os.path.join(a.results, TIED, f"seed{seed}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            print("skip", seed); continue
        r = T.train_one(TIED, seed, data, d, dev)
        print(f"DONE {TIED} seed {seed}: val {r['val']['macro']['auroc']:.4f}, test {r['test']['macro']['auroc']:.4f}, "
              f"params {r['n_params']:,}, {r['train_seconds'] / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- R2-3 benchmark baseline
def task_baseline(a):
    import torch
    T = patch_builds()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(a)
    for seed in SEEDS:
        d = os.path.join(a.results, BASE, f"seed{seed}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            print("skip training", seed); continue
        r = T.train_one(BASE, seed, data, d, dev)
        print(f"DONE {BASE} seed {seed}: val {r['val']['macro']['auroc']:.4f}, test {r['test']['macro']['auroc']:.4f}, "
              f"params {r['n_params']:,}, {r['train_seconds'] / 60:.1f} min", flush=True)
    # Phase-2A analysis (zero-out, permutation, IG, deletion curves), exactly as for the Phase-1 networks
    import lead_perturb
    sys.argv = ["lead_perturb.py", "--root", a.root, "--cache", a.cache, "--results", a.results, "--models", BASE]
    lead_perturb.main()
    # validation-derived IG (128 steps) and its deletion curve, as in the IG sensitivity analysis
    import ig_sensitivity
    ns = argparse.Namespace(root=a.root, cache=a.cache, results=a.results, steps=128, ig_n=500, device=dev)
    ig_sensitivity.run_gpu(ns, [BASE])


# ---------------------------------------------------------------- R2-5 MI by location
def task_mi(a):
    """Lead removal importance for MI computed separately for anterior (AMI) and inferior (IMI) infarcts,
    each against all MI-negative test records, from the saved single-lead zero-out predictions."""
    import ast
    import pandas as pd
    from data import load_ptbxl
    from config import CFG
    from leadlib import fast_auroc, mean_pairwise_spearman
    raw = load_ptbxl(a.root, a.cache)
    te = raw["fold"] == CFG["test_fold"]
    ids = raw["ecg_id"][te]
    Y = np.load(os.path.join(a.results, "y_test.npy"))
    if not np.array_equal(raw["Y"][te], Y):
        raise SystemExit("STOP: test labels differ from the results folder.")
    df = pd.read_csv(os.path.join(a.root, "ptbxl_database.csv"), index_col="ecg_id")
    st = pd.read_csv(os.path.join(a.root, "scp_statements.csv"), index_col=0)
    st = st[st.diagnostic == 1]
    sub = {}
    for e in ids:
        codes = ast.literal_eval(df.loc[e, "scp_codes"])
        sub[e] = {st.loc[c, "diagnostic_subclass"] for c in codes if c in st.index}
    is_ami = np.array(["AMI" in sub[e] for e in ids]); is_imi = np.array(["IMI" in sub[e] for e in ids])
    neg = Y[:, 1] < 0.5
    groups = {"AMI only": is_ami & ~is_imi, "IMI only": is_imi & ~is_ami, "all MI": Y[:, 1] > 0.5}
    rows, out = [], {}
    for name in ["cnn_lstm", "m4_mh_se"] + ([BASE] if os.path.exists(os.path.join(a.results, "phase2a", BASE)) else []):
        imp = {g: [] for g in groups}
        for s in SEEDS:
            z = np.load(os.path.join(a.results, "phase2a", name, f"seed{s}", "lead_perturb.npz"))
            for g, m in groups.items():
                k = m | neg
                yb = Y[k, 1]
                base = fast_auroc(yb, z["P_base_test"][k, 1])
                imp[g].append([base - fast_auroc(yb, z["P_zero_test"][l][k, 1]) for l in range(12)])
        for g in groups:
            V = np.array(imp[g])
            out[(name, g)] = V.mean(0)
            rows.append({"model": name, "subset": g, "n positive": int(groups[g].sum()),
                         "stability over seeds": round(mean_pairwise_spearman(V), 3),
                         **{L: round(float(v), 4) for L, v in zip(LEADS, V.mean(0))}})
    T = pd.DataFrame(rows)
    # gate weights of M4 for the MI class, by location (per-record weights saved in b1_ci)
    gw = []
    for s in SEEDS:
        A = np.load(os.path.join(a.results, "b1_ci", f"m4_mh_se_seed{s}.npz"))["A"][:, 1, :]      # MI gate (N, 12)
        gw.append([A[groups["AMI only"]].mean(0), A[groups["IMI only"]].mean(0)])
    gw = np.array(gw).mean(0)
    G = pd.DataFrame(gw, index=["M4 MI gate, AMI records", "M4 MI gate, IMI records"], columns=LEADS).round(3)
    d = os.path.join(a.results, "revision"); os.makedirs(d, exist_ok=True)
    T.to_csv(os.path.join(d, "table_mi_location.csv"), index=False)
    G.to_csv(os.path.join(d, "table_mi_location_gates.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(T.to_string(index=False)); print(G.to_string())


# ---------------------------------------------------------------- summary (CPU)
def task_summary(a):
    import pandas as pd
    rev = os.path.join(a.results, "revision"); os.makedirs(rev, exist_ok=True)
    Y = np.load(os.path.join(a.results, "y_test.npy")); pid = np.load(os.path.join(a.results, "test_patient_ids.npy"))
    # R2-2
    rows, agg = [], []
    done = lambda sub, n: all(os.path.exists(os.path.join(rev, sub, f"{n}_seed{s}.npz")) for s in SEEDS)
    for name in [m for m in subset_models(a) if done("subsets", m)]:
        Z = [np.load(os.path.join(rev, "subsets", f"{name}_seed{s}.npz")) for s in SEEDS]
        for k in (3, 6):
            for key in [n.split(" ")[0] for n in Z[0]["order_names"]]:
                if f"k{k}_{key}_auc" not in Z[0].files:
                    continue
                d_seed, pct = [], []
                for s, z in zip(SEEDS, Z):
                    allauc = z[f"k{k}_auc"]
                    r = float(z[f"k{k}_{key}_auc"])
                    d_seed.append(r - allauc.mean())
                    pct.append(float((allauc < r).mean() * 100))
                    rows.append({"model": name, "k": k, "order": key, "seed": s, "ranked AUROC": round(r, 4),
                                 "mean of all subsets": round(float(allauc.mean()), 4),
                                 "5th percentile of all subsets": round(float(np.percentile(allauc, 5)), 4),
                                 "difference": round(r - float(allauc.mean()), 4),
                                 "% of subsets less damaging": round(100 - pct[-1], 1)})
                boot = np.mean([z[f"k{k}_{key}_boot"] - z[f"k{k}_ref_boot"] for z in Z], 0)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                agg.append({"model": name, "k": k, "order": key, "n subsets": len(Z[0][f"k{k}_auc"]),
                            "mean difference over seeds": round(float(np.mean(d_seed)), 4),
                            "95% patient-bootstrap CI (100-subset reference)": f"[{lo:.4f}, {hi:.4f}]",
                            "seeds below the subset mean": f"{int(np.sum(np.array(d_seed) < 0))}/5",
                            "mean % of subsets less damaging": round(100 - float(np.mean(pct)), 1)})
    if agg:
        pd.DataFrame(rows).to_csv(os.path.join(rev, "table_deletion_exhaustive_per_seed.csv"), index=False)
        A = pd.DataFrame(agg); A.to_csv(os.path.join(rev, "table_deletion_exhaustive.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(A.to_string(index=False))
    # R2-1
    rows = []
    for name in [m for m in subset_models(a) if done("groups", m)]:
        Z = [np.load(os.path.join(rev, "groups", f"{name}_seed{s}.npz")) for s in SEEDS]
        zero = np.stack([z["zero"] for z in Z]); perm = np.stack([z["perm"].mean(0) for z in Z])
        for g, gname in enumerate(GROUPS):
            r = {"model": name, "group": gname}
            for lab, M in (("zero", zero), ("joint replacement", perm)):
                r[f"{lab}: macro drop"] = f"{M[:, g].mean(1).mean():.4f} ± {M[:, g].mean(1).std(ddof=1):.4f}"
                for c, cn in enumerate(CLASSES):
                    r[f"{lab}: {cn}"] = round(float(M[:, g, c].mean()), 4)
            rows.append(r)
    if rows:
        G = pd.DataFrame(rows); G.to_csv(os.path.join(rev, "table_group_perturbation.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(G.to_string(index=False))
    # R2-4
    if all(os.path.exists(os.path.join(a.results, TIED, f"seed{s}", "preds_test.npy")) for s in SEEDS):
        from aggregate import boot_contrasts, point_contrasts
        ms_ = ["m3_cs_se", TIED, "m4_mh_se"]
        PS = {m: np.stack([np.load(os.path.join(a.results, m, f"seed{s}", "preds_test.npy")) for s in SEEDS]) for m in ms_}
        con = [{"m4_mh_se": 1, TIED: -1}, {TIED: 1, "m3_cs_se": -1}, {"m4_mh_se": 1, "m3_cs_se": -1}]
        lab = ["M4 - M4-tied (class-specific vs shared gate, same readout)",
               "M4-tied - M3 (readout and dropout implementation, same shared gate)", "M4 - M3 (as in Table VIII)"]
        pS, pE = point_contrasts(Y, PS, con)
        DS, DE, sk = boot_contrasts(Y, PS, con, pid, 5000, np.random.default_rng(2026))
        rows = []
        for i, l in enumerate(lab):
            ma, mb = list(con[i])
            auc = lambda m, s: json.load(open(os.path.join(a.results, m, f"seed{s}", "metrics.json")))["test"]["macro"]["auroc"]
            d = np.array([auc(ma, s) - auc(mb, s) for s in SEEDS])
            rows.append({"contrast": l, "single": f"{pS[i]:+.4f}", "seeds positive": f"{int((d > 0).sum())}/5",
                         "95% CI": "[{:+.4f}, {:+.4f}]".format(*np.percentile(DS[:, i], [2.5, 97.5])),
                         "99% CI": "[{:+.4f}, {:+.4f}]".format(*np.percentile(DS[:, i], [0.5, 99.5])),
                         "ensemble": f"{pE[i]:+.4f}",
                         "ensemble 95% CI": "[{:+.4f}, {:+.4f}]".format(*np.percentile(DE[:, i], [2.5, 97.5]))})
        T = pd.DataFrame(rows); T.to_csv(os.path.join(rev, "table_tied_control.csv"), index=False)
        m = [json.load(open(os.path.join(a.results, TIED, f"seed{s}", "metrics.json"))) for s in SEEDS]
        print(T.to_string(index=False))
        print("M4-tied test macro-AUROC: {:.4f} ± {:.4f}; params {:,}; epochs {:.0f}".format(
            np.mean([x["test"]["macro"]["auroc"] for x in m]), np.std([x["test"]["macro"]["auroc"] for x in m], ddof=1),
            m[0]["n_params"], np.mean([x["epochs_run"] for x in m])))
    # R2-3 baseline: P1-P3 for CNN-LSTM and the benchmark ResNet, same endpoints
    if os.path.exists(os.path.join(a.results, "ig_sens", BASE)):
        from leadlib import importance_from_preds, mean_pairwise_spearman, spearman_null, spearman
        thr = float(np.percentile(spearman_null(12, 5, 5000, 0), 95))
        uniq, inv = np.unique(pid, return_inverse=True)
        groups_ = [np.where(inv == k)[0] for k in range(len(uniq))]
        rows = []
        for name in ["cnn_lstm", BASE]:
            Z = [np.load(os.path.join(a.results, "phase2a", name, f"seed{s}", "lead_perturb.npz")) for s in SEEDS]
            G = [np.load(os.path.join(a.results, "ig_sens", name, f"seed{s}.npz")) for s in SEEDS]
            imp = np.stack([importance_from_preds(Y, z["P_base_test"], z["P_zero_test"]) for z in Z])   # (5,12,6)
            p1 = mean_pairwise_spearman(imp[:, :, -1])
            rng = np.random.default_rng(2026); bs = []
            while len(bs) < 1000:
                idx = np.concatenate([groups_[k] for k in rng.integers(0, len(uniq), len(uniq))])
                pos = Y[idx].sum(0)
                if (pos == 0).any() or (pos == len(idx)).any():
                    continue
                bs.append(mean_pairwise_spearman(np.stack([importance_from_preds(Y[idx], z["P_base_test"][idx],
                                                                                  z["P_zero_test"][:, idx])[:, -1] for z in Z])))
            p2 = [np.nanmean([spearman(i[:, c], g["ig_test"][:, c]) for c in range(5)]) for i, g in zip(imp, G)]
            row = {"model": name, "test macro-AUROC": "{:.4f} ± {:.4f}".format(
                       *(lambda v: (np.mean(v), np.std(v, ddof=1)))([json.load(open(os.path.join(a.results, name, f"seed{s}", "metrics.json")))["test"]["macro"]["auroc"] for s in SEEDS])),
                   "P1 stability [95% CI]": "{:.3f} [{:.3f}, {:.3f}]".format(p1, *np.percentile(bs, [2.5, 97.5])),
                   "random 95th percentile": round(thr, 3),
                   "P2 removal vs IG (128 steps)": "{:.3f} ± {:.3f}".format(np.mean(p2), np.std(p2, ddof=1))}
            for k in (3, 6):
                names = [str(n) for n in Z[0]["del_names"]]
                rnd = [i for i, n in enumerate(names) if n.startswith("random_")]
                rv = np.array([z["del_curves"][names.index("occlusion_val_most_first"), k] for z in Z])
                ig = np.array([g["curve_val_ig"][k] for g in G])
                rr = np.array([z["del_curves"][rnd, k].mean() for z in Z])
                row[f"k={k} removal-ranked"] = "{:.4f} ± {:.4f}".format(rv.mean(), rv.std(ddof=1))
                row[f"k={k} val-IG-ranked"] = "{:.4f} ± {:.4f}".format(ig.mean(), ig.std(ddof=1))
                row[f"k={k} random (10)"] = "{:.4f} ± {:.4f}".format(rr.mean(), rr.std(ddof=1))
                row[f"k={k} seeds removal < random"] = f"{int((rv < rr).sum())}/5"
            rows.append(row)
        Bt = pd.DataFrame(rows); Bt.to_csv(os.path.join(rev, "table_baseline_resnet.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(Bt.T.to_string(header=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["constants", "subsets", "groups", "tied", "baseline", "mi", "summary"])
    ap.add_argument("--root"); ap.add_argument("--cache", default="/content/drive/MyDrive/ptbxl_cache")
    ap.add_argument("--results", default="/content/drive/MyDrive/ptbxl_results")
    a = ap.parse_args()
    {"constants": task_constants, "subsets": task_subsets, "groups": task_groups, "tied": task_tied,
     "baseline": task_baseline, "mi": task_mi, "summary": task_summary}[a.task](a)


if __name__ == "__main__":
    main()
