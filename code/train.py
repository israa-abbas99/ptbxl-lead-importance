"""Train/evaluate one (model, seed) pair. Test fold is evaluated once, at the
best-validation checkpoint; thresholds and model selection use fold 9 only."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config import CFG, CLASSES, MODELS
from metrics import choose_thresholds, evaluate, macro_auroc
from models import ECGNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def build(name):
    from config_2b import MODELS_2B                       # Phase-2B models (additive; Phase-1 behaviour unchanged)
    if name in MODELS_2B:
        from models_lead import LeadNet
        return LeadNet(MODELS_2B[name]["kind"], n_classes=len(CLASSES), n_leads=CFG["n_leads"])
    return ECGNet(**MODELS[name], n_classes=len(CLASSES), n_leads=CFG["n_leads"],
                  d=CFG["d_model"], heads=CFG["heads"],
                  dropout=CFG["dropout"], attn_dropout=CFG["attn_dropout"])


def train_one(name, seed, data, outdir, device, max_epochs=None):
    os.makedirs(outdir, exist_ok=True)
    max_epochs = max_epochs or CFG["max_epochs"]
    set_seed(seed)
    model = build(name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=CFG["sched_factor"], patience=CFG["sched_patience"])
    crit = nn.BCEWithLogitsLoss()

    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    Xte, Yte = data["test"]
    g = torch.Generator().manual_seed(seed)
    bs = CFG["batch_size"]

    best, best_state, best_ep, bad, log = -1.0, None, -1, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).numpy()
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            tot += loss.item() * len(idx)
        pva = predict(model, Xva, device)
        auc = macro_auroc(Yva, pva)
        sched.step(auc)
        log.append(dict(epoch=ep, train_loss=tot / len(Xtr), val_macro_auroc=auc,
                        lr=opt.param_groups[0]["lr"]))
        print(f"[{name} s{seed}] ep{ep:02d} loss {tot/len(Xtr):.4f} val-AUROC {auc:.4f}", flush=True)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break

    model.load_state_dict(best_state)
    predict(model, Xte[:64], device)                      # warm-up
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()
    pte = predict(model, Xte, device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    infer_ms = (time.time() - t1) / len(Xte) * 1000
    pva = predict(model, Xva, device)
    thr = choose_thresholds(Yva, pva, CFG["thr_grid"])
    res = dict(model=name, seed=seed, n_params=n_params, best_epoch=best_ep,
               epochs_run=len(log), train_seconds=time.time() - t0,
               infer_ms_per_record=infer_ms,
               thresholds=thr.tolist(), val=evaluate(Yva, pva, thr),
               test=evaluate(Yte, pte, thr), log=log)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.save(os.path.join(outdir, "preds_val.npy"), pva)
    np.save(os.path.join(outdir, "preds_test.npy"), pte)
    torch.save(best_state, os.path.join(outdir, "model.pt"))
    return res
