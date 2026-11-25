"""Quick check (about a minute). Run BEFORE the real training:  python smoke_test.py
1) every model builds and returns (B, 5) logits; prints parameter counts
2) a 2-epoch synthetic run of all six models finishes without errors
"""
import shutil
import subprocess
import sys

import torch

from config import MODELS
from train import build

x = torch.randn(4, 12, 1000)
print(f"{'model':18s} {'params':>10s}  out-shape")
for name in MODELS:
    m = build(name).eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 5), (name, y.shape)
    print(f"{name:18s} {sum(p.numel() for p in m.parameters()):>10,d}  {tuple(y.shape)}")

print("\nSynthetic end-to-end run (2 epochs, 1 seed)...")
shutil.rmtree("./results_smoke", ignore_errors=True)
subprocess.check_call([sys.executable, "run_phase1.py", "--synthetic",
                       "--out", "./results_smoke", "--epochs", "2", "--seeds", "11"])
print("\nSMOKE TEST PASSED")
