"""Tests for the Phase-2B models (needs torch; run on Colab):  python test_models_2b.py"""
import numpy as np
import torch

from config import CFG, CLASSES
from config_2b import MODELS_2B
from models_lead import LeadEncoder, LeadNet
from train import build

torch.manual_seed(0)
B, T = 6, 1000
x = torch.randn(B, 12, T)

# 1) shapes, finite outputs, parameter counts
print("parameters:", {n: sum(p.numel() for p in build(n).parameters()) for n in list(MODELS_2B) + ["cnn_lstm"]})
for name in MODELS_2B:
    m = build(name).eval()
    y = m(x)
    assert y.shape == (B, len(CLASSES)) and torch.isfinite(y).all(), name
    w = m.lead_weights(x)
    assert w.shape == (B, len(CLASSES), 12), (name, w.shape)
    assert (w > 0).all() and (w <= 1).all(), name
print("shapes / gate range OK")

# 2) the 12 lead encoders are independent: changing lead 3 changes only r_3
enc = LeadEncoder().eval()
x2 = x.clone(); x2[:, 3] = torch.randn(B, T)
r1, r2 = enc(x), enc(x2)
others = [l for l in range(12) if l != 3]
assert torch.allclose(r1[:, others], r2[:, others], atol=1e-6), "leads are not independent"
assert not torch.allclose(r1[:, 3], r2[:, 3]), "lead 3 did not change"
print("lead independence OK")

# 3) gate structure: M2 weights are all ones; M3 weights identical over classes; M4 differ over classes
w2 = build("m2_separate").eval().lead_weights(x); assert torch.all(w2 == 1)
w3 = build("m3_cs_se").eval().lead_weights(x); assert torch.allclose(w3, w3[:, :1].expand_as(w3))
w4 = build("m4_mh_se").eval().lead_weights(x); assert (w4[:, 0] - w4[:, 1]).abs().max() > 1e-4
print("gate structure OK")

# 4) M2 equals M3 with the gate forced to 1 (so M3 - M2 isolates the gate)
m2, m3 = build("m2_separate").eval(), build("m3_cs_se").eval()
m3.load_state_dict({k: v for k, v in m2.state_dict().items()}, strict=False)
with torch.no_grad():
    m3.gate.fc2.weight.zero_(); m3.gate.fc2.bias.fill_(30.0)          # sigmoid(30) = 1 within 1e-12
assert torch.allclose(m2(x), m3(x), atol=1e-5), "M3 with unit gates should equal M2"
print("M2 == M3(gate=1) OK")

# 5) gradients reach the gate and every lead encoder; zeroing a lead changes the output
for name in MODELS_2B:
    m = build(name).train()
    m(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad), name
xz = x.clone(); xz[:, 5] = 0
m = build("m4_mh_se").eval()
assert not torch.allclose(m(x), m(xz))
print("gradients OK")

# 6) eval mode is deterministic and lead_weights restores the training flag
m = build("m3_cs_se").train(); m.lead_weights(x); assert m.training
m.eval(); assert torch.equal(m(x), m(x))
print("ALL PHASE-2B MODEL TESTS PASSED")
