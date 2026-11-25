"""Phase 2B models (PyTorch). Definitions are in PROTOCOL_PHASE2B.md section 2.

M1  Early-fusion SE      : x' = a(x) * x on the raw leads, then the M0 backbone (CNN -> LSTM -> mean -> linear).
M2  Separate leads       : 12 independent lead encoders (CNN x3 -> LSTM), r_l (B,12,D); z = mean_l r_l; Linear(D,5).
M3  Channel-specific SE  : M2 + one SE gate a(r) in (0,1)^12 shared by all classes; z = mean_l a_l r_l.
M4  Multi-head SE        : M2 + five SE gates a^c and five single-output heads; z^c = mean_l a^c_l r_l.
Lead encoders are independent (grouped convolutions + one LSTM per lead): no weights are shared between leads.
"""
import torch
import torch.nn as nn

from config import CFG
from config_2b import ARCH
from models import ECGNet


class SEGate(nn.Module):
    """Squeeze-and-excitation over the 12 leads: s (B,12) -> sigmoid(W2 relu(W1 s)) in (0,1)^12."""
    def __init__(self, n=12, r=ARCH["gate_bottleneck"]):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n, r), nn.Linear(r, n)

    def forward(self, s):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))


class LeadEncoder(nn.Module):
    """12 fully independent encoders: grouped Conv1d (groups = 12) + BatchNorm + ReLU + MaxPool, three blocks,
    then one LSTM per lead. Output r (B, 12, D): time-mean of each lead's LSTM output."""
    def __init__(self, n_leads=12, c1=ARCH["c1"], c2=ARCH["c2"], c3=ARCH["c3"], k=ARCH["k"]):
        super().__init__()
        self.n, self.c3 = n_leads, c3

        def blk(cin, cout, kk):
            return nn.Sequential(nn.Conv1d(n_leads * cin, n_leads * cout, kk, padding=kk // 2, groups=n_leads),
                                 nn.BatchNorm1d(n_leads * cout), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.cnn = nn.Sequential(blk(1, c1, k[0]), blk(c1, c2, k[1]), blk(c2, c3, k[2]))
        self.rnn = nn.ModuleList([nn.LSTM(c3, c3, batch_first=True) for _ in range(n_leads)])

    def forward(self, x):                                   # x (B, 12, T)
        h = self.cnn(x)                                     # (B, 12*c3, T/8)
        h = h.view(x.size(0), self.n, self.c3, -1)          # (B, 12, c3, T')
        out = [self.rnn[l](h[:, l].transpose(1, 2))[0].mean(1) for l in range(self.n)]   # each (B, c3)
        return torch.stack(out, 1)                          # (B, 12, c3)


class LeadNet(nn.Module):
    def __init__(self, kind, n_classes=5, n_leads=12):
        super().__init__()
        assert kind in ("m1", "m2", "m3", "m4"), kind
        self.kind, self.C, self.L = kind, n_classes, n_leads
        drop = ARCH["dropout"]
        if kind == "m1":
            self.gate = SEGate(n_leads)
            self.backbone = ECGNet(temporal="lstm", mhsa=False, n_classes=n_classes, n_leads=n_leads,
                                   d=CFG["d_model"], heads=CFG["heads"], dropout=drop,
                                   attn_dropout=CFG["attn_dropout"])
            return
        D = ARCH["c3"]
        self.enc = LeadEncoder(n_leads)
        self.drop = nn.Dropout(drop)
        if kind in ("m2", "m3"):
            self.head = nn.Linear(D, n_classes)
        if kind == "m3":
            self.gate = SEGate(n_leads)
        if kind == "m4":
            self.gates = nn.ModuleList([SEGate(n_leads) for _ in range(n_classes)])
            self.heads = nn.ModuleList([nn.Linear(D, 1) for _ in range(n_classes)])

    # ---- gate weights (B, C, 12); for M2 all ones; for M1 / M3 identical across classes
    def _weights(self, x):
        if self.kind == "m1":
            s = x.pow(2).mean(-1).sqrt()                        # RMS of each raw lead
            return self.gate(s), None
        r = self.enc(x)
        s = r.mean(-1)                                          # squeeze: mean over features, (B, 12)
        if self.kind == "m2":
            return torch.ones(x.size(0), self.C, self.L, device=x.device), r
        if self.kind == "m3":
            return self.gate(s).unsqueeze(1).expand(-1, self.C, -1), r
        return torch.stack([g(s) for g in self.gates], 1), r

    def forward(self, x):
        if self.kind == "m1":
            a = self.gate(x.pow(2).mean(-1).sqrt())             # (B, 12)
            return self.backbone(x * a.unsqueeze(-1))
        a, r = self._weights(x)                                 # a (B, C, 12), r (B, 12, D)
        z = (a.unsqueeze(-1) * r.unsqueeze(1)).mean(2)          # (B, C, D): z^c = mean_l a^c_l r_l
        z = self.drop(z)
        if self.kind == "m4":
            return torch.cat([self.heads[c](z[:, c]) for c in range(self.C)], 1)
        return self.head(z[:, 0])                               # M2/M3: z identical over c (a is shared)

    @torch.no_grad()
    def lead_weights(self, x):
        """Gate weights as (B, C, 12) in eval mode. M1: gate on the raw leads (same for all classes)."""
        was = self.training
        self.eval()
        if self.kind == "m1":
            a = self.gate(x.pow(2).mean(-1).sqrt()).unsqueeze(1).expand(-1, self.C, -1)
        else:
            a = self._weights(x)[0]
        self.train(was)
        return a
