"""Six Phase-1 models. Everything except the two switches is identical:
CNN front-end -> [LSTM | BiLSTM | none] -> [MHSA | none] -> mean-pool -> linear.

Width is matched: LSTM hidden = d, BiLSTM hidden = d/2 per direction, so every
variant passes a (B, T, d) sequence to MHSA and to the classifier. Parameter
counts still differ and are reported.
"""
import math

import torch
import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, d, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _block(cin, cout, k):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class ECGNet(nn.Module):
    def __init__(self, temporal="none", mhsa=False, n_classes=5, n_leads=12,
                 d=128, heads=4, dropout=0.2, attn_dropout=0.1):
        super().__init__()
        # (B, 12, 1000) -> (B, d, 125)
        self.cnn = nn.Sequential(_block(n_leads, 64, 7), _block(64, d, 5), _block(d, d, 5))
        self.rnn = None
        if temporal == "lstm":
            self.rnn = nn.LSTM(d, d, batch_first=True)
        elif temporal == "bilstm":
            self.rnn = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.attn = None
        if mhsa:
            self.pos = PosEnc(d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
            self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)          # (B, T, d)
        if self.rnn is not None:
            h, _ = self.rnn(h)
        if self.attn is not None:
            q = self.pos(h)
            a, _ = self.attn(q, q, q, need_weights=False)
            h = self.norm(h + a)                  # residual + LayerNorm
        return self.head(h.mean(dim=1))           # logits (B, n_classes)
