"""The four architectures compared in this project.

Every model takes x of shape [batch, time=1000, leads=12] (standardized) and returns
logits of shape [batch, n_classes]. Each standalone model also exposes ``embed(x)``,
a fixed-size feature vector, which the fusion model reuses as its branches.

* CNN1D            – 1D ResNet: local waveform morphology (QRS shape, ST changes).
* BiLSTMNet        – bidirectional LSTM with attention pooling: rhythm / temporal order.
* ECGTransformer   – patch tokens + self-attention: long-range relations across the 10 s.
* ParallelFusionNet – runs the three branches in parallel on the same ECG, fuses their
                      embeddings with a learned per-ECG gate, and also trains an auxiliary
                      head per branch (deep supervision).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ["cnn", "lstm", "transformer", "fusion"]
DISPLAY_NAMES = {"cnn": "CNN (1D ResNet)", "lstm": "LSTM (BiLSTM)", "transformer": "Transformer",
                 "fusion": "Parallel Fusion (CNN+LSTM+Transformer)", "ensemble": "Late-fusion Ensemble"}


# ----------------------------------------------------------------------------- CNN
class ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, k: int = 7, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class CNN1D(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, width: int = 64, kernel_size: int = 7,
                 dropout: float = 0.2, **_):
        super().__init__()
        w, k = width, kernel_size
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, w, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(w), nn.ReLU(),
            nn.MaxPool1d(2))
        self.blocks = nn.Sequential(
            ResBlock1d(w, w, 1, k, dropout / 2),
            ResBlock1d(w, 2 * w, 2, k, dropout / 2),
            ResBlock1d(2 * w, 2 * w, 1, k, dropout / 2),
            ResBlock1d(2 * w, 2 * w, 2, k, dropout / 2),
            ResBlock1d(2 * w, 2 * w, 1, k, dropout / 2),
        )
        self.emb_dim = 2 * w
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.emb_dim, n_classes)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, channels, T'] (used by Grad-CAM)."""
        return self.blocks(self.stem(x.transpose(1, 2)))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))


# ----------------------------------------------------------------------------- LSTM
class BiLSTMNet(nn.Module):
    """A single strided convolution ("minimal stem") shortens 1000 steps to 250 so the LSTM
    trains in reasonable time; all sequence modelling is done by the bidirectional LSTM."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5, stem_channels: int = 64, stem_stride: int = 4,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.2, **_):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, stem_channels, kernel_size=7, stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels), nn.ReLU())
        self.lstm = nn.LSTM(stem_channels, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.emb_dim = 2 * hidden
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.emb_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)        # [B, T', C]
        out, _ = self.lstm(h)                                    # [B, T', 2H]
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1)     # attention pooling over time
        return (out * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))


# ----------------------------------------------------------------------------- Transformer
class ECGTransformer(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, seq_len: int = 1000, d_model: int = 128,
                 patch_size: int = 10, layers: int = 4, heads: int = 8, ff_dim: int = 256, dropout: float = 0.1, **_):
        super().__init__()
        if seq_len % patch_size:
            raise ValueError("seq_len must be divisible by patch_size")
        self.patch = nn.Conv1d(n_leads, d_model, kernel_size=patch_size, stride=patch_size)
        n_tokens = seq_len // patch_size
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.emb_dim = d_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x.transpose(1, 2)).transpose(1, 2)   # [B, N, D]
        if tokens.shape[1] + 1 > self.pos.shape[1]:
            raise ValueError("Input is longer than the model's maximum sequence length.")
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        return self.norm(self.encoder(tokens)[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))


# ----------------------------------------------------------------------------- Fusion
class ParallelFusionNet(nn.Module):
    """CNN, LSTM and Transformer branches process the same ECG in parallel.

    1. Each branch produces an embedding, projected to a common size D.
    2. A small gating network scores every branch for THIS ECG; softmax gives weights
       alpha (they sum to 1 and are reported in the app as branch contributions).
    3. The weighted sum of branch embeddings goes to the classification head.
    4. Each branch also has an auxiliary head trained with the same labels, so no branch
       is starved of learning signal (deep supervision).
    5. Branch dropout: during training one branch is hidden for a random 10 % of samples,
       which discourages the model from relying on a single branch.
    """

    BRANCHES = ["cnn", "lstm", "transformer"]

    def __init__(self, branch_cfgs: dict, n_leads: int = 12, n_classes: int = 5, seq_len: int = 1000,
                 fusion_dim: int = 128, aux_weight: float = 0.3, branch_dropout: float = 0.1,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.branches = nn.ModuleDict({
            "cnn": CNN1D(n_leads, n_classes, **branch_cfgs["cnn"]),
            "lstm": BiLSTMNet(n_leads, n_classes, **branch_cfgs["lstm"]),
            "transformer": ECGTransformer(n_leads, n_classes, seq_len=seq_len, **branch_cfgs["transformer"]),
        })
        self.proj = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(b.emb_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
            for k, b in self.branches.items()})
        self.gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.Tanh(), nn.Linear(fusion_dim // 2, 1))
        self.aux_heads = nn.ModuleDict({k: nn.Linear(fusion_dim, n_classes) for k in self.BRANCHES})
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, fusion_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(fusion_dim, n_classes))
        self.aux_weight = aux_weight
        self.branch_dropout = branch_dropout

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Returns (logits [B, K], aux logits per branch, alpha [B, 3] branch weights)."""
        z = torch.stack([self.proj[k](self.branches[k].embed(x)) for k in self.BRANCHES], dim=1)  # [B, 3, D]
        scores = self.gate(z).squeeze(-1)                                                          # [B, 3]
        if self.training and self.branch_dropout > 0:
            b = scores.shape[0]
            drop = torch.rand(b, device=scores.device) < self.branch_dropout
            which = torch.randint(0, len(self.BRANCHES), (b,), device=scores.device)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[torch.arange(b, device=scores.device), which] = drop
            scores = scores.masked_fill(mask, -1e4)
        alpha = torch.softmax(scores, dim=1)
        fused = (alpha.unsqueeze(-1) * z).sum(dim=1)
        logits = self.head(fused)
        aux = {k: self.aux_heads[k](z[:, i]) for i, k in enumerate(self.BRANCHES)}
        return logits, aux, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[0]


# ----------------------------------------------------------------------------- factory
def build_model(name: str, models_cfg: dict, n_leads: int = 12, n_classes: int = 5, seq_len: int = 1000) -> nn.Module:
    if name == "cnn":
        return CNN1D(n_leads, n_classes, **models_cfg["cnn"])
    if name == "lstm":
        return BiLSTMNet(n_leads, n_classes, **models_cfg["lstm"])
    if name == "transformer":
        return ECGTransformer(n_leads, n_classes, seq_len=seq_len, **models_cfg["transformer"])
    if name == "fusion":
        branch_cfgs = {k: models_cfg[k] for k in ParallelFusionNet.BRANCHES}
        return ParallelFusionNet(branch_cfgs, n_leads, n_classes, seq_len=seq_len, **models_cfg["fusion"])
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, logits). For the fusion model the auxiliary branch losses are added."""
    if isinstance(model, ParallelFusionNet):
        logits, aux, _ = model.forward_all(x)
        aux_loss = sum(criterion(a, y) for a in aux.values()) / len(aux)
        return criterion(logits, y) + model.aux_weight * aux_loss, logits
    logits = model(x)
    return criterion(logits, y), logits
