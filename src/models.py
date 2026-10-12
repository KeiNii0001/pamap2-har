"""時系列分類モデル。入力は [バッチ, チャネル, 時間]。"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """1D-CNN。畳み込み3段 + 全時間平均プーリング + 全結合。

    全時間平均プーリング(GAP)にしているので、窓の長さを変えても
    全結合層の形が変わらない。
    """

    def __init__(self, in_ch: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width, 7),
            block(width, width * 2, 5),
            block(width * 2, width * 2, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)          # [B, 2W, L/8]
        h = h.mean(dim=-1)            # GAP
        return self.head(h)


class LSTMClassifier(nn.Module):
    """1D畳み込みで時間を1/4に間引いてから、双方向LSTMにかける。

    100Hz・200サンプルをそのままLSTMに入れると系列が長く学習が遅いので、
    先に畳み込みで局所特徴を作って短くする。
    """

    def __init__(self, in_ch: int, n_classes: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.front(x).transpose(1, 2)   # [B, L/4, H]
        out, _ = self.lstm(h)
        return self.head(out.mean(dim=1))


def build_model(name: str, in_ch: int, n_classes: int) -> nn.Module:
    if name == "cnn":
        return CNN1D(in_ch, n_classes)
    if name == "lstm":
        return LSTMClassifier(in_ch, n_classes)
    raise ValueError(f"unknown model: {name}")
