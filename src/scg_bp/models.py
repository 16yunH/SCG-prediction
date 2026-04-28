from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_channels: int
    window_size: int
    cnn_channels: list[int]
    lstm_hidden: int
    lstm_layers: int
    mlp_hidden: list[int]
    dropout: float


class CnnEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for out_c in channels:
            layers.extend(
                [
                    nn.Conv1d(c, out_c, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout),
                ]
            )
            c = out_c
        self.net = nn.Sequential(*layers)
        self.out_dim = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z.mean(dim=-1)


class LstmEncoder(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        return out[:, -1, :]


class MlpHead(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cnn = CnnEncoder(cfg.input_channels, cfg.cnn_channels, cfg.dropout)
        self.lstm = LstmEncoder(cfg.input_channels, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout)
        self.head = MlpHead(self.cnn.out_dim + self.lstm.out_dim, cfg.mlp_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cnn(x)
        b = self.lstm(x)
        z = torch.cat([a, b], dim=1)
        return self.head(z)


class CnnOnlyModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cnn = CnnEncoder(cfg.input_channels, cfg.cnn_channels, cfg.dropout)
        self.head = MlpHead(self.cnn.out_dim, cfg.mlp_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))


class LstmOnlyModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.lstm = LstmEncoder(cfg.input_channels, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout)
        self.head = MlpHead(self.lstm.out_dim, cfg.mlp_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x))


class MlpOnlyModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = MlpHead(cfg.input_channels * cfg.window_size, cfg.mlp_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        return self.head(flat)


def build_model(model_name: str, cfg: ModelConfig) -> nn.Module:
    table = {
        "full": FullModel,
        "cnn_only": CnnOnlyModel,
        "lstm_only": LstmOnlyModel,
        "mlp_only": MlpOnlyModel,
    }
    if model_name not in table:
        raise ValueError(f"Unknown model: {model_name}")
    return table[model_name](cfg)
