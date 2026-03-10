"""
lstm_model.py
Sequence-to-scalar LSTM that predicts Remaining Useful Life (RUL)
from a sliding window of multivariate sensor readings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model architecture ─────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.norm  = nn.LayerNorm(hidden_size)
        self.head  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out    = self.norm(out[:, -1, :])   # last timestep
        return self.head(out).squeeze(-1)


# ── Public wrapper ─────────────────────────────────────────────────────────

class LSTMModel:
    """
    Wraps _LSTMNet with fit / predict / fine_tune / save / load methods.

    Parameters
    ----------
    input_size  : number of sensor features
    seq_len     : sliding window length (timesteps)
    hidden_size : LSTM hidden dimension
    num_layers  : stacked LSTM layers
    """

    def __init__(
        self,
        input_size:  int   = 14,
        seq_len:     int   = 30,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        lr:          float = 1e-3,
    ):
        self.input_size  = input_size
        self.seq_len     = seq_len
        self.net         = _LSTMNet(input_size, hidden_size, num_layers, dropout).to(DEVICE)
        self.criterion   = nn.MSELoss()
        self.optimizer   = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5
        )
        self._trained    = False

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,   # (N, seq_len, input_size)
        y: np.ndarray,   # (N,)  RUL targets
        epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.1,
    ) -> dict:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        split   = int(len(X_t) * (1 - val_split))
        ds_tr   = TensorDataset(X_t[:split], y_t[:split])
        ds_val  = TensorDataset(X_t[split:], y_t[split:])
        dl_tr   = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_val  = DataLoader(ds_val, batch_size=batch_size)

        history = {"train_loss": [], "val_loss": []}
        self.net.train()

        for epoch in range(1, epochs + 1):
            tr_loss = self._run_epoch(dl_tr, train=True)
            vl_loss = self._run_epoch(dl_val, train=False)
            self.scheduler.step(vl_loss)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)

            if epoch % 10 == 0:
                logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, epochs, tr_loss, vl_loss)

        self._trained = True
        return history

    def fine_tune(self, X: np.ndarray, epochs: int = 10) -> float:
        """Lightweight fine-tune on recent data; returns final loss."""
        y_dummy = np.zeros(len(X), dtype=np.float32)
        history = self.fit(X, y_dummy, epochs=epochs, batch_size=32)
        return history["train_loss"][-1]

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> float:
        """Predict RUL for a single feature vector or sequence."""
        self.net.eval()
        with torch.no_grad():
            if x.ndim == 1:
                # Repeat single vector to fill window
                x = np.tile(x, (self.seq_len, 1))
            tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return float(self.net(tensor).item())

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        logger.info("LSTMModel saved → %s", path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        self._trained = True
        logger.info("LSTMModel loaded ← %s", path)

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.net.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = self.net(xb)
                loss = self.criterion(pred, yb)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optimizer.step()
                total += loss.item() * len(xb)
        return total / len(loader.dataset)
