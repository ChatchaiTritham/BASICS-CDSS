"""Torch-based sequence classifiers (LSTM / TCN) for trajectory-level CDSS evaluation.

These models consume per-twin time series of shape ``(n, T, d)`` -- the full
24-hour digital-twin trajectory rather than a single initial-state row -- and
expose a small scikit-learn-style API (``fit`` / ``predict_proba`` / ``predict``,
plus ``classes_``) so they integrate with the same static-vs-temporal evaluation,
calibration, decision-curve, and conformal machinery the tabular models use.

Determinism: every model seeds ``torch.manual_seed`` and ``numpy`` from a single
seed (42 in the driver) and trains single-threaded on CPU with a fixed number of
full-batch epochs, so two runs produce identical probabilities. Architectures are
deliberately modest (LSTM: 1-2 layers, hidden 64; TCN: a dilated Conv1d stack)
and train in seconds on a commodity CPU.

Input convention
----------------
``fit(X_seq, y)`` / ``predict_proba(X_seq)`` take ``X_seq`` shaped ``(n, T, d)``.
Features are standardized per channel using statistics estimated on the training
trajectories (stored at fit time and reused at inference). NaNs are not expected
(the driver median-imputes disease-disjoint columns before calling), but any
residual NaN is replaced with the per-channel training mean for safety.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # torch is an optional but now-declared dependency (see requirements.txt)
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _TORCH_AVAILABLE = False


def _seed_everything(seed: int) -> None:
    """Seed numpy + torch and force deterministic, single-threaded CPU math."""
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    """Modest LSTM encoder + linear head over the last hidden state."""

    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (n, T, d)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final-timestep hidden state
        return self.head(last).squeeze(-1)  # logits (n,)


class _TCNNet(nn.Module if _TORCH_AVAILABLE else object):
    """Dilated temporal convolutional network (causal Conv1d stack)."""

    def __init__(self, n_features: int, channels: int = 64, n_blocks: int = 3,
                 kernel_size: int = 3):
        super().__init__()
        layers = []
        in_ch = n_features
        for b in range(n_blocks):
            dilation = 2 ** b
            pad = (kernel_size - 1) * dilation  # causal left padding
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size,
                          padding=pad, dilation=dilation),
                _Chomp(pad),  # trim right padding to keep length, stay causal
                nn.ReLU(),
            ]
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):  # x: (n, T, d)
        h = x.transpose(1, 2)  # -> (n, d, T) for Conv1d
        h = self.tcn(h)        # (n, channels, T)
        last = h[:, :, -1]     # final-timestep features
        return self.head(last).squeeze(-1)  # logits (n,)


class _Chomp(nn.Module if _TORCH_AVAILABLE else object):
    """Remove the trailing ``chomp`` time steps left by causal padding."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TorchSequenceClassifier:
    """scikit-learn-style wrapper around an LSTM or TCN over ``(n, T, d)`` input.

    Parameters
    ----------
    arch : {"lstm", "tcn"}
        Which architecture to build.
    seed : int
        Seed for numpy + torch (determinism).
    hidden : int
        Hidden / channel width.
    epochs : int
        Full-batch training epochs.
    lr : float
        Adam learning rate.
    """

    def __init__(self, arch: str = "lstm", seed: int = 42, hidden: int = 64,
                 epochs: int = 60, lr: float = 0.01, num_layers: int = 1):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "torch is required for TorchSequenceClassifier; install torch."
            )
        self.arch = arch
        self.seed = seed
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.num_layers = num_layers
        self.classes_ = np.array([0, 1])
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._net = None

    def _standardize(self, X_seq: np.ndarray) -> np.ndarray:
        X = np.asarray(X_seq, dtype=np.float32)
        # Replace residual NaNs with per-channel training mean.
        if self._mean is not None:
            inds = np.where(np.isnan(X))
            if inds[0].size:
                X[inds] = self._mean[inds[2]]
        Xs = (X - self._mean) / self._std
        return Xs.astype(np.float32)

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "TorchSequenceClassifier":
        _seed_everything(self.seed)
        X = np.asarray(X_seq, dtype=np.float32)  # (n, T, d)
        y = np.asarray(y, dtype=np.float32)
        # Per-channel standardization stats from the training trajectories.
        flat = X.reshape(-1, X.shape[-1])
        self._mean = np.nanmean(flat, axis=0).astype(np.float32)
        std = np.nanstd(flat, axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self._std = std
        self.classes_ = np.unique(y).astype(int)
        if self.classes_.size == 1:  # degenerate guard
            self.classes_ = np.array([0, 1])

        Xs = self._standardize(X)
        n_features = Xs.shape[-1]
        if self.arch == "lstm":
            self._net = _LSTMNet(n_features, hidden=self.hidden,
                                 num_layers=self.num_layers)
        elif self.arch == "tcn":
            self._net = _TCNNet(n_features, channels=self.hidden)
        else:
            raise ValueError(f"unknown arch {self.arch!r}")

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y)
        # Class-balanced positive weight so the rarer class is not ignored.
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        pos_weight = torch.tensor(
            [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._net(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
        self._net.eval()
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        Xs = self._standardize(X_seq)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xs))
            p1 = torch.sigmoid(logits).numpy().astype(float)
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_seq)[:, 1] >= 0.5).astype(int)
