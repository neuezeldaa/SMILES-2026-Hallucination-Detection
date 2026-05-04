"""
probe.py — Hallucination probe classifier (student-implemented).
Optimized Deep MLP + Bagging for high-dimensional inputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._nets: list[nn.Sequential] = []  # Ensemble of models
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> nn.Sequential:
        """Architecture: Deep MLP with Dropout (No BatchNorm for stability)."""
        return nn.Sequential(
            # Вход 3584 -> 2048
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 2048 -> 1024
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            # 1024 -> 256
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Выход
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits."""
        if not self._nets:
            raise RuntimeError("Ensemble not built.")
        # Squeeze для совместимости размеров
        return self._nets[0](x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe using Bagging."""
        X_scaled = self._scaler.fit_transform(X)

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Вес для дисбаланса классов
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = []
        n_estimators = 5  # 5 моделей в ансамбле

        for i in range(n_estimators):
            net = self._build_network(X_scaled.shape[1])
            # LR 5e-4 оптимален для такой архитектуры
            optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            # Bootstrap sampling
            n_samples = len(X_t)
            indices = torch.randint(0, n_samples, (n_samples,))
            X_boot = X_t[indices]
            y_boot = y_t[indices]

            net.train()
            for epoch in range(200):
                optimizer.zero_grad()
                # Squeeze логитов, чтобы совпадали с y_boot
                logits = net(X_boot).squeeze(-1)
                loss = criterion(logits, y_boot)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            net.eval()
            self._nets.append(net)

        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        """Tune threshold for F1."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average probabilities from all ensemble members."""
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        all_probs = []
        with torch.no_grad():
            for net in self._nets:
                # Squeeze для корректного расчета sigmoid
                logits = net(X_t).squeeze(-1)
                prob_pos = torch.sigmoid(logits).numpy()
                all_probs.append(prob_pos)

        mean_probs = np.mean(all_probs, axis=0)
        return np.stack([1.0 - mean_probs, mean_probs], axis=1)