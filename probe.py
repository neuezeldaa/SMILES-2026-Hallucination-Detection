"""
probe.py — Stable Linear Probe + Light Bagging + AUROC Optimization
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._nets: list[nn.Sequential] = []
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> nn.Sequential:
        # Легкий MLP: достаточно для линейно разделимых представлений LLM
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._nets:
            raise RuntimeError("Ensemble not built.")
        return self._nets[0](x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = []
        n_estimators = 5

        for i in range(n_estimators):
            net = self._build_network(X_scaled.shape[1])
            optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

            # Stratified-like subsampling instead of full bootstrap
            n_samples = len(X_t)
            indices = torch.randperm(n_samples)[:int(0.85 * n_samples)]
            X_sub = X_t[indices]
            y_sub = y_t[indices]

            net.train()
            for epoch in range(120):
                optimizer.zero_grad()
                logits = net(X_sub).squeeze(-1)
                loss = criterion(logits, y_sub)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            net.eval()
            self._nets.append(net)

        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]

        # Оптимизация под AUROC через Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        self._threshold = float(thresholds[best_idx])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        all_probs = []
        with torch.no_grad():
            for net in self._nets:
                logits = net(X_t).squeeze(-1)
                prob_pos = torch.sigmoid(logits).numpy()
                all_probs.append(prob_pos)

        mean_probs = np.mean(all_probs, axis=0)
        return np.stack([1.0 - mean_probs, mean_probs], axis=1)