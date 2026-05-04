"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; implements an internal ensemble (Bagging)
    for robustness on small datasets.  Architecture uses BatchNorm + Dropout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nets: list[nn.Sequential] = []  # Ensemble of models
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # STUDENT: Enhanced network definition with BatchNorm & Dropout
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> nn.Sequential:
        """Instantiate the network layers for a single ensemble member."""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``."""
        if not self._nets:
            raise RuntimeError(
                "Ensemble has not been built yet. Call fit() before forward()."
            )
        # Returns 1D logits for compatibility
        return self._nets[0](x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors using Bagging."""
        X_scaled = self._scaler.fit_transform(X)

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Weight positive examples by neg/pos ratio to handle class imbalance.
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = []
        n_estimators = 5  # Number of models in the ensemble

        for i in range(n_estimators):
            net = self._build_network(X_scaled.shape[1])
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

            # Bagging: Bootstrap sampling with replacement
            n_samples = len(X_t)
            indices = torch.randint(0, n_samples, (n_samples,))
            X_boot = X_t[indices]
            y_boot = y_t[indices]  # Shape: [batch]

            net.train()
            for epoch in range(150):
                optimizer.zero_grad()
                # FIX: Squeeze to [batch] to match y_boot shape
                logits = net(X_boot).squeeze(-1)
                loss = criterion(logits, y_boot)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            net.eval()
            self._nets.append(net)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

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
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates by averaging ensemble members."""
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()

        all_probs = []
        with torch.no_grad():
            for net in self._nets:
                # FIX: Squeeze to 1D for consistent probability calculation
                logits = net(X_t).squeeze(-1)
                prob_pos = torch.sigmoid(logits).numpy()
                all_probs.append(prob_pos)

        # Soft voting: average probabilities across the ensemble
        mean_probs = np.mean(all_probs, axis=0)
        return np.stack([1.0 - mean_probs, mean_probs], axis=1)