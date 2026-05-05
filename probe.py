"""
probe.py — Hallucination probe classifier (Logistic Regression).

Why LogReg instead of an MLP:
    Probing literature (Azaria & Mitchell 2023, Burns et al. 2022, SAPLMA)
    consistently finds that L2-regularised logistic regression matches or
    beats MLP probes on hidden-state features when training data is small
    (here: ~550 samples per fold).  An MLP also tends to memorise the train
    set within the first dozen epochs, inflating train AUROC without
    improving generalisation.

Pipeline:
    StandardScaler -> optional PCA(256) -> LogisticRegression (L2, C tuned).
    PCA activates when feature dim > 512 (the multi-layer aggregation
    produces ~10752 features).

The class still inherits from ``nn.Module`` to keep the original interface
contract from ``solution.py`` unchanged, but the network is unused —
predictions go through scikit-learn.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


_PCA_TRIGGER_DIM = 512
_PCA_COMPONENTS = 256
_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
_SEED = 42


class HallucinationProbe(nn.Module):
    """Logistic-regression probe with optional PCA preprocessing."""

    def __init__(self) -> None:
        super().__init__()
        # nn.Module compatibility — unused at inference but kept so that
        # external code calling forward() does not crash.
        self._net: nn.Sequential | None = None

        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # nn.Module compatibility
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            self._net = nn.Sequential(nn.Linear(x.shape[-1], 1))
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.fit_transform(X)
        if X_scaled.shape[1] > _PCA_TRIGGER_DIM:
            n_comp = min(_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=_SEED)
            return self._pca.fit_transform(X_scaled)
        self._pca = None
        return X_scaled

    def _preprocess_apply(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            return self._pca.transform(X_scaled)
        return X_scaled

    # ------------------------------------------------------------------
    # Training — picks the best C on a stratified inner CV by AUROC
    # ------------------------------------------------------------------
    def _select_best_C(self, X: np.ndarray, y: np.ndarray) -> float:
        """3-fold inner CV to pick the L2 strength."""
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=_SEED)
        best_C = 1.0
        best_score = -1.0
        for C in _C_GRID:
            fold_scores = []
            for idx_tr, idx_vl in skf.split(X, y):
                clf = LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=_SEED,
                )
                clf.fit(X[idx_tr], y[idx_tr])
                probs = clf.predict_proba(X[idx_vl])[:, 1]
                try:
                    fold_scores.append(roc_auc_score(y[idx_vl], probs))
                except ValueError:
                    fold_scores.append(0.5)
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
        return best_C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, PCA, and the LogisticRegression classifier."""
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)

        X_proc = self._preprocess_fit(X)
        y_arr = np.asarray(y).astype(np.int64)

        if len(np.unique(y_arr)) >= 2 and len(y_arr) >= 30:
            best_C = self._select_best_C(X_proc, y_arr)
        else:
            best_C = 1.0

        self._clf = LogisticRegression(
            C=best_C,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=_SEED,
        )
        self._clf.fit(X_proc, y_arr)
        return self

    # ------------------------------------------------------------------
    # Threshold tuning (for F1 / accuracy display only)
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")
        X_proc = self._preprocess_apply(X)
        return self._clf.predict_proba(X_proc)