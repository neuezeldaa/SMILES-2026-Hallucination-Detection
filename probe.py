"""
probe.py — Multi-seed LogReg ensemble (auto-detects optional GEO slice).

If aggregation.SLICE_INFO contains a GEO key (i.e. solution.py was run
with USE_GEOMETRIC=True), a fourth sub-probe is added that operates on
the geometric-feature slice.

This file is for the ablation experiment "what if geometric features
are properly wired into the ensemble".  Use the standard probe.py in
production.
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

from aggregation import SLICE_INFO


_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
_PCA_TRIGGER_DIM = 4096
_PCA_COMPONENTS = 256
_SEEDS = (42, 7, 123, 2024, 31)


class _SubProbe:
    def __init__(self, slice_key: str, use_pca: bool) -> None:
        self.slice_key = slice_key
        self.use_pca = use_pca
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.clfs: list[LogisticRegression] = []
        self.best_Cs: list[float] = []

    def _slice(self, X: np.ndarray) -> np.ndarray:
        sl = SLICE_INFO[self.slice_key]
        return X[:, sl]

    @staticmethod
    def _select_best_C(X: np.ndarray, y: np.ndarray, seed: int) -> float:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        best_C, best_score = 1.0, -1.0
        for C in _C_GRID:
            scores = []
            for idx_tr, idx_vl in skf.split(X, y):
                clf = LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced",
                    solver="lbfgs", random_state=seed,
                )
                clf.fit(X[idx_tr], y[idx_tr])
                probs = clf.predict_proba(X[idx_vl])[:, 1]
                try:
                    scores.append(roc_auc_score(y[idx_vl], probs))
                except ValueError:
                    scores.append(0.5)
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best_C = m, C
        return best_C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SubProbe":
        Xs = self._slice(X)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(Xs)

        if self.use_pca and Xs.shape[1] > _PCA_TRIGGER_DIM:
            n_comp = min(_PCA_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=42)
            Xs = self.pca.fit_transform(Xs)
        else:
            self.pca = None

        self.clfs = []
        self.best_Cs = []

        for seed in _SEEDS:
            if len(np.unique(y)) >= 2 and len(y) >= 30:
                C = self._select_best_C(Xs, y, seed=seed)
            else:
                C = 1.0
            clf = LogisticRegression(
                C=C, max_iter=5000, class_weight="balanced",
                solver="lbfgs", random_state=seed,
            )
            clf.fit(Xs, y)
            self.clfs.append(clf)
            self.best_Cs.append(C)

        return self

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        if not self.clfs:
            raise RuntimeError(f"_SubProbe[{self.slice_key}] not fitted.")
        Xs = self._slice(X)
        Xs = self.scaler.transform(Xs)
        if self.pca is not None:
            Xs = self.pca.transform(Xs)
        probs_list = [clf.predict_proba(Xs)[:, 1] for clf in self.clfs]
        return np.mean(np.stack(probs_list), axis=0)


def _build_subprobes() -> list[_SubProbe]:
    """Build sub-probes based on what slices exist in SLICE_INFO.

    Always builds A, C, D2.  Adds a GEO sub-probe if SLICE_INFO contains
    the GEO key (which it does only when USE_GEOMETRIC=True).
    """
    keys = [("A", False), ("C", False), ("D2", False)]
    if "GEO" in SLICE_INFO:
        keys.append(("GEO", False))
    return [_SubProbe(k, p) for k, p in keys]


class HallucinationProbe(nn.Module):
    """Probability-level ensemble of multi-seed LogReg sub-probes."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._subprobes: list[_SubProbe] = []
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            self._net = nn.Sequential(nn.Linear(x.shape[-1], 1))
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        np.random.seed(42)
        torch.manual_seed(42)

        if not SLICE_INFO:
            raise RuntimeError(
                "aggregation.SLICE_INFO is empty; aggregate() must be called "
                "at least once before HallucinationProbe.fit()."
            )

        # Build sub-probes lazily based on what slices are present.
        if not self._subprobes:
            self._subprobes = _build_subprobes()

        y_arr = np.asarray(y).astype(np.int64)
        for sp in self._subprobes:
            sp.fit(X, y_arr)
        return self

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs_pos = np.mean(
            np.stack([sp.predict_proba_pos(X) for sp in self._subprobes]),
            axis=0,
        )
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)