"""
probe.py — Hallucination probe classifier (LogReg ensemble of A/C/D).

Changes vs previous version:
  1. Global PCA(128) applied to the full 10752-d vector before splitting
     into sub-probes — reduces dimensionality and overfitting.
  2. Extended C grid down to 1e-5 — all folds chose C=0.01 previously,
     suggesting the optimum may be even lower.
  3. Each sub-probe now also applies an independent StandardScaler + PCA
     on its own slice (unchanged), but the global PCA is the key fix.

Architecture:
    raw features (10752-d)
        └─ GlobalScaler + GlobalPCA(128)  ← NEW
              ├─ SubProbe A  [slice 0:128 after PCA, no sub-PCA]
              ├─ SubProbe C  [slice 0:128 after PCA, no sub-PCA]
              └─ SubProbe D  [slice 0:128 after PCA, no sub-PCA]
                   └─ avg(proba_A, proba_C, proba_D)

Note: after global PCA the slice boundaries from SLICE_INFO no longer
apply — all three sub-probes operate on the same 128-d representation
but with independently tuned C values.  The ensemble diversity now comes
from regularisation differences, not feature differences.  If you want
to preserve feature diversity, set USE_GLOBAL_PCA = False.
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


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# Set True to apply a single PCA to the full vector before sub-probes.
# Fixes the high-dimensionality / overfitting problem.
USE_GLOBAL_PCA = False
GLOBAL_PCA_COMPONENTS = 512      # 128 << 465 train samples per fold

# Extended C grid — goes lower than before (previous optimum was 0.01
# across all folds, suggesting we need more regularisation options).
_C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

_PCA_TRIGGER_DIM = 512           # sub-probe PCA threshold (used when USE_GLOBAL_PCA=False)
_PCA_COMPONENTS  = 512          # sub-probe PCA components
_SEED = 42

# Sub-probes: when USE_GLOBAL_PCA=True all three see the same 128-d input,
# diversity comes from independent C tuning.
_SUBPROBES = (
    ("A", False),
    ("C", False),
    ("D", True),   # no sub-PCA needed after global PCA
)


class _SubProbe:
    """Single LogReg pipeline on one feature slice (or full vector after global PCA)."""

    def __init__(self, slice_key: str, use_pca: bool) -> None:
        self.slice_key = slice_key
        self.use_pca   = use_pca
        self.scaler    = StandardScaler()
        self.pca: PCA | None = None
        self.clf: LogisticRegression | None = None
        self.best_C: float = 1e-3

    def _slice(self, X: np.ndarray) -> np.ndarray:
        if USE_GLOBAL_PCA or self.slice_key not in SLICE_INFO:
            return X          # global PCA already reduced; use full vector
        return X[:, SLICE_INFO[self.slice_key]]

    def _select_best_C(self, X: np.ndarray, y: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=_SEED)
        best_C, best_score = _C_GRID[0], -1.0
        for C in _C_GRID:
            scores = []
            for idx_tr, idx_vl in skf.split(X, y):
                clf = LogisticRegression(
                    C=C, max_iter=2000, class_weight="balanced",
                    solver="lbfgs", random_state=_SEED,
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
        Xs = self.scaler.fit_transform(Xs)

        # Sub-level PCA only when global PCA is disabled and dim is large
        if not USE_GLOBAL_PCA and self.use_pca and Xs.shape[1] > _PCA_TRIGGER_DIM:
            n_comp = min(_PCA_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=_SEED)
            Xs = self.pca.fit_transform(Xs)
        else:
            self.pca = None

        if len(np.unique(y)) >= 2 and len(y) >= 30:
            self.best_C = self._select_best_C(Xs, y)
        else:
            self.best_C = 1e-3

        print(f"    SubProbe-{self.slice_key}: best_C={self.best_C}")

        self.clf = LogisticRegression(
            C=self.best_C, max_iter=5000, class_weight="balanced",
            solver="lbfgs", random_state=_SEED,
        )
        self.clf.fit(Xs, y)
        return self

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        Xs = self._slice(X)
        Xs = self.scaler.transform(Xs)
        if self.pca is not None:
            Xs = self.pca.transform(Xs)
        return self.clf.predict_proba(Xs)[:, 1]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
class HallucinationProbe(nn.Module):
    """Probability-level ensemble of three LogReg sub-probes (A, C, D).

    With USE_GLOBAL_PCA=True:
        raw (10752-d) → GlobalScaler → GlobalPCA(128) → 3×LogReg → avg proba

    With USE_GLOBAL_PCA=False (legacy):
        raw (10752-d) → slice A/C/D → each sub-probe scales+[PCA]+LogReg → avg proba
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None

        self._global_scaler: StandardScaler | None = None
        self._global_pca: PCA | None = None

        self._subprobes: list[_SubProbe] = [
            _SubProbe(key, use_pca) for key, use_pca in _SUBPROBES
        ]
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            self._net = nn.Sequential(nn.Linear(x.shape[-1], 1))
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    def _global_transform(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Apply global scaling + PCA if enabled."""
        if not USE_GLOBAL_PCA:
            return X
        if fit:
            self._global_scaler = StandardScaler()
            X = self._global_scaler.fit_transform(X)
            n_comp = min(GLOBAL_PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
            self._global_pca = PCA(n_components=n_comp, random_state=_SEED)
            X = self._global_pca.fit_transform(X)
            print(f"  GlobalPCA: {self._global_pca.n_components_} components, "
                  f"explained variance = "
                  f"{self._global_pca.explained_variance_ratio_.sum()*100:.1f}%")
        else:
            X = self._global_scaler.transform(X)
            X = self._global_pca.transform(X)
        return X

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)

        if not USE_GLOBAL_PCA and not SLICE_INFO:
            raise RuntimeError(
                "aggregation.SLICE_INFO is empty; aggregate() must be called "
                "at least once before HallucinationProbe.fit()."
            )

        y_arr = np.asarray(y).astype(np.int64)
        X_t = self._global_transform(X, fit=True)

        for sp in self._subprobes:
            sp.fit(X_t, y_arr)
        return self

    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )
        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score((probs >= t).astype(int), y_val, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, float(t)
        self._threshold = best_threshold
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = self._global_transform(X, fit=False)
        probs_pos = np.mean(
            np.stack([sp.predict_proba_pos(X_t) for sp in self._subprobes]),
            axis=0,
        )
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)