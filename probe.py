"""
probe.py — Hallucination probe classifier (LogReg ensemble of A/C/D).

Final design — probability-level ensemble of three logistic-regression
sub-probes, each fitted on a different aggregation slice of the same
hidden-state extraction:

    A : layer 15, last token only       (896-d)
    C : layer 15, last + mean + max     (2688-d)
    D : layers 13/14/15/16, last + mean (7168-d)  -> with PCA(256)

Inference:
    predict_proba averages the three sub-probes' positive-class probabilities.

Why an ensemble:
    Single sub-probes scored 70.4-72.2% AUROC with std 4.3-7.1% across
    folds.  Probability averaging keeps the shared truthfulness signal
    while cancelling the pooling-specific noise — diagnose_v3 measured
    the A+C+D ensemble at 73.07% ± 5.47%, the highest mean-minus-std
    across all configurations tested.

Slice boundaries are read from ``aggregation.SLICE_INFO`` so the probe
stays in sync with the aggregator without hard-coding dimensions.
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
_PCA_TRIGGER_DIM = 512
_PCA_COMPONENTS = 256
_SEED = 42

# Sub-probes participating in the ensemble.  A sub-probe is identified by
# the slice key it reads from SLICE_INFO; ``use_pca`` is enabled for the
# higher-dimensional D slice.
_SUBPROBES = (
    ("A", False),
    ("C", False),
    ("D", True),
)


class _SubProbe:
    """Single LogReg pipeline on one feature slice."""

    def __init__(self, slice_key: str, use_pca: bool) -> None:
        self.slice_key = slice_key
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.clf: LogisticRegression | None = None
        self.best_C: float = 1.0

    # ------------------------------------------------------------------
    def _slice(self, X: np.ndarray) -> np.ndarray:
        sl = SLICE_INFO[self.slice_key]
        # The aggregation feature vector may be longer than `D.stop` if
        # USE_GEOMETRIC=True appended geometric features; we still slice
        # only the relevant block.
        return X[:, sl]

    def _select_best_C(self, X: np.ndarray, y: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=_SEED)
        best_C, best_score = 1.0, -1.0
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

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SubProbe":
        Xs = self._slice(X)
        Xs = self.scaler.fit_transform(Xs)

        if self.use_pca and Xs.shape[1] > _PCA_TRIGGER_DIM:
            n_comp = min(_PCA_COMPONENTS, Xs.shape[0] - 1, Xs.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=_SEED)
            Xs = self.pca.fit_transform(Xs)
        else:
            self.pca = None

        if len(np.unique(y)) >= 2 and len(y) >= 30:
            self.best_C = self._select_best_C(Xs, y)
        else:
            self.best_C = 1.0

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
    """Probability-level ensemble of three LogReg sub-probes (A, C, D)."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # nn.Module compatibility

        self._subprobes: list[_SubProbe] = [
            _SubProbe(key, use_pca) for key, use_pca in _SUBPROBES
        ]
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # nn.Module compatibility — ``forward`` is unused at inference but
    # exists so that any external code that calls it does not crash.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            self._net = nn.Sequential(nn.Linear(x.shape[-1], 1))
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)

        if not SLICE_INFO:
            raise RuntimeError(
                "aggregation.SLICE_INFO is empty; aggregate() must be called "
                "at least once before HallucinationProbe.fit()."
            )

        y_arr = np.asarray(y).astype(np.int64)
        for sp in self._subprobes:
            sp.fit(X, y_arr)
        return self

    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        AUROC (the competition metric) is rank-based and unaffected; this
        is here purely so the printed Accuracy/F1 are meaningful.
        """
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
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Average positive-class probabilities across sub-probes.
        probs_pos = np.mean(
            np.stack([sp.predict_proba_pos(X) for sp in self._subprobes]),
            axis=0,
        )
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)