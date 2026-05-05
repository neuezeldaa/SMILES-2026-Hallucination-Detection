"""
probe.py — Hallucination probe classifier.

Pipeline:
    StandardScaler -> optional PCA(256) -> 2-hidden-layer MLP with dropout.

PCA is enabled automatically when the input feature dimensionality exceeds
512.  This is necessary for the multi-layer aggregation strategy in
``aggregation.py`` (which produces ~10 752 features) given only ~580 training
samples.  Without dimensionality reduction the MLP overfits in a few epochs.

Training:
    AdamW (weight_decay=1e-2), mini-batch SGD (size 64), pos-weighted BCE,
    early stopping on internal validation AUROC (the competition metric).
    A fixed seed makes the run reproducible end-to-end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_PCA_TRIGGER_DIM = 512
_PCA_COMPONENTS = 256
_INNER_VAL_FRAC = 0.15
_BATCH_SIZE = 64
_MAX_EPOCHS = 300
_PATIENCE = 25
_LR = 1e-3
_WEIGHT_DECAY = 1e-2
_SEED = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HallucinationProbe(nn.Module):
    """MLP probe that detects hallucinations from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the MLP once ``input_dim`` is known."""
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``."""
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler + optional PCA, return processed features."""
        X_scaled = self._scaler.fit_transform(X)
        if X_scaled.shape[1] > _PCA_TRIGGER_DIM:
            n_comp = min(_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=_SEED)
            return self._pca.fit_transform(X_scaled)
        self._pca = None
        return X_scaled

    def _preprocess_apply(self, X: np.ndarray) -> np.ndarray:
        """Transform features with the fitted scaler + PCA."""
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            return self._pca.transform(X_scaled)
        return X_scaled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe with early stopping on internal validation AUROC."""
        _set_seed(_SEED)

        X_proc = self._preprocess_fit(X)
        self._build_network(X_proc.shape[1])

        # Internal hold-out for early stopping.  Falls back to full data if
        # stratification is impossible (single class or tiny set).
        y_arr = np.asarray(y).astype(np.int64)
        try:
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X_proc, y_arr,
                test_size=_INNER_VAL_FRAC,
                stratify=y_arr,
                random_state=_SEED,
            )
            has_inner_val = True
        except ValueError:
            X_tr, y_tr = X_proc, y_arr
            X_vl, y_vl = X_proc, y_arr
            has_inner_val = False

        X_tr_t = torch.from_numpy(X_tr).float()
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
        X_vl_t = torch.from_numpy(X_vl).float()

        # Pos-weighted BCE handles the 70/30 class imbalance.
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY
        )

        best_auroc = -1.0
        best_state: dict | None = None
        patience = 0

        n_train = len(X_tr_t)

        for _ in range(_MAX_EPOCHS):
            self.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, _BATCH_SIZE):
                ix = perm[i : i + _BATCH_SIZE]
                optimizer.zero_grad()
                logits = self(X_tr_t[ix])
                loss = criterion(logits, y_tr_t[ix])
                loss.backward()
                optimizer.step()

            # Early-stop on AUROC (the competition metric).
            self.eval()
            with torch.no_grad():
                val_logits = self(X_vl_t)
                val_probs = torch.sigmoid(val_logits).numpy()
            try:
                cur_auroc = roc_auc_score(y_vl, val_probs)
            except ValueError:
                cur_auroc = 0.0

            if cur_auroc > best_auroc:
                best_auroc = cur_auroc
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if has_inner_val and patience >= _PATIENCE:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        return self

    # ------------------------------------------------------------------
    # Threshold tuning (used for F1 / accuracy display only)
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        AUROC is rank-based and unaffected by the threshold; this is here
        purely so that the printed Accuracy/F1 in the evaluation summary are
        meaningful.
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
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates of shape ``(n_samples, 2)``."""
        X_proc = self._preprocess_apply(X)
        X_t = torch.from_numpy(X_proc).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)