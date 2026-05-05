"""
splitting.py — Train / validation / test split utilities.

Group-aware K-fold cross-validation: samples that share the same context
(the passage between the system prompt and the "Note that..." marker) are
kept entirely within a single fold to prevent context leakage.

Why group-aware K-fold:
    The dataset has 538 unique contexts across 689 samples; 126 contexts
    appear 2-5 times each, and 49 of those have mixed labels.  A naive
    stratified split lets the probe see both halves of a duplicate context
    — the same passage in train and in test — which inflates the reported
    metric.  K-fold further reduces variance: with only ~138 samples per
    test fold, single-split estimates are noisy (~±5 p.p.); 5-fold averaging
    gives a far more stable AUROC.

Output for each fold:
    (idx_train, idx_val, idx_test)
    where idx_val is carved out of the train portion of the fold so that
    threshold tuning (fit_hyperparameters) has a separate inner validation
    set.  The competition metric (test AUROC) is rank-based and unaffected
    by the threshold, but the printed Accuracy/F1 numbers depend on it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)
_N_FOLDS = 5
_VAL_FRAC_OF_TRAIN = 0.15
_SEED = 42


def _extract_context(prompt: str) -> str:
    """Extract the context passage from a ChatML prompt."""
    m = _CONTEXT_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return prompt[:500]


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Group-aware K-fold split with an inner validation carve-out per fold.

    For each of K outer folds we additionally carve a validation subset out
    of the train portion (group-aware as well) so that probe threshold
    tuning has a separate inner set.

    Args:
        y:            Label array of shape ``(N,)``.  Used only for shape.
        df:           DataFrame with a ``prompt`` column for grouping.
        test_size:    Unused for K-fold (size is determined by K).  Kept for
                      API compatibility with the original signature.
        val_size:     Approximate fraction of the dataset reserved for
                      validation inside each fold.
        random_state: Seed for the inner GroupShuffleSplit.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples — one per fold.
    """
    idx_all = np.arange(len(y))

    # Fallback: no DataFrame -> single stratified split (legacy behaviour).
    if df is None or "prompt" not in df.columns:
        from sklearn.model_selection import train_test_split
        idx_trval, idx_test = train_test_split(
            idx_all, test_size=test_size, random_state=random_state, stratify=y,
        )
        relative_val = val_size / max(1.0 - test_size, 1e-6)
        idx_train, idx_val = train_test_split(
            idx_trval, test_size=relative_val,
            random_state=random_state, stratify=y[idx_trval],
        )
        return [(idx_train, idx_val, idx_test)]

    groups = df["prompt"].apply(_extract_context).values

    # Outer K-fold by group.
    gkf = GroupKFold(n_splits=_N_FOLDS)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_i, (idx_trval, idx_test) in enumerate(gkf.split(idx_all, y, groups)):
        # Inner group-aware split: carve val out of trval.
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=_VAL_FRAC_OF_TRAIN,
            random_state=random_state + fold_i,
        )
        rel_train, rel_val = next(
            gss.split(idx_trval, y[idx_trval], groups[idx_trval])
        )
        idx_train = idx_trval[rel_train]
        idx_val = idx_trval[rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits