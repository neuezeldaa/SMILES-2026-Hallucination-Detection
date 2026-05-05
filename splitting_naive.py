"""
splitting.py — TEMPORARY ablation version (naive stratified K-fold).

This file is for ONE-TIME ablation only.  It deliberately ignores group
structure to measure the AUROC inflation caused by context leakage.
After this experiment, restore the group-aware version.

Strategy:
    Stratified K-fold over labels, completely ignoring context groups.
    Samples that share a context will leak across train/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


_N_FOLDS = 5
_VAL_FRAC_OF_TRAIN = 0.15
_SEED = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Naive stratified K-fold — IGNORES context groups (ablation only)."""
    idx_all = np.arange(len(y))

    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_SEED)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_i, (idx_trval, idx_test) in enumerate(skf.split(idx_all, y)):
        # Inner stratified split: carve val out of trval.
        rel_train, rel_val = train_test_split(
            np.arange(len(idx_trval)),
            test_size=_VAL_FRAC_OF_TRAIN,
            random_state=random_state + fold_i,
            stratify=y[idx_trval],
        )
        idx_train = idx_trval[rel_train]
        idx_val = idx_trval[rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits