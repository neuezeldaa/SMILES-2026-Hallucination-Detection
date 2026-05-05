"""
splitting.py — Train / validation / test split utilities.

Group-aware split: samples that share the same context (the passage between
the system prompt and the "Note that..." marker) are kept entirely within a
single fold to prevent context leakage across train/val/test.

Why group-aware:
    The dataset contains 538 unique contexts across 689 samples; 126 contexts
    appear 2-5 times each, and 49 of those have mixed labels.  A naive
    stratified split therefore lets the probe see both halves of a duplicate
    context — the same passage in train and in test — which inflates the
    reported metric without improving real generalisation.  Grouping by
    context yields an honest, leakage-free estimate.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)


def _extract_context(prompt: str) -> str:
    """Extract the context passage from a ChatML prompt.

    Returns the substring between the user-turn opener and the "Note that..."
    instruction.  Falls back to a length-based prefix if the marker is absent
    so unfamiliar prompts still produce a stable group key.
    """
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
    """Split dataset indices into train, validation, and test subsets.

    Strategy:
        * Group-aware split when ``df`` contains a ``prompt`` column — every
          context lives in exactly one of train / val / test.
        * Stratified random split as a fallback.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays (a single tuple for the default single-split strategy).
    """
    idx = np.arange(len(y))

    # ------------------------------------------------------------------
    # Group-aware path: keep duplicated contexts within one fold.
    # ------------------------------------------------------------------
    if df is not None and "prompt" in df.columns:
        groups = df["prompt"].apply(_extract_context).values

        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        idx_trval, idx_test = next(gss_test.split(idx, y, groups))

        # Validation carved out of train+val so its size is `val_size` of total.
        relative_val = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=relative_val, random_state=random_state
        )
        rel_train, rel_val = next(
            gss_val.split(idx_trval, y[idx_trval], groups[idx_trval])
        )
        idx_train = idx_trval[rel_train]
        idx_val = idx_trval[rel_val]

        return [(idx_train, idx_val, idx_test)]

    # ------------------------------------------------------------------
    # Fallback: stratified random split.
    # ------------------------------------------------------------------
    idx_trval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trval,
        test_size=relative_val,
        random_state=random_state,
        stratify=y[idx_trval],
    )
    return [(idx_train, idx_val, idx_test)]