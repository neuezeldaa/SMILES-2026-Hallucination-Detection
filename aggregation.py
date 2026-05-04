"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configurable constants – tweak these to experiment without changing the
# whole module.
# ---------------------------------------------------------------------------
# Which layers to use (indices into the 0‑based hidden_states tensor).
# Layer 0 = token embeddings; layer -1 = last transformer layer.
LAYER_INDICES = [-4, -3, -2, -1]

# Token pooling mode: "mean" (recommended) or "last".
POOLING_MODE = "mean"
# ---------------------------------------------------------------------------


def _real_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Boolean mask of real (non-padding) positions."""
    return attention_mask.bool()


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    The default configuration uses the last 4 transformer layers and
    mean‑pools over all real tokens within each layer, then concatenates
    the results.  This captures richer multi‑layer and full‑sequence
    information compared to the baseline (last token of the final layer).

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1‑D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1‑D feature tensor of shape ``(num_selected_layers * hidden_dim,)``.
    """
    # Ensure mask is on the same device as hidden_states
    device = hidden_states.device
    mask = _real_token_mask(attention_mask.to(device))  # (seq_len,)
    n_real = mask.sum().clamp(min=1)  # scalar on device

    selected = hidden_states[LAYER_INDICES]   # (num_layers, seq_len, hidden_dim)

    if POOLING_MODE == "mean":
        # Mean over real tokens for each selected layer
        pooled = (selected * mask[None, :, None]).sum(dim=1) / n_real
    elif POOLING_MODE == "last":
        # Last real token for each selected layer
        real_pos = attention_mask.nonzero(as_tuple=False)[-1]  # last position
        pooled = selected[:, real_pos, :].to(device)
    else:
        raise ValueError(f"Unknown POOLING_MODE: {POOLING_MODE}")

    # Concatenate all selected layers into a single vector
    return pooled.flatten()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand‑crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in the notebook.  The returned
    tensor is concatenated with the output of ``aggregate``.

    Currently implemented features (3):
      - L2 norm of the final transformer layer (averaged over real tokens).
      - Cosine distance between mean embeddings of the 0‑th layer (token
        embeddings) and the final transformer layer.
      - Standard deviation of activations of the final transformer layer
        (computed over real tokens).

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1‑D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1‑D float tensor of shape ``(3,)``.
    """
    device = hidden_states.device
    mask = _real_token_mask(attention_mask.to(device))
    n_real = mask.sum().clamp(min=1)

    # Final transformer layer (index -1) and token embeddings (index 0)
    last_layer = hidden_states[-1]    # (seq_len, hidden_dim)
    first_layer = hidden_states[0]    # (seq_len, hidden_dim)

    # 1. Mean L2 norm of the last layer
    norm_last = (last_layer * mask.unsqueeze(-1)).norm(dim=-1).sum() / n_real

    # 2. Cosine distance between mean embeddings of first and last layer
    mean_first = (first_layer * mask.unsqueeze(-1)).sum(dim=0) / n_real
    mean_last  = (last_layer * mask.unsqueeze(-1)).sum(dim=0) / n_real
    cos_sim = F.cosine_similarity(mean_first, mean_last, dim=0)
    cos_dist = 1.0 - cos_sim

    # 3. Mean standard deviation of activations in the last layer
    #    (std along the hidden_dim computed per token, then averaged)
    std_last = (
        last_layer.std(dim=-1) * mask.float()
    ).sum() / n_real

    return torch.stack([norm_last, cos_dist, std_last])


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1‑D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1‑D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = num_selected_layers * hidden_dim + n_geo_features``
        (when use_geometric=True).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features