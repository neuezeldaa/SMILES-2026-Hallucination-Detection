"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.

--- Student notes ---
layer_idx and pool_mode are passed as arguments so that solution.ipynb
can run a sweep over all layers without editing this file.

Usage in solution.ipynb:
    # Single run
    features = aggregation_and_feature_extraction(
        hidden_states, attention_mask,
        layer_idx=-1, pool_mode="last"
    )

    # Layer sweep (see sweep cell in solution.ipynb)
    for i in range(n_layers):
        features = aggregation_and_feature_extraction(
            hidden_states, attention_mask,
            layer_idx=i, pool_mode="last"
        )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pool_layer(
    layer: torch.Tensor,           # (seq_len, hidden_dim)
    attention_mask: torch.Tensor,  # (seq_len,)
    pool_mode: str,
) -> torch.Tensor:
    """Pool a single layer tensor across token positions."""
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    if pool_mode == "last":
        last_pos = int(real_positions[-1].item())
        return layer[last_pos]

    if pool_mode == "mean":
        return layer[real_positions].mean(dim=0)

    if pool_mode == "max":
        return layer[real_positions].max(dim=0).values

    if pool_mode == "response_mean":
        n_real = len(real_positions)
        response_positions = real_positions[n_real // 2:]
        if len(response_positions) == 0:
            response_positions = real_positions
        return layer[response_positions].mean(dim=0)

    raise ValueError(f"Unknown pool_mode: {pool_mode!r}")


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int = -1,
    pool_mode: str = "last",
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        layer_idx:      Which layer to use. Supports negative indexing.
                        Pass a list to concatenate multiple layers.
        pool_mode:      Token pooling strategy: "last", "mean", "max",
                        "response_mean".

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` for a single layer,
        or ``(k * hidden_dim,)`` when layer_idx is a list of k layers.
    """
    n_layers = hidden_states.shape[0]

    # Accept either a single int or a list of ints
    indices = layer_idx if isinstance(layer_idx, list) else [layer_idx]
    resolved = [i % n_layers for i in indices]

    pooled = [_pool_layer(hidden_states[i], attention_mask, pool_mode)
              for i in resolved]

    return torch.cat(pooled, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.

    Returns a 1-D float tensor of shape ``(2 * n_layers + 1,)``:
        - n_layers activation norms (normalised)
        - n_layers - 1 inter-layer cosine similarities (representation drift)
        - 1 variance-across-layers scalar
        - 1 normalised sequence length
    """
    n_layers = hidden_states.shape[0]
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    # Layer-wise mean representations
    layer_means = torch.stack(
        [hidden_states[i][real_positions].mean(dim=0) for i in range(n_layers)]
    )  # (n_layers, hidden_dim)

    # 1. Activation norms per layer
    norms = layer_means.norm(dim=-1)
    norms = norms / (norms.max() + 1e-8)

    # 2. Inter-layer cosine similarity (representation drift)
    cosine_sims = []
    for i in range(n_layers - 1):
        sim = F.cosine_similarity(
            layer_means[i].unsqueeze(0),
            layer_means[i + 1].unsqueeze(0)
        ).squeeze()
        cosine_sims.append(sim)
    drift = torch.stack(cosine_sims)  # (n_layers - 1,)

    # 3. Variance of representations across layers (scalar)
    var_across_layers = layer_means.var(dim=0).mean().unsqueeze(0)

    # 4. Normalised sequence length (scalar)
    seq_len_feat = torch.tensor(
        [real_positions.numel() / hidden_states.shape[1]],
        dtype=torch.float32
    )

    return torch.cat([norms, drift, var_across_layers, seq_len_feat], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int | list[int] = -1,
    pool_mode: str = "last",
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        layer_idx:      Layer index (int) or list of indices to concatenate.
        pool_mode:      Token pooling: "last", "mean", "max", "response_mean".
        use_geometric:  Whether to append geometric features.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)``.
    """
    agg_features = aggregate(hidden_states, attention_mask, layer_idx, pool_mode)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features