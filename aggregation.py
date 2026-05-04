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

--- Student notes ---
Implemented strategies:
  - LAYER_INDICES: which layers to use (last token, mean of last N, custom)
  - POOL_MODE: how to aggregate tokens ("mean", "max", "last", "response_mean")
  - Geometric features: layer-wise activation norms, inter-layer cosine drift,
    variance across layers.

To run a layer sweep (find best single layer):
    for i in range(n_layers):
        feat = hidden_states[i][mask].mean(0)  # or use last token
        ... train probe, report AUROC for layer i
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# CONFIGURATION — change these to experiment
# ------------------------------------------------------------------

# Which layers to use. Examples:
#   [-1]          — only the last layer (original baseline)
#   [-1, -6, -12] — last + middle + earlier layer concatenated
#   list(range(24)) — all layers (very high-dim, use PCA in probe)
LAYER_INDICES: list[int] = [-1]

# How to pool across token positions:
#   "mean"          — mean over all real (non-padding) tokens
#   "max"           — max over all real tokens
#   "last"          — last real token (original baseline)
#   "response_mean" — mean over the second half of real tokens (proxy for
#                     response tokens when no explicit split is available)
POOL_MODE: str = "last"

# ------------------------------------------------------------------


def _pool_layer(
    layer: torch.Tensor,         # (seq_len, hidden_dim)
    attention_mask: torch.Tensor,  # (seq_len,)
) -> torch.Tensor:
    """Apply POOL_MODE to a single layer tensor."""
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)  # (n_real,)

    if POOL_MODE == "last":
        last_pos = int(real_positions[-1].item())
        return layer[last_pos]

    if POOL_MODE == "mean":
        return layer[real_positions].mean(dim=0)

    if POOL_MODE == "max":
        return layer[real_positions].max(dim=0).values

    if POOL_MODE == "response_mean":
        # Rough proxy: take the second half of real tokens as the "response".
        # Replace with an exact split if token-level role labels are available.
        n_real = len(real_positions)
        response_positions = real_positions[n_real // 2:]
        if len(response_positions) == 0:
            response_positions = real_positions
        return layer[response_positions].mean(dim=0)

    raise ValueError(f"Unknown POOL_MODE: {POOL_MODE!r}")


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` for a single layer,
        or ``(k * hidden_dim,)`` when multiple layers are concatenated.
    """
    n_layers = hidden_states.shape[0]

    # Resolve negative indices once, so -1 always means the last layer
    # even when n_layers differs from what was assumed during config.
    resolved = [i % n_layers for i in LAYER_INDICES]

    pooled = [_pool_layer(hidden_states[i], attention_mask) for i in resolved]

    # Concatenate along the feature dimension.
    return torch.cat(pooled, dim=0)  # (k * hidden_dim,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.
        For 24 layers: 24 (norms) + 23 (cosine drifts) + 1 (seq_len) = 48 features.
        The length is constant across all samples.
    """
    n_layers = hidden_states.shape[0]
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    # --- 1. Layer-wise activation norms (shape: n_layers) ---
    # Hypothesis: hallucinating model may show abnormal norm growth patterns.
    layer_means = torch.stack(
        [hidden_states[i][real_positions].mean(dim=0) for i in range(n_layers)]
    )  # (n_layers, hidden_dim)
    norms = layer_means.norm(dim=-1)  # (n_layers,)

    # Normalise norms so they are scale-invariant across samples.
    norms = norms / (norms.max() + 1e-8)

    # --- 2. Inter-layer cosine similarity / representation drift (shape: n_layers-1) ---
    # Hypothesis: uncertain / hallucinating model drifts more between layers.
    cosine_sims = []
    for i in range(n_layers - 1):
        a = layer_means[i]
        b = layer_means[i + 1]
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze()
        cosine_sims.append(sim)
    drift = torch.stack(cosine_sims)  # (n_layers - 1,)

    # --- 3. Variance of representations across layers (scalar) ---
    # High variance → representation changes a lot across depth.
    var_across_layers = layer_means.var(dim=0).mean().unsqueeze(0)  # (1,)

    # --- 4. Sequence length (scalar, normalised) ---
    seq_len = real_positions.float().numel() / hidden_states.shape[1]
    seq_len_feat = torch.tensor([seq_len], dtype=torch.float32)

    return torch.cat([norms, drift, var_across_layers, seq_len_feat], dim=0)
    # Total: n_layers + (n_layers - 1) + 1 + 1 = 2*n_layers + 1 features


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
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = k * hidden_dim [+ 2*n_layers + 1]``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features