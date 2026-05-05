"""
aggregation.py — Token aggregation strategy and feature extraction.

Multi-layer pooling:
    Hidden states from four layers (mid, mid-late, late, final) are each
    pooled three ways — last real token, mean over real tokens, max over
    real tokens — and concatenated.  Mid-to-late layers are known to carry
    more truthfulness signal than the final layer in decoder-only causal LMs
    (Azaria & Mitchell, 2023; Burns et al., 2022).

Geometric features:
    When ``USE_GEOMETRIC = True`` in solution.py, hand-crafted statistics
    are appended: sequence length, layer-wise activation norms, inter-layer
    cosine similarity (representation drift), cross-layer norm of the last
    real token, and final-layer per-dimension std across tokens.

Output dimension (Qwen2.5-0.5B, hidden_dim=896, 25 hidden states):
    aggregate                : 4 layers × 3 pools × 896 = 10752
    extract_geometric        : 1 + 25 + 24 + 24 + 3      =    77
    total (USE_GEOMETRIC)    :                              10829
"""

from __future__ import annotations

import torch


# Layer indices are computed from the number of available hidden states so the
# code remains correct if the backbone changes.  For Qwen2.5-0.5B (24 layers
# + 1 embedding = 25 entries) this picks indices 12, 16, 21, 24.
_LAYER_FRACTIONS = (0.50, 0.65, 0.85, 1.00)


def _select_layer_indices(n_hidden_states: int) -> list[int]:
    """Pick layer indices from ``_LAYER_FRACTIONS`` of total depth."""
    indices: list[int] = []
    for f in _LAYER_FRACTIONS:
        idx = min(int(round(f * (n_hidden_states - 1))), n_hidden_states - 1)
        if idx not in indices:
            indices.append(idx)
    return indices


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Concatenates last-token / mean / max pooling over four selected layers.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)``; 1 for real tokens.

    Returns:
        1-D feature tensor of shape ``(k * 3 * hidden_dim,)``.
    """
    n_layers = hidden_states.shape[0]
    layer_indices = _select_layer_indices(n_layers)

    # Index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    mask_f = attention_mask.float().unsqueeze(-1)        # (seq_len, 1)
    n_real = mask_f.sum().clamp(min=1.0)

    pieces: list[torch.Tensor] = []
    for li in layer_indices:
        layer = hidden_states[li]                        # (seq_len, hidden_dim)

        # 1. Last real token.
        last_tok = layer[last_pos]

        # 2. Mean pool over real tokens.
        mean_tok = (layer * mask_f).sum(dim=0) / n_real

        # 3. Max pool over real tokens (padding masked to -inf).
        masked_for_max = layer.masked_fill(mask_f == 0, float("-inf"))
        max_tok = masked_for_max.max(dim=0).values

        pieces.extend([last_tok, mean_tok, max_tok])

    return torch.cat(pieces, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted statistical features from hidden states.

    Length is the strongest signal here — hallucinated answers in this
    dataset are ~2× longer than truthful ones — and is provided directly.
    Layer-wise norms and inter-layer drift follow the EigenScore / INSIDE
    line of work (Chen et al., 2024).

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)``; 1 for real tokens.

    Returns:
        A fixed-length 1-D float tensor.
    """
    real_mask = attention_mask.bool()
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    real_states = hidden_states[:, real_mask, :]         # (n_layers, n_real, hidden_dim)
    n_real = float(real_mask.sum().item())

    pieces: list[torch.Tensor] = []

    # 1. Sequence length, scaled to roughly [0, 1].
    pieces.append(torch.tensor([n_real / 512.0], dtype=torch.float32))

    # 2. Layer-wise mean activation norm.
    layer_norms = real_states.norm(dim=-1).mean(dim=-1)  # (n_layers,)
    pieces.append(layer_norms.float())

    # 3. Inter-layer cosine similarity of mean-pooled representation.
    layer_means = real_states.mean(dim=1)                # (n_layers, hidden_dim)
    cos_sims = torch.nn.functional.cosine_similarity(
        layer_means[:-1], layer_means[1:], dim=-1
    )                                                    # (n_layers - 1,)
    pieces.append(cos_sims.float())

    # 4. Norm of last-token delta between consecutive layers (drift).
    last_per_layer = hidden_states[:, last_pos, :]       # (n_layers, hidden_dim)
    last_drift = (last_per_layer[1:] - last_per_layer[:-1]).norm(dim=-1)
    pieces.append(last_drift.float())                    # (n_layers - 1,)

    # 5. Final-layer per-dimension std across tokens (last 3 layers averaged).
    final_std = real_states[-3:].std(dim=1).mean(dim=-1) # (3,)
    pieces.append(final_std.float())

    return torch.cat(pieces, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)``; 1 for real tokens.
        use_geometric:  Whether to append geometric features.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features