"""
aggregation.py — A + C + D2 (+ optional GEO) with slice tracking.

Same as the final aggregation, but when ``use_geometric=True`` adds a
fourth slice ``GEO`` to SLICE_INFO so the probe can pick it up as an
extra sub-probe.

This file is for the ablation experiment "what if geometric features
are properly wired into the ensemble".  Use the standard aggregation.py
in production.
"""

from __future__ import annotations

import torch


_BEST_LAYER = 15
_SECOND_LAYER = 14

# Slice boundaries, populated lazily.
SLICE_INFO: dict[str, slice] = {}


def _set_slice_info(hidden_dim: int, geo_dim: int = 0) -> None:
    """Populate SLICE_INFO once dimensions are known."""
    a_end = hidden_dim
    c_end = a_end + 3 * hidden_dim
    d2_end = c_end + 2 * hidden_dim
    SLICE_INFO["A"] = slice(0, a_end)
    SLICE_INFO["C"] = slice(a_end, c_end)
    SLICE_INFO["D2"] = slice(c_end, d2_end)
    if geo_dim > 0:
        SLICE_INFO["GEO"] = slice(d2_end, d2_end + geo_dim)
    SLICE_INFO["hidden_dim"] = hidden_dim    # type: ignore[assignment]
    SLICE_INFO["total"] = d2_end + geo_dim   # type: ignore[assignment]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenated A + C + D2 feature vector (no GEO)."""
    attention_mask = attention_mask.to(hidden_states.device)
    real_pos = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_pos[-1].item())

    mask_f = attention_mask.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)

    n_avail = hidden_states.shape[0]
    best_layer = min(_BEST_LAYER, n_avail - 1)
    second_layer = min(_SECOND_LAYER, n_avail - 1)

    pieces: list[torch.Tensor] = []

    # A: layer 15, last token only
    layer_best = hidden_states[best_layer]
    pieces.append(layer_best[last_pos])

    # C: layer 15, last + mean + max
    last_tok_C = layer_best[last_pos]
    mean_tok_C = (layer_best * mask_f).sum(dim=0) / n_real
    masked_for_max = layer_best.masked_fill(mask_f == 0, float("-inf"))
    max_tok_C = masked_for_max.max(dim=0).values
    pieces.extend([last_tok_C, mean_tok_C, max_tok_C])

    # D2: layer 14, last + mean
    layer_second = hidden_states[second_layer]
    last_tok_D2 = layer_second[last_pos]
    mean_tok_D2 = (layer_second * mask_f).sum(dim=0) / n_real
    pieces.extend([last_tok_D2, mean_tok_D2])

    return torch.cat(pieces, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted statistics from hidden states (77 features)."""
    attention_mask = attention_mask.to(hidden_states.device)
    real_mask = attention_mask.bool()
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    real_states = hidden_states[:, real_mask, :]
    n_real = float(real_mask.sum().item())

    pieces: list[torch.Tensor] = []

    pieces.append(
        torch.tensor([n_real / 512.0], dtype=torch.float32, device=hidden_states.device)
    )

    layer_norms = real_states.norm(dim=-1).mean(dim=-1)
    pieces.append(layer_norms.float())

    layer_means = real_states.mean(dim=1)
    cos_sims = torch.nn.functional.cosine_similarity(
        layer_means[:-1], layer_means[1:], dim=-1
    )
    pieces.append(cos_sims.float())

    last_per_layer = hidden_states[:, last_pos, :]
    last_drift = (last_per_layer[1:] - last_per_layer[:-1]).norm(dim=-1)
    pieces.append(last_drift.float())

    final_std = real_states[-3:].std(dim=1).mean(dim=-1)
    pieces.append(final_std.float())

    return torch.cat(pieces, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states; optionally append geometric features.

    Populates SLICE_INFO on the first call, including the GEO slice when
    ``use_geometric=True``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)

        # Populate slice info (including GEO).
        if "A" not in SLICE_INFO:
            _set_slice_info(hidden_states.shape[-1], geo_dim=geo_features.shape[0])

        return torch.cat([agg_features, geo_features], dim=0)

    # Populate slice info (no GEO).
    if "A" not in SLICE_INFO:
        _set_slice_info(hidden_states.shape[-1], geo_dim=0)

    return agg_features