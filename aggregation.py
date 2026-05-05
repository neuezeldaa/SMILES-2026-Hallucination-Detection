"""
aggregation.py — Token aggregation strategy and feature extraction.

Final strategy: concatenate three pooling variants for two layers (15 and
14) so that the probe can ensemble three logistic-regression sub-probes
at inference time:

    A : layer 15, last token only        ->  hidden_dim       (896)
    C : layer 15, last + mean + max pool ->  3 * hidden_dim   (2688)
    D2: layer 14, last + mean pool       ->  2 * hidden_dim   (1792)

Concatenated feature dim: 6 * hidden_dim = 5376 (for hidden_dim=896).
The probe knows the slice boundaries via the SLICE_INFO dict exposed
below.

Layer choice rationale:
    The per-layer LogReg + group K-fold CV diagnostic identified layers 15
    and 14 as the strongest individual layers (70.68% and 69.45% AUROC,
    respectively).  Final-layer features (layer 24) scored only 63.00% —
    representations near the output are tuned for next-token prediction
    rather than truthfulness.

Pooling choice rationale (diagnose_v4):
    Earlier we tested an ensemble that also included a 4-neighbour-layer
    aggregate (D = layers 13/14/15/16, last+mean, 7168-d).  Replacing D
    with D2 (layer 14 alone, 1792-d) gave the same or higher mean AUROC
    (73.64% vs 73.36%) at lower variance and 4x fewer features.

Geometric features are kept available behind USE_GEOMETRIC for ablation
but are NOT used in the final solution: the diagnose_final benchmark
showed they slightly hurt AUROC, most likely because length cues vary
across context groups under group-aware splits.
"""

from __future__ import annotations

import torch


# Best two layers identified by per-layer LogReg + group K-fold CV.
# (Qwen2.5-0.5B: 24 transformer layers + 1 embedding = 25 hidden states.)
_BEST_LAYER = 15
_SECOND_LAYER = 14

# Feature-vector slice boundaries, populated lazily so the probe can read
# them without hard-coding the hidden dimension.
SLICE_INFO: dict[str, slice] = {}


def _set_slice_info(hidden_dim: int) -> None:
    """Populate SLICE_INFO once the hidden dimension is known."""
    a_end = hidden_dim                       # A: 1 * h
    c_end = a_end + 3 * hidden_dim           # C: +3 * h
    d2_end = c_end + 2 * hidden_dim          # D2: +2 * h
    SLICE_INFO["A"] = slice(0, a_end)
    SLICE_INFO["C"] = slice(a_end, c_end)
    SLICE_INFO["D2"] = slice(c_end, d2_end)
    SLICE_INFO["hidden_dim"] = hidden_dim    # type: ignore[assignment]
    SLICE_INFO["total"] = d2_end             # type: ignore[assignment]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenated A + C + D2 feature vector for ensemble probing.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)``; 1 for real tokens.

    Returns:
        1-D feature tensor of shape ``(6 * hidden_dim,)``.
    """
    attention_mask = attention_mask.to(hidden_states.device)
    real_pos = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_pos[-1].item())

    mask_f = attention_mask.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)

    # Cap layer indices at the deepest available hidden state for safety.
    n_avail = hidden_states.shape[0]
    best_layer = min(_BEST_LAYER, n_avail - 1)
    second_layer = min(_SECOND_LAYER, n_avail - 1)

    pieces: list[torch.Tensor] = []

    # ---------- A: layer 15, last token only ----------
    layer_best = hidden_states[best_layer]
    pieces.append(layer_best[last_pos])

    # ---------- C: layer 15, last + mean + max ----------
    last_tok_C = layer_best[last_pos]
    mean_tok_C = (layer_best * mask_f).sum(dim=0) / n_real
    masked_for_max = layer_best.masked_fill(mask_f == 0, float("-inf"))
    max_tok_C = masked_for_max.max(dim=0).values
    pieces.extend([last_tok_C, mean_tok_C, max_tok_C])

    # ---------- D2: layer 14, last + mean ----------
    layer_second = hidden_states[second_layer]
    last_tok_D2 = layer_second[last_pos]
    mean_tok_D2 = (layer_second * mask_f).sum(dim=0) / n_real
    pieces.extend([last_tok_D2, mean_tok_D2])

    # Populate slice info on first call.
    if "A" not in SLICE_INFO:
        _set_slice_info(hidden_states.shape[-1])

    return torch.cat(pieces, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted statistics from hidden states.

    NOT used in the final solution (diagnose_final showed a slight AUROC
    drop vs the layer-15 baseline), but kept here for ablation.  Set
    ``USE_GEOMETRIC = True`` in solution.py to append them.
    """
    attention_mask = attention_mask.to(hidden_states.device)
    real_mask = attention_mask.bool()
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    real_states = hidden_states[:, real_mask, :]
    n_real = float(real_mask.sum().item())

    pieces: list[torch.Tensor] = []

    # Sequence length (scaled).
    pieces.append(
        torch.tensor([n_real / 512.0], dtype=torch.float32, device=hidden_states.device)
    )

    # Layer-wise mean activation norms.
    layer_norms = real_states.norm(dim=-1).mean(dim=-1)
    pieces.append(layer_norms.float())

    # Inter-layer cosine similarity of mean-pooled representation.
    layer_means = real_states.mean(dim=1)
    cos_sims = torch.nn.functional.cosine_similarity(
        layer_means[:-1], layer_means[1:], dim=-1
    )
    pieces.append(cos_sims.float())

    # Last-token drift between consecutive layers.
    last_per_layer = hidden_states[:, last_pos, :]
    last_drift = (last_per_layer[1:] - last_per_layer[:-1]).norm(dim=-1)
    pieces.append(last_drift.float())

    # Final-layer per-dimension std across tokens (last 3 layers).
    final_std = real_states[-3:].std(dim=1).mean(dim=-1)
    pieces.append(final_std.float())

    return torch.cat(pieces, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states; optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features