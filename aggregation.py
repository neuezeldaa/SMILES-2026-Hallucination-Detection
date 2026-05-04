"""
aggregation.py — Token aggregation strategy and feature extraction
"""
from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Strategy: Mean pooling over response tokens for the last 3 layers.
    More stable than last-token extraction for generation tasks.
    """
    # Берем последние 3 слоя
    layers_to_use = hidden_states[-3:]  # (3, seq_len, 896)

    # Маска для реальных токенов
    mask = attention_mask.unsqueeze(0).unsqueeze(-1).to(hidden_states.device)  # (1, seq_len, 1)

    # Mean pooling по токенам (dim=1)
    masked = layers_to_use * mask
    sum_states = masked.sum(dim=1)  # (3, 896)
    count = mask.sum(dim=1).clamp(min=1)  # (3, 1)
    mean_per_layer = sum_states / count  # (3, 896)

    # Склеиваем слои
    return mean_per_layer.flatten()  # (3 * 896,)

def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Отключаем шумовые признаки, сосредоточимся на основных
    return torch.zeros(0)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features