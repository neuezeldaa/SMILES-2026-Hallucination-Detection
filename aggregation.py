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
    Strategy: Multi-layer Last Token Concatenation.
    We extract the LAST token vector from the last 4 layers and concatenate them.
    This preserves the final generation state enriched by deep layer semantics.
    """

    # 1. Берем последние 4 слоя (обычно там самая релевантная информация)
    # Shape: (4, seq_len, hidden_dim)
    layers_to_use = hidden_states[-4:]

    # 2. Находим индекс последнего реального токена (конец генерации)
    # attention_mask: (seq_len,) -> 1 для токенов, 0 для паддинга
    real_positions = attention_mask.nonzero(as_tuple=False)

    if len(real_positions) == 0:
        # Fallback (не должно произойти при корректных данных)
        last_pos = 0
    else:
        last_pos = int(real_positions[-1].item())

    # 3. Извлекаем векторы для этого токена из каждого из 4 слоев
    # layers_to_use[:, last_pos, :] -> Shape: (4, hidden_dim)
    selected_vectors = layers_to_use[:, last_pos, :]

    # 4. Склеиваем (Flatten) в один вектор
    # Shape: (4 * 896,) = (3584,)
    final_feature = selected_vectors.flatten()

    return final_feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Отключаем геометрические признаки, так как конкатенация слоев (3584 признака)
    дает достаточно информации. Дополнительные признаки могут добавить шум.
    """
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Main entry point."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features