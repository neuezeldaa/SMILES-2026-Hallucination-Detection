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
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Improved Strategy
    # 1. Use last 6 layers (richer semantic info).
    # 2. Mean pool over the response tokens (not just the last one).
    # 3. Mean pool over the selected layers.
    # ------------------------------------------------------------------

    # 1. Берем последние 6 слоев (эмпирически лучший выбор для LLM)
    # Shape: (6, seq_len, hidden_dim)
    layers_to_use = hidden_states[-6:]

    # 2. Создаем маску для токенов ответа (игнорируем padding)
    # attention_mask: (seq_len,) -> 1 для токенов, 0 для паддинга
    # Добавляем измерения для броадкастинга: (1, seq_len, 1)
    mask = attention_mask.unsqueeze(0).unsqueeze(-1)

    # 3. Mean Pooling по токенам (dim=1)
    # Сначала зануляем паддинг
    masked_states = layers_to_use * mask

    # Считаем сумму по токенам
    sum_states = masked_states.sum(dim=1)  # (6, hidden_dim)

    # Делим на количество реальных токенов (чтобы среднее было честным)
    count_tokens = mask.sum(dim=1).clamp(min=1)  # (6, 1)
    mean_per_layer = sum_states / count_tokens  # (6, hidden_dim)

    # 4. Mean Pooling по слоям (dim=0)
    # Усредняем информацию из разных слоев в один вектор
    final_feature = mean_per_layer.mean(dim=0)  # (hidden_dim,)

    return final_feature


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
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Improved Strategy - Representation Drift
    # ------------------------------------------------------------------

    # Возьмем последние 6 слоев
    layers = hidden_states[-6:]

    # Усредним по токенам для каждого слоя, чтобы получить вектор слоя
    mask = attention_mask.unsqueeze(-1)

    # Сумма и деление на количество (mean pool)
    layer_means = (layers * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    # shape: (6, hidden_dim)

    # Посчитаем косинусное расстояние между соседними слоями
    # Если модель "уверена", слои похожи (drift low). Если "галлюцинирует", могут быть хаотичны.
    drifts = []
    for i in range(len(layer_means) - 1):
        cos_sim = torch.nn.functional.cosine_similarity(
            layer_means[i].unsqueeze(0),
            layer_means[i+1].unsqueeze(0)
        )
        drifts.append(cos_sim)

    # Вернем вектор из косинусных сходств (длина 5)
    return torch.cat(drifts).detach()


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
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    # Эта функция осталась оригинальной, она просто склеивает результаты
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features