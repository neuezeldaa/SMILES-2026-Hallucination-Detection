"""
diagnose_layers.py — Find the best layers for hallucination probing.

Runs a simple LogisticRegression probe on EACH hidden state independently
(embedding + 24 transformer layers = 25 layers for Qwen2.5-0.5B), using
group-aware 5-fold cross-validation.  For each layer reports mean and std
of test AUROC across folds.

Pooling for this diagnostic: concatenation of last-token + mean-pool over
real tokens (2 * hidden_dim per layer).  This is enough to compare layers
on equal footing without making the feature vector huge.

Run from repo root:
    python diagnose_layers.py

Output:
    layer_diagnostics.json — full per-layer metrics
    Console table sorted by mean test AUROC
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
OUTPUT_FILE = "layer_diagnostics.json"
BATCH_SIZE = 4
N_FOLDS = 5

_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)


def _extract_context(prompt: str) -> str:
    m = _CONTEXT_RE.search(prompt)
    return m.group(1).strip() if m else prompt[:500]


def aggregate_layer(
    layer_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Last-token + mean-pool for a SINGLE layer.

    Args:
        layer_hidden:   (seq_len, hidden_dim)
        attention_mask: (seq_len,)
    Returns:
        (2 * hidden_dim,)
    """
    attention_mask = attention_mask.to(layer_hidden.device)
    real_pos = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_pos[-1].item())

    mask_f = attention_mask.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)

    last_tok = layer_hidden[last_pos]
    mean_tok = (layer_hidden * mask_f).sum(dim=0) / n_real
    return torch.cat([last_tok, mean_tok], dim=0)


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(h)) for h in df["label"]])
    groups = df["prompt"].apply(_extract_context).values
    print(f"Loaded {len(y)} samples  ({y.sum()} hallucinated / {(y == 0).sum()} truthful)")
    print(f"Unique groups: {len(set(groups))}")

    # ---------- Load model ----------
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # ---------- Extract per-layer features ----------
    # We'll store features per layer as: features_per_layer[layer_idx] = (N, 2*hidden_dim)
    n_layers = None
    layer_features: list[list[np.ndarray]] = []

    t0 = time.time()
    for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                      desc="Extracting per-layer features", unit="batch"):
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()  # (B, n_layers, seq, hid)

        if n_layers is None:
            n_layers = hidden.size(1)
            layer_features = [[] for _ in range(n_layers)]
            print(f"\nDetected n_hidden_states = {n_layers}")

        for i in range(hidden.size(0)):
            for li in range(n_layers):
                feat = aggregate_layer(hidden[i, li], attention_mask[i])
                layer_features[li].append(feat.cpu().numpy())

    extract_time = time.time() - t0
    print(f"\nExtraction done in {extract_time:.1f}s")

    # Stack each layer
    X_per_layer = [np.vstack(feats) for feats in layer_features]
    print(f"Per-layer feature shape: {X_per_layer[0].shape}")

    # ---------- Per-layer cross-validation ----------
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(len(y)), y, groups))

    results: list[dict] = []
    print(f"\nRunning {N_FOLDS}-fold group-aware CV on each of {n_layers} layers...")
    print(f"{'Layer':>6} {'Mean AUROC':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)

    for li in range(n_layers):
        X = X_per_layer[li]
        fold_aurocs = []
        for idx_train, idx_test in splits:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            )
            clf.fit(X[idx_train], y[idx_train])
            probs = clf.predict_proba(X[idx_test])[:, 1]
            fold_aurocs.append(roc_auc_score(y[idx_test], probs))

        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs))
        min_auroc = float(np.min(fold_aurocs))
        max_auroc = float(np.max(fold_aurocs))

        results.append({
            "layer": li,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "min_auroc": min_auroc,
            "max_auroc": max_auroc,
            "fold_aurocs": fold_aurocs,
        })

        print(f"{li:>6} {mean_auroc*100:>11.2f}% {std_auroc*100:>7.2f}% "
              f"{min_auroc*100:>7.2f}% {max_auroc*100:>7.2f}%")

    # ---------- Sort and report top layers ----------
    results_sorted = sorted(results, key=lambda r: r["mean_auroc"], reverse=True)
    print("\n" + "=" * 50)
    print("TOP 10 LAYERS by mean test AUROC")
    print("=" * 50)
    for r in results_sorted[:10]:
        print(f"  Layer {r['layer']:>2}: "
              f"{r['mean_auroc']*100:.2f}% ± {r['std_auroc']*100:.2f}%")

    # ---------- Save ----------
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "n_layers": n_layers,
            "n_folds": N_FOLDS,
            "extract_time_s": extract_time,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to '{OUTPUT_FILE}'")

    # ---------- Combined top-3 layers experiment ----------
    print("\n" + "=" * 50)
    print("BONUS: top-3 layers concatenated")
    print("=" * 50)
    top3_idx = [r["layer"] for r in results_sorted[:3]]
    print(f"Concatenating layers: {top3_idx}")
    X_top3 = np.hstack([X_per_layer[li] for li in top3_idx])
    print(f"Concatenated shape: {X_top3.shape}")

    fold_aurocs = []
    for idx_train, idx_test in splits:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=42,
            ),
        )
        clf.fit(X_top3[idx_train], y[idx_train])
        probs = clf.predict_proba(X_top3[idx_test])[:, 1]
        fold_aurocs.append(roc_auc_score(y[idx_test], probs))

    print(f"Top-3 combined: {np.mean(fold_aurocs)*100:.2f}% ± {np.std(fold_aurocs)*100:.2f}%")


if __name__ == "__main__":
    main()