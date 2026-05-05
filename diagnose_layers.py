"""
diagnose_final.py — Compare aggregation variants with LogReg + group K-fold.

After diagnose_layers.py identified layer 15 as the best single layer
(70.68% ± 3.22%), this script compares richer aggregation strategies on
top of that layer to pick the final configuration.  Each variant is
evaluated with the same LogReg + group-aware 5-fold CV pipeline so the
numbers are directly comparable.

Variants:
    A. layer 15, last token only                         (896-d)
    B. layer 15, last + mean pool                        (1792-d)
    C. layer 15, last + mean + max pool                  (2688-d)
    D. layers [13, 14, 15, 16], last + mean              (7168-d)  [+PCA]
    E. layer 15 (B) + geometric features (length, drift) (~1820-d)

Run from repo root:
    python diagnose_final.py

Output:
    final_diagnostics.json  — per-variant metrics
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
OUTPUT_FILE = "final_diagnostics.json"
BATCH_SIZE = 4
N_FOLDS = 5

_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)
_BEST_LAYER = 15
_NEIGHBOR_LAYERS = (13, 14, 15, 16)
_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def _extract_context(prompt: str) -> str:
    m = _CONTEXT_RE.search(prompt)
    return m.group(1).strip() if m else prompt[:500]


# ------------------------------------------------------------
# Per-sample features for each variant
# ------------------------------------------------------------
def variant_A_last(hidden_states, attention_mask):
    """Layer 15, last token only."""
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    return layer[last_pos]


def variant_B_last_mean(hidden_states, attention_mask):
    """Layer 15: last + mean pool."""
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    mask_f = am.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)
    last_tok = layer[last_pos]
    mean_tok = (layer * mask_f).sum(0) / n_real
    return torch.cat([last_tok, mean_tok], dim=0)


def variant_C_last_mean_max(hidden_states, attention_mask):
    """Layer 15: last + mean + max pool."""
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    mask_f = am.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)
    last_tok = layer[last_pos]
    mean_tok = (layer * mask_f).sum(0) / n_real
    masked_for_max = layer.masked_fill(mask_f == 0, float("-inf"))
    max_tok = masked_for_max.max(0).values
    return torch.cat([last_tok, mean_tok, max_tok], dim=0)


def variant_D_neighbors(hidden_states, attention_mask):
    """Layers 13-16: last + mean per layer."""
    am = attention_mask.to(hidden_states.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    mask_f = am.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)
    pieces = []
    for li in _NEIGHBOR_LAYERS:
        layer = hidden_states[li]
        last_tok = layer[last_pos]
        mean_tok = (layer * mask_f).sum(0) / n_real
        pieces.extend([last_tok, mean_tok])
    return torch.cat(pieces, dim=0)


def variant_E_with_geometric(hidden_states, attention_mask):
    """Variant B + geometric features."""
    base = variant_B_last_mean(hidden_states, attention_mask)
    am = attention_mask.to(hidden_states.device)
    real_mask = am.bool()
    n_real = float(real_mask.sum().item())

    # Length (scaled)
    length_feat = torch.tensor([n_real / 512.0], dtype=torch.float32,
                                device=hidden_states.device)

    # Inter-layer cosine drift (mean-pooled per layer)
    real_states = hidden_states[:, real_mask, :]
    layer_means = real_states.mean(dim=1)
    cos_sims = F.cosine_similarity(layer_means[:-1], layer_means[1:], dim=-1)

    # Last-token drift across layers
    real_pos = am.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_pos[-1].item())
    last_per_layer = hidden_states[:, last_pos, :]
    last_drift = (last_per_layer[1:] - last_per_layer[:-1]).norm(dim=-1)

    # Layer-wise norms
    layer_norms = real_states.norm(dim=-1).mean(dim=-1)

    geo = torch.cat([
        length_feat,
        cos_sims.float(),
        last_drift.float(),
        layer_norms.float(),
    ], dim=0)
    return torch.cat([base, geo], dim=0)


VARIANTS = [
    ("A_last_only",      variant_A_last,           False),
    ("B_last_mean",      variant_B_last_mean,      False),
    ("C_last_mean_max",  variant_C_last_mean_max,  False),
    ("D_neighbors",      variant_D_neighbors,      True),   # PCA on
    ("E_with_geometric", variant_E_with_geometric, False),
]


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------
def _select_best_C(X, y, seed=42):
    """3-fold inner CV by AUROC."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_C, best_score = 1.0, -1.0
    for C in _C_GRID:
        scores = []
        for idx_tr, idx_vl in skf.split(X, y):
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                      solver="lbfgs", random_state=seed)
            clf.fit(X[idx_tr], y[idx_tr])
            probs = clf.predict_proba(X[idx_vl])[:, 1]
            try:
                scores.append(roc_auc_score(y[idx_vl], probs))
            except ValueError:
                scores.append(0.5)
        m = float(np.mean(scores))
        if m > best_score:
            best_score, best_C = m, C
    return best_C


def evaluate_variant(X, y, splits, use_pca=False):
    """Group-K-fold AUROC for a given feature matrix."""
    fold_aurocs = []
    chosen_Cs = []
    for idx_train, idx_test in splits:
        Xtr_raw, Xte_raw = X[idx_train], X[idx_test]
        ytr = y[idx_train]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        if use_pca and Xtr.shape[1] > 512:
            n_comp = min(256, Xtr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)

        best_C = _select_best_C(Xtr, ytr)
        chosen_Cs.append(best_C)

        clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced",
                                  solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xte)[:, 1]
        fold_aurocs.append(roc_auc_score(y[idx_test], probs))

    return fold_aurocs, chosen_Cs


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(h)) for h in df["label"]])
    groups = df["prompt"].apply(_extract_context).values
    print(f"Loaded {len(y)} samples; unique groups: {len(set(groups))}")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Compute features for ALL variants in one extraction pass
    print("\nExtracting hidden states once for all variants...")
    features_by_variant: dict[str, list] = {name: [] for name, _, _ in VARIANTS}

    t0 = time.time()
    for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                      desc="Extracting", unit="batch"):
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_LENGTH)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()

        for i in range(hidden.size(0)):
            for name, fn, _ in VARIANTS:
                feat = fn(hidden[i], attention_mask[i])
                features_by_variant[name].append(feat.cpu().numpy())

    extract_time = time.time() - t0
    print(f"Extraction done in {extract_time:.1f}s")

    # Stack
    X_by_variant = {name: np.vstack(feats) for name, feats in features_by_variant.items()}
    print("Variant feature shapes:")
    for name, X in X_by_variant.items():
        print(f"  {name:>20}: {X.shape}")

    # Group K-fold
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(len(y)), y, groups))

    # Evaluate each variant
    print(f"\nEvaluating {len(VARIANTS)} variants with {N_FOLDS}-fold group CV...")
    print(f"{'Variant':>22} {'Mean AUROC':>13} {'Std':>8}  Picked Cs")
    print("-" * 70)

    results = {}
    for name, _, use_pca in VARIANTS:
        X = X_by_variant[name]
        fold_aurocs, chosen_Cs = evaluate_variant(X, y, splits, use_pca=use_pca)
        mean_a = float(np.mean(fold_aurocs))
        std_a = float(np.std(fold_aurocs))
        results[name] = {
            "mean_auroc": mean_a,
            "std_auroc": std_a,
            "fold_aurocs": fold_aurocs,
            "chosen_Cs": chosen_Cs,
            "feature_dim": int(X.shape[1]),
            "use_pca": use_pca,
        }
        print(f"{name:>22} {mean_a*100:>12.2f}% {std_a*100:>7.2f}%  {chosen_Cs}")

    # Sorted summary
    sorted_variants = sorted(results.items(), key=lambda kv: kv[1]["mean_auroc"], reverse=True)
    print("\n" + "=" * 60)
    print("RANKED RESULTS")
    print("=" * 60)
    for name, r in sorted_variants:
        print(f"  {name:>22}: {r['mean_auroc']*100:.2f}% ± {r['std_auroc']*100:.2f}%  "
              f"(dim={r['feature_dim']})")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "n_folds": N_FOLDS,
            "extract_time_s": extract_time,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()