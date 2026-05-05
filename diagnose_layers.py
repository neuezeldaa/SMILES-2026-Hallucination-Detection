"""
diagnose_v3.py — Search for the most stable configuration.

Variant A (layer 15, last token, 896-d) gave the highest mean AUROC
(72.17%) but with a worrying std of 7.07%, while C/D were 1-2 p.p. lower
but noticeably more stable.  This script answers two questions:

  1. Can stronger L2 regularisation lower A's std without hurting its mean?
  2. Does a probability-level ensemble of A and C beat either alone?

Variants tested:
    A_extreme_C    layer 15 last-only, C grid extended to 1e-5..10
    A_high_reg     layer 15 last-only, C fixed at 1e-4 / 1e-3 / 1e-2 / 1e-1
    Ensemble_AC    avg(predict_proba) of A and C — diversity helps stability
    Ensemble_ACD   avg of A, C, and D
    Ensemble_BC    avg of B and C — both stable variants

Run:
    python diagnose_v3.py
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
OUTPUT_FILE = "v3_diagnostics.json"
BATCH_SIZE = 4
N_FOLDS = 5

_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)
_BEST_LAYER = 15
_NEIGHBOR_LAYERS = (13, 14, 15, 16)


def _extract_context(prompt: str) -> str:
    m = _CONTEXT_RE.search(prompt)
    return m.group(1).strip() if m else prompt[:500]


# ------------------------------------------------------------
# Per-sample feature extractors (same as in diagnose_final.py)
# ------------------------------------------------------------
def variant_A(hidden_states, attention_mask):
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    return layer[last_pos]


def variant_B(hidden_states, attention_mask):
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    mask_f = am.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)
    last_tok = layer[last_pos]
    mean_tok = (layer * mask_f).sum(0) / n_real
    return torch.cat([last_tok, mean_tok], dim=0)


def variant_C(hidden_states, attention_mask):
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


def variant_D(hidden_states, attention_mask):
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


# ------------------------------------------------------------
# Pipeline helpers
# ------------------------------------------------------------
def _select_best_C(X, y, C_grid, seed=42):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_C, best_score = C_grid[0], -1.0
    for C in C_grid:
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


def fit_predict_proba(X, y, idx_train, idx_test, C_grid, use_pca=False):
    """Standard pipeline: scale -> [PCA] -> LogReg with C-tuned. Returns probs on test."""
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

    best_C = _select_best_C(Xtr, ytr, C_grid)

    clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced",
                              solver="lbfgs", random_state=42)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], best_C


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

    # Extract features for A, B, C, D once
    print("\nExtracting hidden states for A/B/C/D...")
    feats = {"A": [], "B": [], "C": [], "D": []}
    extractors = {"A": variant_A, "B": variant_B, "C": variant_C, "D": variant_D}

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
            for name, fn in extractors.items():
                feats[name].append(fn(hidden[i], attention_mask[i]).cpu().numpy())

    extract_time = time.time() - t0
    print(f"Extraction done in {extract_time:.1f}s")

    X = {name: np.vstack(arr) for name, arr in feats.items()}
    for name, M in X.items():
        print(f"  X[{name}]: {M.shape}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(len(y)), y, groups))

    results = {}

    # ---------- A_extreme_C: extended grid ----------
    print("\n--- A_extreme_C (layer 15 last, C grid 1e-5..10) ---")
    grid_extreme = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    fold_aurocs, chosen = [], []
    for idx_tr, idx_te in splits:
        probs, c = fit_predict_proba(X["A"], y, idx_tr, idx_te, grid_extreme)
        fold_aurocs.append(roc_auc_score(y[idx_te], probs))
        chosen.append(c)
    results["A_extreme_C"] = {
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc": float(np.std(fold_aurocs)),
        "fold_aurocs": fold_aurocs,
        "chosen_Cs": chosen,
    }
    print(f"  mean={np.mean(fold_aurocs)*100:.2f}%  std={np.std(fold_aurocs)*100:.2f}%  Cs={chosen}")

    # ---------- A_high_reg: small Cs only ----------
    print("\n--- A_high_reg (layer 15 last, C grid 1e-4..1e-1) ---")
    grid_high = (1e-4, 1e-3, 1e-2, 1e-1)
    fold_aurocs, chosen = [], []
    for idx_tr, idx_te in splits:
        probs, c = fit_predict_proba(X["A"], y, idx_tr, idx_te, grid_high)
        fold_aurocs.append(roc_auc_score(y[idx_te], probs))
        chosen.append(c)
    results["A_high_reg"] = {
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc": float(np.std(fold_aurocs)),
        "fold_aurocs": fold_aurocs,
        "chosen_Cs": chosen,
    }
    print(f"  mean={np.mean(fold_aurocs)*100:.2f}%  std={np.std(fold_aurocs)*100:.2f}%  Cs={chosen}")

    # ---------- Standard grid for A/B/C/D (recompute, used for ensembles) ----------
    grid_std = (0.001, 0.01, 0.1, 1.0, 10.0)
    print("\n--- Computing per-fold probs for A/B/C/D for ensemble ---")
    per_variant_probs: dict[str, list[np.ndarray]] = {n: [] for n in "ABCD"}
    per_variant_aurocs: dict[str, list[float]] = {n: [] for n in "ABCD"}

    for fold_i, (idx_tr, idx_te) in enumerate(splits):
        for name in "ABCD":
            use_pca = (name == "D")
            probs, _ = fit_predict_proba(X[name], y, idx_tr, idx_te, grid_std, use_pca=use_pca)
            per_variant_probs[name].append(probs)
            per_variant_aurocs[name].append(roc_auc_score(y[idx_te], probs))

    # Sanity: print per-variant per-fold to confirm we reproduce v2 numbers
    print("Per-variant AUROCs per fold (sanity check):")
    for name in "ABCD":
        m = np.mean(per_variant_aurocs[name])
        s = np.std(per_variant_aurocs[name])
        print(f"  {name}: mean={m*100:.2f}% std={s*100:.2f}%  folds={[f'{a:.4f}' for a in per_variant_aurocs[name]]}")

    # ---------- Ensembles ----------
    def ensemble_aurocs(name_list):
        aurocs = []
        for fold_i, (_, idx_te) in enumerate(splits):
            stacked = np.mean(np.stack(
                [per_variant_probs[n][fold_i] for n in name_list]
            ), axis=0)
            aurocs.append(roc_auc_score(y[idx_te], stacked))
        return aurocs

    for combo in [("A", "C"), ("A", "B"), ("A", "D"),
                  ("A", "C", "D"), ("B", "C"), ("A", "B", "C", "D")]:
        name = "Ensemble_" + "".join(combo)
        fa = ensemble_aurocs(combo)
        results[name] = {
            "mean_auroc": float(np.mean(fa)),
            "std_auroc": float(np.std(fa)),
            "fold_aurocs": fa,
        }
        print(f"  {name:>20}: mean={np.mean(fa)*100:.2f}%  std={np.std(fa)*100:.2f}%")

    # ---------- Ranked summary ----------
    print("\n" + "=" * 60)
    print("RANKED RESULTS")
    print("=" * 60)
    sorted_r = sorted(results.items(), key=lambda kv: kv[1]["mean_auroc"], reverse=True)
    for name, r in sorted_r:
        print(f"  {name:>22}: {r['mean_auroc']*100:.2f}% ± {r['std_auroc']*100:.2f}%  "
              f"(lower bound {(r['mean_auroc']-r['std_auroc'])*100:.2f}%)")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()