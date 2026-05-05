"""
diagnose_v4.py — Test low-risk improvements over the ACD ensemble baseline.

Tests four families of changes on top of the v3 ACD ensemble:

  1. Multi-seed bagging: each sub-probe averaged over N_SEEDS LogReg fits
     with different random_state values.  Reduces variance from solver
     non-determinism and inner-CV C selection.

  2. Layer 14 sub-probe: add a 4th sub-probe (D2) using layer 14
     last+mean.  Layer 14 was the 2nd-best single layer (69.45%) and is
     correlated-but-not-identical to 15, so it should diversify the
     ensemble without adding noise.

  3. Probability aggregation: arithmetic mean vs geometric mean vs
     median over sub-probe probs.

  4. Calibration: try sigmoid (Platt) and isotonic calibration on each
     sub-probe before averaging.

Run:
    python diagnose_v4.py
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
OUTPUT_FILE = "v4_diagnostics.json"
BATCH_SIZE = 4
N_FOLDS = 5

_CONTEXT_RE = re.compile(
    r"<\|im_start\|>user\r?\n(.*?)Note that your answer", re.DOTALL
)
_BEST_LAYER = 15
_SECOND_LAYER = 14
_NEIGHBOR_LAYERS = (13, 14, 15, 16)
_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
_SEEDS = (42, 7, 123, 2024, 31)


def _extract_context(prompt):
    m = _CONTEXT_RE.search(prompt)
    return m.group(1).strip() if m else prompt[:500]


# ------------------------------------------------------------
# Feature extractors
# ------------------------------------------------------------
def variant_A(hidden_states, attention_mask):
    layer = hidden_states[_BEST_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    return layer[last_pos]


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


def variant_D2(hidden_states, attention_mask):
    """Layer 14 last + mean — the 2nd-best single layer."""
    layer = hidden_states[_SECOND_LAYER]
    am = attention_mask.to(layer.device)
    last_pos = int(am.nonzero(as_tuple=False)[-1].item())
    mask_f = am.float().unsqueeze(-1)
    n_real = mask_f.sum().clamp(min=1.0)
    last_tok = layer[last_pos]
    mean_tok = (layer * mask_f).sum(0) / n_real
    return torch.cat([last_tok, mean_tok], dim=0)


# ------------------------------------------------------------
# Pipeline helpers
# ------------------------------------------------------------
def _select_best_C(X, y, seed=42):
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


def fit_sub_single_seed(X_train, y_train, X_test, use_pca, seed):
    """Fit one LogReg with given seed, return predict_proba positive on test."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    if use_pca and Xtr.shape[1] > 512:
        n_comp = min(256, Xtr.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    best_C = _select_best_C(Xtr, y_train, seed=seed)
    clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced",
                              solver="lbfgs", random_state=seed)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def fit_sub_calibrated(X_train, y_train, X_test, use_pca, method="isotonic"):
    """Fit a calibrated LogReg (sigmoid or isotonic) on the train set."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    if use_pca and Xtr.shape[1] > 512:
        n_comp = min(256, Xtr.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    best_C = _select_best_C(Xtr, y_train, seed=42)
    base = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced",
                               solver="lbfgs", random_state=42)
    cal = CalibratedClassifierCV(base, method=method, cv=3)
    cal.fit(Xtr, y_train)
    return cal.predict_proba(Xte)[:, 1]


def fit_sub_multiseed(X_train, y_train, X_test, use_pca, seeds=_SEEDS):
    """Average probs over multiple seeds."""
    probs_list = [fit_sub_single_seed(X_train, y_train, X_test, use_pca, s)
                  for s in seeds]
    return np.mean(np.stack(probs_list), axis=0)


# ------------------------------------------------------------
# Aggregation methods for ensembling sub-probe outputs
# ------------------------------------------------------------
def agg_arith(probs_list):
    return np.mean(np.stack(probs_list), axis=0)


def agg_geo(probs_list):
    """Geometric mean (in log space for stability)."""
    eps = 1e-9
    log_probs = np.log(np.clip(np.stack(probs_list), eps, 1 - eps))
    return np.exp(np.mean(log_probs, axis=0))


def agg_median(probs_list):
    return np.median(np.stack(probs_list), axis=0)


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

    # Extract features for A, C, D, D2
    print("\nExtracting hidden states for A/C/D/D2...")
    feats = {"A": [], "C": [], "D": [], "D2": []}
    extractors = {"A": variant_A, "C": variant_C, "D": variant_D, "D2": variant_D2}

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

    X = {n: np.vstack(arr) for n, arr in feats.items()}
    use_pca_for = {"A": False, "C": False, "D": True, "D2": False}
    for n, M in X.items():
        print(f"  X[{n}]: {M.shape}  pca={use_pca_for[n]}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(len(y)), y, groups))

    results = {}

    # ----------------------------------------------------------
    # Pre-compute per-fold probs for each sub-probe in 3 modes:
    # single-seed, multi-seed, isotonic-calibrated
    # ----------------------------------------------------------
    modes = ["single", "multiseed", "isotonic", "sigmoid"]
    probe_keys = ["A", "C", "D", "D2"]
    # cache[mode][probe_key] = list of test-fold probs (one ndarray per fold)
    cache: dict[str, dict[str, list[np.ndarray]]] = {
        m: {k: [] for k in probe_keys} for m in modes
    }

    print("\nComputing per-fold probs across modes...")
    for fold_i, (idx_tr, idx_te) in enumerate(splits):
        print(f"  fold {fold_i+1}/{N_FOLDS}")
        for k in probe_keys:
            X_tr, X_te = X[k][idx_tr], X[k][idx_te]
            y_tr = y[idx_tr]
            cache["single"][k].append(
                fit_sub_single_seed(X_tr, y_tr, X_te, use_pca_for[k], seed=42)
            )
            cache["multiseed"][k].append(
                fit_sub_multiseed(X_tr, y_tr, X_te, use_pca_for[k])
            )
            cache["isotonic"][k].append(
                fit_sub_calibrated(X_tr, y_tr, X_te, use_pca_for[k], method="isotonic")
            )
            cache["sigmoid"][k].append(
                fit_sub_calibrated(X_tr, y_tr, X_te, use_pca_for[k], method="sigmoid")
            )

    # ----------------------------------------------------------
    # Evaluate combinations
    # ----------------------------------------------------------
    def eval_combo(name, mode, probe_subset, agg_fn):
        fold_aurocs = []
        for fold_i, (_, idx_te) in enumerate(splits):
            sub_probs = [cache[mode][k][fold_i] for k in probe_subset]
            ens = agg_fn(sub_probs)
            fold_aurocs.append(roc_auc_score(y[idx_te], ens))
        mean_a = float(np.mean(fold_aurocs))
        std_a = float(np.std(fold_aurocs))
        results[name] = {"mean_auroc": mean_a, "std_auroc": std_a,
                          "fold_aurocs": fold_aurocs}
        print(f"  {name:>32}: {mean_a*100:.2f}% ± {std_a*100:.2f}%  "
              f"(lower {(mean_a-std_a)*100:.2f}%)")

    print("\n" + "=" * 60)
    print("EXPERIMENTS")
    print("=" * 60)

    print("\nSanity (single-seed, ACD, arith) — should reproduce v3:")
    eval_combo("v3_baseline_ACD", "single", ["A", "C", "D"], agg_arith)

    print("\nMulti-seed bagging:")
    eval_combo("multiseed_ACD",     "multiseed", ["A", "C", "D"],         agg_arith)
    eval_combo("multiseed_AC",      "multiseed", ["A", "C"],              agg_arith)
    eval_combo("multiseed_AD",      "multiseed", ["A", "D"],              agg_arith)
    eval_combo("multiseed_ACDD2",   "multiseed", ["A", "C", "D", "D2"],   agg_arith)
    eval_combo("multiseed_ACD2",    "multiseed", ["A", "C", "D2"],        agg_arith)

    print("\nLayer 14 addition (single seed):")
    eval_combo("single_ACDD2",      "single",    ["A", "C", "D", "D2"],   agg_arith)
    eval_combo("single_AD2",        "single",    ["A", "D2"],             agg_arith)

    print("\nProbability aggregation methods (multi-seed, ACD):")
    eval_combo("ms_ACD_geo",        "multiseed", ["A", "C", "D"],         agg_geo)
    eval_combo("ms_ACD_median",     "multiseed", ["A", "C", "D"],         agg_median)

    print("\nCalibration:")
    eval_combo("isotonic_ACD",      "isotonic",  ["A", "C", "D"],         agg_arith)
    eval_combo("sigmoid_ACD",       "sigmoid",   ["A", "C", "D"],         agg_arith)

    # ----------------------------------------------------------
    # Ranked summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RANKED RESULTS")
    print("=" * 60)
    sorted_r = sorted(results.items(), key=lambda kv: kv[1]["mean_auroc"], reverse=True)
    for name, r in sorted_r:
        print(f"  {name:>32}: {r['mean_auroc']*100:.2f}% ± {r['std_auroc']*100:.2f}%  "
              f"(lower {(r['mean_auroc']-r['std_auroc'])*100:.2f}%)")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()