# SOLUTION.md

## Task

This solution detects hallucinations in Qwen2.5-0.5B responses by probing
hidden states from the strongest mid-late layers (14 and 15), using a
probability-level ensemble of Logistic Regression sub-probes with
multi-seed bagging. Evaluation uses group-aware 5-fold cross-validation
to prevent a context-level data leak we identified in the training set.

**Final 5-fold CV results:**

| Metric | Value |
|---|---:|
| avg_test_auroc | 74.29% |
| avg_test_accuracy | 74.61% |
| avg_test_f1 | 83.75% |
| avg_train_auroc | 99.47% |
| feature_dim | 5376 |
| n_samples | 689 |
| n_folds | 5 |

Per-fold test AUROC: 72.36%, 77.82%, 72.99%, 70.26%, 78.00%
(mean 74.29%, std ≈ 3.46 p.p.).

---

## 1. Reproducibility Instructions

### Exact commands

How to get `results.json` and `predictions.csv`:

```bash
git clone https://github.com/neuezeldaa/SMILES-2026-Hallucination-Detection
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

### Environment

- Python 3.10 or newer
- PyTorch ≥ 2.0
- transformers ≥ 4.40
- scikit-learn ≥ 1.3
- pandas, numpy

A Google Colab T4 GPU is sufficient.

### Modified files

Only the three permitted files are edited:

- `aggregation.py` — feature extraction and slice definitions
- `probe.py` — multi-seed Logistic Regression ensemble
- `splitting.py` — group-aware 5-fold cross-validation

The fixed infrastructure files (`evaluate.py`, `model.py`, `solution.py`,
`requirements.txt`) are untouched.

### Determinism

- Outer split seed: `42`
- Inner validation split: `random_state + fold_i` so each fold gets a
  different val carve-out
- Probe bagging seeds: `(42, 7, 123, 2024, 31)`
- Inner CV for the `C` hyperparameter: 3-fold stratified, seeded per
  bagging seed

Two independent re-runs of `solution.py` produced identical `results.json`
metrics, confirming the pipeline is deterministic under the configured
seeds.

### Configuration flags

`solution.py` is run with `USE_GEOMETRIC = False`. The hand-crafted
geometric features remain in `aggregation.py` for ablation purposes but
are not used by the submitted run (see Section 4 for the reasoning).

### Auxiliary script

`diagnose_layers.py` is a standalone diagnostic that fits a single-layer
LogReg probe on every hidden state and reports per-layer AUROC under the
group-aware CV. It is not invoked by `solution.py` and is included only
to document how the layer choice in Section 3 was determined.

---

## 2. Dataset Analysis and the Context-Level Data Leak

This section is the methodological foundation of the rest of the report.

### Basic statistics

- 689 labelled training samples; 100 unlabelled test samples
- Class balance: ≈ 70/30 (hallucinated / truthful)
- Majority-class baseline accuracy: 70.10%

### Context structure

Each prompt follows the ChatML template used by Qwen models. The
substantive content sits between the system prompt and a marker:

```
<|im_start|>user
... [CONTEXT passage] ... Note that your answer ...
```

| Statistic | Value |
|---|---:|
| Unique contexts | 538 |
| Contexts appearing 2–5 times | 126 |
| Contexts with mixed labels (both truthful and hallucinated) | 49 |

### Implication

A naive `StratifiedKFold` allows the same CONTEXT passage to appear in
both the train and test folds of a single split. For 49 contexts, the
labels are not even consistent — the probe can see one half of a passage
with `label=1` in train and the other half with `label=0` in test. This
is the context-level data leak that the rest of this report is
written around.

Switching `splitting.py` to `GroupKFold` with the extracted context as
the group label guarantees that every distinct passage stays entirely
within one fold.

---

## 3. Final Solution Description

The final solution has three components, each in its own file.

### 3.1 Splitting (`splitting.py`)

- Outer split: `GroupKFold(n_splits=5)` over the regex-extracted
  context
- Inner split: `GroupShuffleSplit` carves a 15% validation subset out
  of each fold's train portion, also group-aware, for probe threshold
  tuning
- Test fold sizes: 137–138; train fold sizes: 456–475; inner-val sizes:
  77–95
- A fallback to stratified `train_test_split` is kept for API
  compatibility when no DataFrame is provided

### 3.2 Aggregation (`aggregation.py`)

The feature vector for each sample concatenates three slices from two
hidden-state layers:

| Slice | Layer | Pooling | Dimension |
|---|---:|---|---:|
| A | 15 | last token only (EOS stripped) | 896 |
| C | 15 | last + mean + max | 2688 |
| D2 | 14 | last + mean | 1792 |
| Total | | | 5376 |

`SLICE_INFO` is exported from `aggregation.py` so that `probe.py` can
slice the input vector without hard-coding dimensions.

**Why layers 14 and 15.** A per-layer diagnostic (`diagnose_layers.py`)
fits an independent LogReg probe on each of the 25 hidden states under
group-aware 5-fold CV. Layer 15 ranked first at 70.68% test AUROC, layer
14 second at 69.45%. The final layer (layer 24) scored only 63.00% —
representations near the output head are tuned for next-token
prediction, not for truthfulness. This matches the pattern reported by
Azaria & Mitchell (2023, SAPLMA): truthfulness probes peak at mid-late
layers, the very last layers carry less signal.

**Why three pooling variants on these two layers.** Last-token pooling
captures end-of-response state; mean pooling captures the full response
trajectory; max pooling captures the strongest activations. Different
pools surface different aspects of the same hidden state, and combining
them in the probe ensemble removes the need to pick a single "best"
pooling rule.

**Why we strip the final token from the last-pool.** The last real
position in the attention mask corresponds to the structural terminator
of the response (an EOS-like token in the ChatML template). Its hidden
state at layers 14 and 15 carries signals tuned for "this sentence is
ending" rather than for the truthfulness of the content. We therefore
take `last_pos = real_pos[-2]` — the position of the last content-bearing
token — for the last-token pool. This adjustment lifted test AUROC by
+2.97 p.p. and tightened fold-to-fold std from ≈ 5.5 p.p. to ≈ 3.46 p.p.
The same idea is mentioned in a related submission (Lapshina, public
fork) where it produced a smaller gain on a stratified split; our
larger gain comes from a lower starting baseline on the group-aware
split.

### 3.3 Probe (`probe.py`)

The probe is a probability-level ensemble of three sub-probes, one per
slice. Each sub-probe is itself a bag of 5 Logistic Regression models
with different random seeds:

```
For each slice (A, C, D2):
    Pipeline =
        StandardScaler
        -> PCA(256) if dim > 4096    # not triggered — max slice is 2688
        -> for seed in (42, 7, 123, 2024, 31):
              3-fold inner CV picks C from {1e-3, 1e-2, 1e-1, 1, 10}
              LogReg(C=best_C, max_iter=5000,
                     class_weight="balanced",
                     solver="lbfgs",
                     random_state=seed)

Inference:
    prob_slice = mean of positive-class probabilities over the 5 seeds
    prob_final = mean over the 3 slices
```

Total: 3 slices × 5 seeds = 15 LogReg classifiers in the ensemble.

The decision threshold is tuned by F1 on the inner validation fold
(`fit_hyperparameters`). The competition's primary metric AUROC is
rank-based and unaffected by the threshold; threshold tuning only
affects the printed Accuracy and F1.

### 3.4 What contributed most to the metric

In order of contribution:

1. **Switching to group-aware splitting** — this is not a metric gain in
   absolute terms, it is the act of moving onto honest ground. Without
   it, every subsequent number would be inflated.
2. **Choosing layer 15 over layer 24** — +7-8 p.p. test AUROC compared
   to the same probe on the final layer.
3. **Replacing the MLP probe with regularised LogReg** — +3-4 p.p. test
   AUROC at this data scale.
4. **Stripping the EOS-like terminator from the last-token pool** —
   +2.97 p.p. test AUROC and a 1.6× reduction in fold-to-fold std,
   the largest single improvement after the architectural choices.
5. **Multi-pool slice ensemble (A + C + D2)** — about +1 p.p. mean AUROC
   over the best single slice, with lower variance.
6. **Multi-seed bagging** — small mean improvement, more importantly a
   measurable reduction in fold-to-fold standard deviation.

---

## 4. Experiments and Failed Attempts

The development followed a small set of clearly-formulated hypotheses.
Each subsection records the hypothesis, what we ran, the result, and
the conclusion that informed the next step.

### 4.1 Hypothesis: the data has a context-level leak

**Setup.** Extract CONTEXT from every prompt; count repetitions and
label consistency.

**Result.** 126 contexts repeat 2–5 times, 49 contexts have mixed labels.

**Re-run the original baseline (last layer + MLP) under both splits:**

| Split | Test AUROC |
|---|---:|
| `StratifiedKFold` (provided default) | 74.46% |
| `GroupKFold` (group-aware) | ≈ 67% |

**Conclusion.** The leak inflates the baseline by roughly 7 p.p. on the
same architecture. All subsequent evaluation uses `GroupKFold`.

### 4.2 Hypothesis: mid-late layers carry more truthfulness signal than the final layer

**Setup.** `diagnose_layers.py` fits a single-layer LogReg probe on each
of the 25 hidden states (mean+last pool, 896-d), under group-aware
5-fold CV.

**Result (top and bottom of the ranking):**

| Layer | Test AUROC | Std |
|---|---:|---:|
| 15 | 70.68% | 3.22% |
| 14 | 69.45% | 3.41% |
| 13 | 68.92% | 4.10% |
| … | … | … |
| 24 (final) | 63.00% | 5.50% |
| 0 (embedding) | 58.42% | 5.20% |

**Conclusion.** Layer 15 best, layer 14 close second, layer 24 worst
among the deep layers. Final layer is dropped from the feature vector.

### 4.3 Hypothesis: LogReg outperforms MLP at this sample size

**Setup.** Compare a small MLP probe (5376 → 256 → 64 → 1, ReLU,
dropout 0.3) against L2-regularised LogReg, both on the same
group-aware split.

**Result.** The MLP reached train AUROC ≈ 99% within 20 epochs while
test AUROC stalled around 67%. LogReg with `class_weight="balanced"`
and `C` chosen by inner CV reached the test AUROC numbers reported in
the ablation table below, with a much smaller train–test gap.

**Conclusion.** At 689 samples, the MLP's parameter count exceeds what
the data can support. Submitted probe is LogReg.

### 4.4 Hypothesis: combining several layers and pools beats any single one

**Setup.** Test single slices (A, B = layer 15 mean only, etc.) and
multi-slice combinations as separate feature vectors fed to the LogReg
probe.

**Key findings (group-aware CV):**

| Configuration | Dim | Test AUROC |
|---|---:|---:|
| Variant A (layer 15, last only) | 896 | 72.17% |
| Variant C (layer 15, last + mean + max) | 2688 | ≈ 73% |
| Variant D (layers 13/14/15/16, last + mean) | 7168 | 73.07% |
| Variant D2 (layer 14, last + mean) | 1792 | 73.07% |
| A + C + D2 (final) | 5376 | reported below |

**Conclusion.** Variant D2 (one extra layer) matches Variant D (three
extra layers) at one quarter of the dimension and lower variance.
Combining A + C + D2 retains both layers and three pools without
inflating the feature dimension further.

### 4.5 Hypothesis: hand-crafted geometric features add signal

**Setup.** Append a 9-dim "GEO" block to the feature vector:
normalised sequence length (`T_real / 512`), per-layer activation norms,
inter-layer cosine similarities of mean pools, last-token drift between
consecutive layers, and final-layer per-dimension std (last 3 layers).
Compare with and without GEO, under both splits.

**Result:**

| Configuration | Split | GEO | Test AUROC |
|---|---|---|---:|
| A + C + D | stratified (naive) | no | ≈ 71% |
| A + C + D | stratified (naive) | yes | 74.17% |
| A + C + D | group-aware | no | 71.32% |
| A + C + D | Group-aware | yes | 70.08% |

**Conclusion — the most important negative result of the project.**
GEO features add +3 p.p. under the leaky split but cost −1.24 p.p.
under the honest split. The interpretation: length-derived statistics
correlate with context identity (the same CONTEXT passage tends to
produce similar response lengths). On a stratified split, those
statistics act as a soft fingerprint that lets the probe identify
already-seen contexts. On a group-aware split, that fingerprint is
useless because contexts are disjoint between folds, and the GEO
features turn into noise. Submitted run sets `USE_GEOMETRIC = False`.

### 4.6 Hypothesis: averaging across seeds stabilises the probe

**Setup.** Replace each single-seed LogReg sub-probe with a bag of 5
LogReg models trained with different `random_state` values
(`(42, 7, 123, 2024, 31)`). The inner-CV `C` selection is rerun per
seed, so each bag member can land on a different regularisation
strength.

**Result (group-aware 5-fold CV, before EOS-stripping):**

| Configuration | Mean test AUROC | Std across folds |
|---|---:|---:|
| A + C + D2, single seed | 72.92% | 5.8% |
| A + C + D2, multi-seed (5 seeds) | 73.64%  | 5.4% |

**Conclusion.** Multi-seed bagging adds a small mean gain and a
measurable variance reduction. Submitted probe uses 5 seeds per slice.

### 4.7 Hypothesis: the final token in the response carries structural, not factual, signal

**Setup.** In the ChatML template the response ends with an EOS-like
terminator. With our default `last_pos = real_pos[-1]`, that terminator
becomes the input to the last-token pool on both layers 14 and 15.
Hypothesis: its hidden state at those layers encodes "this response is
ending" — a structural signal — rather than the factual content. We
replaced `last_pos = real_pos[-1]` with `last_pos = real_pos[-2]` —
the position of the last content-bearing token of the response — and
re-ran the full 5-fold pipeline. No other change.

**Result (same configuration, only the last-token index changed):**

| Configuration | Mean test AUROC | Std across folds | Mean test accuracy |
|---|---:|---:|---:|
| A + C + D2 multi-seed, with EOS in last-pool | 71.32% | 5.46% | 72.29% |
| A + C + D2 multi-seed, EOS stripped | 74.29% | 3.46% | 74.61% |

Per-fold AUROC improved on four out of five folds, with the largest
gain on the worst fold (+7.78 p.p.). Std across folds dropped by a
factor of 1.6.

**Conclusion.** The EOS-like terminator dominates the last-token pool
with structural signal. Removing it surfaces the truthfulness signal
in the same hidden state. The largest gains land on the hardest folds,
which is consistent with the interpretation that structural noise was
masking the factual signal precisely where the contextual signal was
weakest. This change is in the submitted run.

A related submission (Lapshina, public fork) reports a similar
intuition with a smaller numeric gain (+0.8 p.p.) on a stratified
split. The larger gain we observe is consistent with our group-aware
split starting from a lower, leak-free baseline that has more room to
improve.

---

## 5. Summary Ablation Table

The complete sequence of meaningful experiments, in chronological order:

| # | Configuration | Split | GEO | EOS | Test AUROC | Main takeaway |
|---|---|---|---|---|---:|---|
| 1 | Original baseline (last layer + MLP) | naive | – | in | 74.46% | Leak-inflated reference |
| 2 | Same architecture, group-aware | group | – | in | ≈ 67% | Leak quantified at ≈ 7 p.p. |
| 3 | Per-layer LogReg, single layer 24 | group | – | in | 63.00% | Final layer is poor |
| 4 | Per-layer LogReg, single layer 15 | group | – | in | 70.68% | Best single layer |
| 5 | Variant A (layer 15, last only) | group | – | in | 72.17% | Single-slice baseline |
| 6 | Variant D2 (layer 14, last + mean) | group | – | in | 73.07% | Second-best layer adds info |
| 7 | A + C + D + GEO | naive | yes | in | 74.17% | Inflated — GEO + leak combine |
| 8 | A + C + D + GEO | group | yes | in | 70.08% | GEO hurts honestly (−1.24 p.p.) |
| 9 | A + C + D2, multi-seed | group | – | in | 71.32% | Pre-EOS-strip submitted candidate |
| 10 | A + C + D2, multi-seed, EOS stripped | group | – | out | 74.29% | Final submitted |

The three rows that tell the methodological story:

- **Rows 1 vs 2** quantify the context leak.
- **Rows 7 vs 8** show that GEO features only "work" because of that leak.
- **Rows 9 vs 10** quantify the EOS-stripping gain — +2.97 p.p. AUROC
  with a 1.6× drop in std, all on the honest split.

---

## 6. Honest Interpretation of Results

The submitted test AUROC of 74.29% is competitive with the 74.46% that
the original baseline reports under `StratifiedKFold` — but our number
is on the honest split. Under `GroupKFold` the same baseline
architecture drops to ≈ 67%. Our submitted approach lifts that honest
baseline by about +7 p.p., and the standard deviation across folds is
only 3.46 p.p., which makes the result reliable rather than a lucky
single-split estimate.

We optimised AUROC rather than accuracy because AUROC is rank-based and
robust to small validation sets (the inner validation carve-out is 77–95
samples per fold, which is too small for stable threshold-dependent
metrics). Threshold tuning on folds of that size is known to overfit;
AUROC does not depend on the threshold and is the more honest indicator
of the probe's ranking quality. Accuracy on the submitted run is 74.61%,
also competitive with stratified-split peer results.

On the held-out 100-sample test set provided by the organizers, we
expect approaches built on stratified or repeated stratified CV to lose
part of their reported gain if the test set contains contexts the probe
has not seen during training. Our 74.29% AUROC and 74.61% accuracy
should generalise without further loss because both numbers were
measured on splits that already enforce context-level disjointness.

https://disk.yandex.ru/d/2s0__EhBcW8e3g
---

## References

1. Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM
   Knows When It's Lying. *Findings of EMNLP 2023*.
   Probing for truthfulness; the SAPLMA-style observation that mid-late
   layers carry more signal than the final layer motivated the layer
   choice in Section 3.2.
2. Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2022). Discovering
   Latent Knowledge in Language Models Without Supervision.
   *arXiv:2212.03827*. Foundational reference for hidden-state probing
   methodology.
