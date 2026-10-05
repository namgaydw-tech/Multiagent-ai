# ML Algorithms and Mathematics (Phase 5)

Every formula below corresponds to an **actually implemented** algorithm or metric in
this repository. This document is the single mathematical reference for Phase 5 — it
does not duplicate `docs/PHASE2_REPORT.md` (experiment results) or
`docs/DATASET_CARD.md` (data definitions).

**Research prototype — not a medical device; not for patient care.**

Implementation map:

| Area | Source of truth |
| --- | --- |
| Metrics (Macro F0.5, AUROC, AUPRC, MCC, Brier, ECE, CIs) | `src/evaluation/phase5_metrics.py` |
| Threshold policy (sensitivity floor, validation-only) | `src/evaluation/threshold_policy.py` |
| Model zoo (17 tabular algorithms + ensembles) | `config/model_zoo.yaml`, `src/models/model_registry.py` |
| Training / calibration / sealed-test evaluation | `src/models/phase5_pipeline.py` |
| Calibration (Platt / isotonic) | `src/calibration/phase5.py` |
| Statistics (paired bootstrap, McNemar, DeLong, BH) | `src/evaluation/phase5_metrics.py`, `src/evaluation/phase5_stats.py` |
| Scorecard / winner selection | `src/evaluation/scorecard.py` |
| SHAP attributions | `shap.TreeExplainer` via `src/models/inference.py` and `src/evaluation/phase5_figures.py` |

Notation: `y ∈ {0,1}` ground truth, `ŷ` prediction at the locked threshold,
`p` calibrated probability, `K` number of classes, `β` the F-measure exponent
(Phase 5 pre-registers `β = 0.5`).

---

## 1. Metric mathematics

Confusion-matrix counts for the positive class: `TP = |{y=1, ŷ=1}|`,
`FP = |{y=0, ŷ=1}|`, `FN = |{y=1, ŷ=0}|`, `TN = |{y=0, ŷ=0}|`.

```
Precision                P  = TP / (TP + FP)
Sensitivity / Recall     R  = TP / (TP + FN)
Specificity              S  = TN / (TN + FP)
Accuracy                 Acc = (TP + TN) / (TP + TN + FP + FN)
NPV                      NPV = TN / (TN + FN)
False Positive Rate      FPR = FP / (FP + TN)
False Negative Rate      FNR = FN / (FN + TP)

F1        = 2·P·R / (P + R)
F_β       = (1 + β²)·P·R / (β²·P + R)
F0.5      = 1.25·P·R / (0.25·P + R)                     (β = 0.5)
Macro F0.5 = (1/K) · Σ_k F0.5_k                         (unweighted mean over classes)

Balanced accuracy = (Sensitivity + Specificity) / 2

MCC = (TP·TN − FP·FN) /
      sqrt( (TP+FP)·(TP+FN)·(TN+FP)·(TN+FN) )

Brier = (1/N) · Σ_i (p_i − y_i)²
ECE   = Σ_b (n_b/N) · | mean(y)_b − mean(p)_b |         (10 equal-width bins)
```

* **Macro F0.5 is not accuracy.** Macro F0.5 averages per-class F0.5 (each class's
  precision and recall combined with `β = 0.5`, i.e. precision weighted 2× over
  recall); accuracy is the fraction of correct labels. They coincide only by accident.
* Zero-division convention: undefined precision/recall contribute `0`
  (`sklearn.metrics.fbeta_score(..., zero_division=0)` equivalence — asserted in
  `tests/test_phase5.py::test_macro_f05_matches_sklearn_binary`).
* Multiclass: per-class one-vs-rest F0.5, averaged uniformly (same test asserts the
  sklearn macro average for `K = 3`).

### Safety-constrained selection rule (threshold policy)

For each model, the operating threshold is chosen **on the validation partition only**:

```
t* = argmax_t  MacroF0.5(y_val, 1[p_val ≥ t])
     subject to   Sensitivity(y_val, 1[p_val ≥ t]) ≥ floor     (floor = 0.90)
```

If no threshold on the grid reaches the floor, the fallback maximizes **sensitivity
first** and uses Macro F0.5 only as the secondary ranking. The chosen threshold,
calibrator and model are then frozen (`locked_before_test: true`) before the sealed
test partition is opened exactly once. Test labels are never used for preprocessing,
calibration, thresholding, ensemble weighting or model selection.

---

## 2. Baseline algorithms

### Logistic Regression

```
z     = β0 + β1·x1 + … + βn·xn
P(y=1|x) = 1 / (1 + e^(−z))

Odds ratio:  OR_j = e^(β_j)
```

Coefficients are fitted by L-BFGS on standardized one-hot features (scaler fit on
TRAIN only). `OR_j > 1` increases the odds of the positive class, `OR_j < 1`
decreases them; the odds-ratio plot in the Algorithm Lab is a direct plot of
`e^(β_j)`.

### Decision Tree

```
Gini(S) = 1 − Σ_k p_k²
H(S)    = −Σ_k p_k · log2(p_k)

Information Gain:
IG = H(parent) − Σ_children (n_child / n_parent) · H(child)
```

Splits maximize IG (equivalently minimize Gini/entropy). The illustrative tree in
`decision_tree_analysis.png` is capped at `max_depth = 3` for readability; the
persisted model keeps its fitted depth.

### Random Forest

```
ŷ        = mode{ h1(x), …, hB(x) }
P(y=c|x) = (1/B) · Σ_b P_b(y=c|x)
```

B bagged trees with bootstrap samples and random feature subsets; class probabilities
are the average of per-tree probabilities (hard vote = mode). Class weights are
available from the zoo config.

### Extra Trees

Same aggregation as Random Forest, but split thresholds are drawn **uniformly at
random** instead of being optimized — typically faster, higher bias, lower variance.
The `extra_trees_vs_random_forest.png` panel compares both under identical data.

### K-Nearest Neighbors

```
d(x, z) = sqrt( Σ_i (x_i − z_i)² )          (Euclidean)
ŷ(x)    = majority vote of the k nearest TRAIN points
```

Distance features require scaling: the one-hot + standard-scaler pipeline is fitted
on TRAIN only, then applied unchanged to validation/test.

### Naive Bayes (Gaussian)

```
P(C|X) = P(X|C) · P(C) / P(X)
P(X|C) = Π_j N(x_j ; μ_{j,C}, σ²_{j,C})      (conditional independence)
```

### Linear & Quadratic Discriminant Analysis

```
LDA:  δ_k(x) = xᵀ Σ⁻¹ μ_k − ½ μ_kᵀ Σ⁻¹ μ_k + log π_k      (shared Σ)
QDA:  δ_k(x) = −½ log|Σ_k| − ½ (x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k) + log π_k
```

LDA uses `solver="lsqr"` (deterministic; no `random_state` parameter exists for it).
QDA requires per-class covariance inversion; on the Regensburg tabular dataset the
class-0 covariance is rank-deficient (`n = 176 < 212` features after encoding), so
QDA is recorded as **UNAVAILABLE** with that exact reason — never as a zero score.

---

## 3. Margin models (SVM)

```
Linear:   f(x) = wᵀx + b
          min ½‖w‖² + C · Σ_i max(0, 1 − y_i·f(x_i))

RBF:      K(x_i, x_j) = exp( −γ ‖x_i − x_j‖² )
          f(x) = Σ_i α_i·K(x_i, x) + b
```

Both kernels are supported (`svm_linear`, `svm_rbf`); `C` is swept on the
DEVELOPMENT/VALIDATION partition in `svm_kernel_c_curve.png`. Probabilities come
from the estimator's Platt scaling and are then recalibrated on validation like
every other model.

---

## 4. Boosting algorithms

### Gradient Boosting

```
F_m(x) = F_(m−1)(x) + η · h_m(x)
```

Each stage `h_m` fits the negative gradient of the log-loss (steepest-descent
functional gradient step); `η` is the learning rate. `hist_gradient_boosting`
uses the same recurrence with histogram-binned, feature-wise split candidates and
native missing-value routing.

### AdaBoost

```
F(x) = Σ_t α_t · h_t(x)

α_t = ½ · ln( (1 − ε_t) / ε_t )        ε_t = weighted error of h_t
w_i ← w_i · exp( −α_t · y_i · h_t(x_i) )
```

Exponential-loss reweighting: misclassified cases gain weight for the next round.

### XGBoost

```
Obj = Σ_i L(y_i, ŷ_i) + Σ_k Ω(f_k)
Ω(f) = γ·T + ½·λ·‖w‖²
```

Regularized objective: data-fit term plus a complexity penalty on the number of
leaves `T` and leaf weights `w` (second-order Taylor expansion of `L`).

### LightGBM

Same additive boosting recurrence `F_m = F_(m−1) + η·h_m` with leaf-wise (best-first)
growth and histogram-based split finding — the Phase 2 reference backbone.

### CatBoost

Same additive recurrence with **ordered boosting**: trees are grown on prefixes of
the data with a permutation-driven target statistic, avoiding the target leakage of
ordinary gradient boosting on identical examples. Categorical features are handled
natively (no one-hot encoding).

---

## 5. Neural baseline

### Multi-Layer Perceptron

```
z^(l) = W^(l) · a^(l−1) + b^(l)
a^(l) = φ(z^(l))                       φ = ReLU (hidden), softmax/sigmoid (output)
```

Architecture `[64, 32]`, early stopping on a training-only validation fraction,
adaptive learning rate. Input scaling is fit on TRAIN only.

---

## 6. Ensembles

### Soft voting (uniform)

```
P_ensemble(y=c|x) = Σ_m w_m · P_m(y=c|x)
Σ_m w_m = 1 ,   w_m = 1/M
```

### Validation-weighted probability ensemble

```
w_m = MacroF0.5_m(val) / Σ_j MacroF0.5_j(val)          (Σ_m w_m = 1)
P(y=c|x) = Σ_m w_m · P_m(y=c|x)
```

Weights use **validation** performance only; member probabilities are the
validation-calibrated ones. Test data never contributes a weight.

### Hard voting

```
ŷ = mode{ ŷ_1(x), …, ŷ_M(x) }
```

No probabilities exist for a pure hard vote — therefore AUROC/AUPRC/Brier are
reported as **null (n/a)** in the scorecard, never 0.

### Stacking

```
P_meta(y|x) = g( P_1(x), …, P_M(x) )
```

Meta-learner `g` (logistic regression) is fitted on **out-of-fold** member
probabilities from the training partition plus held-out validation predictions;
the meta feature column is never trained on the member's own in-sample prediction.

---

## 7. Model attribution (SHAP)

```
f(x) = φ_0 + Σ_i φ_i
```

`φ_0` is the expected model output, `φ_i` the attributed contribution of feature
`i` for this instance (sum of shapley-value contributions equals `f(x) − φ_0`).
Tree models use `shap.TreeExplainer` on the fitted LightGBM/forest; the summary and
waterfall figures come from persisted validation cases.

> **SHAP is not causal.** A SHAP value states how the *model's prediction function*
> changes with the feature; it does not state that intervening on the feature changes
> the diagnosis. All UI/API wording enforces this (`wording_rule` in
> `src/models/inference.py`).

---

## 8. Statistical comparison mathematics

All of the following run **post-hoc on the sealed test partition** and only after
selection is locked; test labels are used for inference, never for tuning.

```
Bootstrap 95% CI:    θ̂* ~ percentile over B = 2000 resamples with replacement
                     CI = [ q_0.025(θ̂*), q_0.975(θ̂* ) ]

Paired Δ Macro F0.5: Δ* = MacroF0.5(A on resample) − MacroF0.5(B on resample)
                     same resampled indices for A and B (paired design)
                     p = 2 · min( P(Δ* ≤ 0), P(Δ* ≥ 0) )

McNemar (exact):     b = |A correct, B wrong| ,  c = |A wrong, B correct|
                     p = 2 · Σ_{k ≤ min(b,c)} C(b+c, k) · ½^(b+c)   (capped at 1)

DeLong AUROC:        U-statistic over pairwise pos–neg score comparisons;
                     Var(AUC) from structural components → 95% CI = AUC ± 1.96·SE

Benjamini–Hochberg:  sort p_(1) ≤ … ≤ p_(m);  q_(i) = min_{j ≥ i} ( p_(j) · m / j )
                     reject when q ≤ α = 0.05
```

Implemented in `src/evaluation/phase5_metrics.py` (`bootstrap_ci`,
`paired_bootstrap_delta_f05`, `mcnemar`, `delong_roc`, `bh_family`) and persisted to
`outputs/phase5/statistical_comparison.{json,csv}` by
`src/evaluation/phase5_stats.py`. The vectorized paired bootstrap used for speed is
asserted **numerically identical** to the reference implementation in
`tests/test_phase5.py::test_fast_paired_bootstrap_equals_reference`.

---

## 9. Calibration mathematics

```
Raw:        p_cal = p
Platt:      p_cal = 1 / (1 + e^(−(a·logit(p) + b)))        (logistic sigmoid)
Isotonic:   p_cal = isotonic regression of y on p           (monotone step function)
```

Candidate selection compares **validation** Brier and ECE
(`src/calibration/phase5.py::select_binary`), ties broken by ECE. The chosen
per-model method is recorded in `calibration.json` (`fit_on: validation`).
Multiclass calibration uses one-vs-rest Platt scaling (`OVRSigmoidCalibrator`).

---

## 10. What is deliberately absent

* **Linear Regression as a disease classifier** — not part of the zoo; Logistic
  Regression is the linear classification model.
* Test-set fitting of any kind (preprocessing, imputation, feature selection,
  hyperparameters, calibration, thresholds, ensemble weights, early stopping).
* Any figure or table value that is not produced by executed code under `outputs/`.
* Image-model results: the CNN training / Grad-CAM pipeline is **NOT_EXECUTED** in
  this environment (PyTorch not installed) and is reported as such, never invented.

**Research prototype — not a medical device; not for patient care.**

