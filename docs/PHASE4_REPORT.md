# PHASE 4 REPORT — Confirmation-Bias Experiments, Ablations, Evaluation

> **Status: EXPERIMENTALLY VALIDATED (deterministic backend, test partition).**
> Every number below was produced by executing this repository's code against
> the real Regensburg dataset — `python scripts/run_phase4.py anchors|ablations|analyze`.
> Nothing is projected, hand-entered or copied from documentation.
>
> Research prototype — **not a medical device**, not clinical evidence.

---

## 1. What was executed

| Item | Value |
|---|---|
| Partition | **test only** (117 cases, per `run_plan.eligible_cases`) |
| Repetitions | **3** per condition (`repetitions_per_condition`, → diagnostic stability DR) |
| Anchor conditions | **7** (`config/experiment.yaml` `anchor_conditions`) |
| Ablation profiles | **11** (`ablation_matrix`), each under 2 anchor settings |
| Anchor experiment rows | 2,457 multi-agent (7×117×3) + 7,371 single-agent (3 profiles×7×117×3) |
| Ablation rows | 7,722 (11 profiles × 117 × 3 × {control, incorrect_anchor}) |
| **Total case executions** | **17,550**, each with a persisted `audit.json` under `outputs/audits/experiments/` (62 MB, gitignored) |
| Seed | 20261002 · config hash `cb7198448d25` |
| Wall time | anchors 272.6 s + ablations 136.0 s + 140.7 s + analyze ≈ 44 s |
| LLM backend | **deterministic** (`LLM_BACKEND` unset — same disclosure as PHASE3_REPORT §1) |

Headline framework = `9_full_with_rag` (five agents + RAG + isolation C).
Single-agent baselines = `2_single_llm` (record only) and
`3_lightgbm_plus_single_llm` (record + model), used for
`bias_reduction = CBR_single − CBR_multi`.

## 2. Files created

| Area | Files |
|---|---|
| Anchors | `src/bias/anchor_generator.py` (7 conditions; observed + label-swapped real anchors) |
| Metrics/stats | `src/bias/metrics.py` (CBR/AOR/BCR/HFR/CRR/DDR/CMR/DR, bootstrap CI, exact McNemar, paired bootstrap, Benjamini-Hochberg) |
| Experiment engine | `src/bias/experiment.py` (11 profiles, deterministic single-agent baselines, uniform result rows) |
| Error analysis | `src/evaluation/error_analysis.py` → `outputs/metrics/error_taxonomy.csv`, `error_analysis.json` |
| Figures | `src/evaluation/publication_figures.py` → 5 PNGs in `outputs/figures/` (regenerable; gitignored like Phase 2 figures) |
| Runner | `scripts/run_phase4.py` (`anchors` / `ablations --anchor …` / `analyze`) |
| Tests | `tests/test_phase4.py` (26 tests) |
| Results | `outputs/metrics/{anchor_experiment,single_baseline,ablation_experiment,bias_metrics,final_results}.{json,csv}`, `error_taxonomy.csv` |

Supporting changes (ablation plumbing): the arbitrator/engine now support
**skipped stages recorded as absent** (`opponent/watchdog/retrieval = None` →
"not run in this condition" in the audit trail, weighting terms neutralised —
never fabricated), `CaseState.pipeline_stages` subsets, and
`run_case(stage_files=False)` (case audit without full transcripts).

## 3. Anchor conditions (executed)

`control` (none) · `correct_anchor` · `incorrect_anchor` ·
`high_confidence_incorrect_anchor` (0.95, assertive) ·
`low_confidence_incorrect_anchor` (0.55, tentative) · `model_anchor` (LightGBM top) ·
`no_model_anchor` (model revealed only after the differential).

* Observed anchors = the real `Diagnosis_Presumptive` value (76/117 test cases
  naturally agree with ground truth in family space → 41 naturally-incorrect cases).
* Synthetic anchors = **label-swapped real presumptive values** from the same
  dataset (plausible by construction; strings are never invented).
* Paired design verified: patient features byte-identical across conditions
  (test `test_anchor_generation_is_deterministic_and_non_mutating`).

## 4. Headline results — multi-agent framework (9_full_with_rag, N=117×3)

From `outputs/metrics/bias_metrics.json` (95% CI = 2000-draw bootstrap):

| Condition | CBR | AOR | BCR | HFR | CRR | CMR | acc(final) | acc(initial) | DR |
|---|---|---|---|---|---|---|---|---|---|
| control | n/a | n/a | 0.000 [0, 0] n=30 | **0.028 [0.013, 0.047]** n=321 | 0.889 [0.855, 0.920] | 0.270 [0.189, 0.351] n=111 | 0.889 | 0.914 | 1.0 |
| correct_anchor | n/a | n/a | n/a | **0.111 [0.080, 0.145]** n=351 | 0.889 | 0.270 | 0.889 | 1.000 | 1.0 |
| incorrect_anchor | **0.051 [0.029, 0.077]** n=351 | 0.949 [0.923, 0.972] | 0.914 [0.883, 0.943] | n/a | 0.914 [0.882, 0.940] | 0.297 [0.216, 0.387] | 0.914 | 0.000 | 1.0 |
| high_conf_incorrect | 0.051 (same cases) | 0.949 | 0.914 | n/a | 0.914 | 0.297 | 0.914 | 0.000 | 1.0 |
| low_conf_incorrect | 0.051 (same cases) | 0.949 | 0.914 | n/a | 0.914 | 0.297 | 0.914 | 0.000 | 1.0 |
| model_anchor | 1.000 n=30 | 0.000 | 0.000 n=30 | 0.028 n=321 | 0.889 | 0.270 | 0.889 | 0.914 | 1.0 |
| no_model_anchor | n/a | n/a | 0.000 | 0.028 | 0.889 | 0.270 | 0.889 | 0.914 | 1.0 |

DDR (differential diversity before consensus): **2.40 candidates/case** mean,
corpus-level unique hypotheses reported in the JSON.
All `n` denominators are per the config formulas (unjudged rows excluded and
counted in `n_excluded` — see `src/bias/metrics.py` docstrings).

## 5. Bias reduction vs single-agent (the primary claim)

| System | CBR on incorrect anchors | 95% CI |
|---|---|---|
| single-agent, record only (`2_single_llm`) | **0.974** | [0.940, 1.000] |
| single-agent + model (`3_lightgbm_plus_single_llm`) | **0.521** | [0.470, 0.576] |
| multi-agent framework (`9_full_with_rag`) | **0.051** | [0.029, 0.077] |

* **bias_reduction (single+model − multi) = 0.470**, paired bootstrap 95% CI
  **[0.376, 0.564]**, p = 0.0009995 (2000 draws, +1 smoothing).
* **Exact McNemar** on the 117 paired cases: b = 55 (single followed, multi
  resisted), c = 0 → **p = 5.55 × 10⁻¹⁷**.
* vs record-only single: paired diff = **0.923** [0.863, 0.966]; McNemar
  b = 108, c = 0 → p = 6.16 × 10⁻³³.
* Anchor-confidence **dose–response is visible only in the single-agent
  baseline**: high-conf 0.564 vs low-conf 0.325 (multi: 0.051 both — the
  deterministic arbitration does not consume the stated confidence; the
  paired high-vs-low comparison for multi is correctly non-significant,
  p = 1.0).

### Statistical comparisons (Benjamini-Hochberg family, α = 0.05)

| # | Comparison | Test | p | q (BH) | Rejected |
|---|---|---|---|---|---|
| 1 | CBR single+model vs multi | exact McNemar | 5.55e-17 | 1.39e-16 | **yes** |
| 2 | CBR record-only single vs multi | exact McNemar | 6.16e-33 | 3.08e-32 | **yes** |
| 3 | CBR high vs low confidence (multi) | paired bootstrap | 1.0 | 1.0 | no |
| 4 | final accuracy control vs incorrect anchor (multi) | paired bootstrap | 0.103 | 0.129 | no |
| 5 | final accuracy full vs LightGBM-only (control) | paired bootstrap | 0.103 | 0.129 | no |

## 6. Ablation matrix (11 profiles × {control, incorrect_anchor}, N=117×3)

Accuracy / HFR / BCR / CBR / CMR (n/a = denominator 0 by definition):

| Profile | control acc | control HFR | control BCR | incorrect acc | incorrect BCR | incorrect CBR | incorrect CMR |
|---|---|---|---|---|---|---|---|
| 1_lightgbm_only | 0.914 | 0.000 | 0.000 | 0.914 | 0.914 | 0.051 | n/a |
| 2_single_llm | 0.564 | 0.402 | 0.200 | 0.026 | 0.026 | 0.974 | n/a |
| 3_lightgbm_plus_single_llm | 0.889 | 0.037 | 0.100 | 0.470 | 0.470 | 0.521 | n/a |
| 4_lightgbm_self_reflection | 0.872 | 0.056 | 0.100 | 0.504 | 0.504 | 0.487 | n/a |
| 5_generic_multi_agent (no opponent) | 0.914 | 0.000 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |
| 6_proponent_opponent (no watchdog) | 0.906 | 0.009 | 0.000 | 0.914 | 0.914 | 0.051 | n/a |
| 7_full_five_agent (RAG off default) | 0.906 | 0.009 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |
| 8_full_without_rag | 0.906 | 0.009 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |
| 9_full_with_rag | 0.889 | 0.028 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |
| 10_all_see_anchor (=isolation A per config) | 0.906 | 0.009 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |
| 11_information_isolated (isolation C) | 0.906 | 0.009 | 0.000 | 0.914 | 0.914 | 0.051 | 0.297 |

Observations (all descriptive unless the test above says otherwise):

1. **The opponent is the flip mechanism**: profile 5 (no opponent) has
   HFR = 0.000 and BCR = 0.000 under control — without adversarial challenge the
   debate never changes the model's answer. Every beneficial *and* harmful flip
   in this system is attributable to stage 4.
2. **RAG slightly hurt under control** (0.889 vs 0.906 for the same pipeline
   without retrieval) — descriptive only; not part of the BH family.
3. **Isolation A ≡ isolation C here** (rows 10 vs 11 identical): the
   deterministic builders do not consume the extra visibility, so the
   difference between A and C is only measurable with a real LLM (disclosed).
4. Profiles 1 and 5 agree under `incorrect_anchor` because both ≈ model-only:
   the incorrect anchor has no privileged path to the answer.
5. Config note: label `10_all_see_anchor` maps to `isolation: A_all_see_prediction`
   exactly as written in `config/experiment.yaml` (config is authoritative).

## 7. Error analysis (`outputs/metrics/error_taxonomy.csv`, 117 rows)

Model vs system (control condition), rep 0:

| | Model (Phase 2) | System (control) |
|---|---|---|
| TN / FP | 41 / 7 | 38 / **10** |
| FN / TP | 3 / 66 | 3 / 66 |

* `improved_rate` = 0.000, `worsened_rate` = 0.026 — the debate **corrected none
  of the model's 10 errors under control** (`BCR control = 0/30` over3 reps) and
  introduced 3 false positives (7 → 10).
* Model FNs: 3, persisted by the system: 3 (0 recovered).
* Under incorrect anchors: 111/117 cases **resisted** the anchor; 6 followed →
  6 anchor-induced errors.
* Uncertainty profile of system errors: 4 confident (LOW uncertainty) vs 9
  uncertain (MODERATE/HIGH).
* `high_acuity_missed` in 10 cases (complication evidence present, final
  diagnosis dropped the surgical target) → CMR ≈ 0.27–0.30 across conditions.

## 8. Diagnostic stability (DR)

DR = **1.0 [1.0, 1.0]** (117/117 cases identical across all 3 repetitions) for
every anchor condition. Honest interpretation: the pipeline is fully
deterministic, so stability is *trivially* satisfied — a real LLM backend would
make DR a meaningful measurement (disclosed, not claimed as robustness).

## 9. Publication figures (regenerate: `python scripts/run_phase4.py analyze`)

`outputs/figures/fig_anchor_cbr_aor.png`, `fig_bias_reduction.png`,
`fig_ablation_matrix.png`, `fig_error_taxonomy.png`, `fig_anchor_outcomes.png`
— all read from the persisted metrics JSONs; missing/undefined metrics render
as explicit `n/a`, never as invented bars. (Gitignored like Phase 2 figures.)

## 10. Failure modes (measured, not hidden)

1. **Harmful flips**: control HFR 0.028 (9/321 initially-correct rows); with a
   *correct* anchor HFR rises to 0.111 (39/351) — the opponent attacks whatever
   hypothesis is "leading" without knowing whether it is right, so a correct
   anchor becomes a correct target to attack. This is the framework's dominant
   safety issue and the top candidate for the next iteration.
2. **Debate does not fix model errors without an anchor**: BCR control = 0/30,
   `improved_rate` = 0.
3. **CMR ≈ 27–30%**: red-flagged cases where the final working diagnosis still
   dropped the surgical target.
4. **Model errors persist under `model_anchor`** (CBR = 1.0, BCR = 0/30): the
   pipeline resists *external* anchors but not its own model's mistakes.
5. **RAG-on vs RAG-off** control accuracy differs (0.889 vs 0.906, descriptive).
6. **A vs C identical** under the deterministic backend (no measurable
   isolation effect without an LLM).
7. Anchor *confidence* only modulates the single-agent baseline, not the
   deterministic multi-agent pipeline.

## 11. Phase 4 gate checklist

- [x] Anchor experiments executed — 7 conditions × 117 × 3 (multi + 3 single profiles)
- [x] Ablations executed — 11 profiles × 2 anchor settings × 117 × 3
- [x] CBR / AOR / BCR / HFR / CRR generated (+ CMR, DR, DDR) with bootstrap CIs
- [x] Statistical comparisons — exact McNemar, paired bootstrap difference,
      Benjamini-Hochberg correction (2 rejections at α = 0.05)
- [x] Error analysis — `error_taxonomy.csv` (117 rows) + `error_analysis.json`
- [x] Figures — 5 publication figures, computed from persisted metrics
- [x] Per-case audit trails for all 17,550 executions
- [x] Tests: **26 new Phase 4 tests; 112/112 total pass**
      (`python -m pytest tests/ -p no:warnings` → `112 passed`)
- [ ] Live-LLM ablations (conditions A/B/D–G prompt-sensitivity) — **blocked:
      no LLM backend/key** (documented; deterministic equivalents executed)

## 12. Reproduction

```bash
python scripts/run_phase4.py anchors                       # ~273 s
python scripts/run_phase4.py ablations --anchor control    # ~136 s
python scripts/run_phase4.py ablations --anchor incorrect_anchor   # ~141 s
python scripts/run_phase4.py analyze                       # stats + figures + final_results
python -m pytest tests/ -p no:warnings                     # 112 passed
```

## 13. Known limitations

1. Deterministic backend only — single-agent baselines are *documented scoring
   rules* (anchor prior weighted by stated confidence), not a live LLM; the
   bias-reduction magnitude is a property of this documented configuration and
   will differ with a real LLM. What is validated here: the pipeline, the
   pairing, the metric/statistics machinery, and the deterministic outcomes.
2. Test partition only (117 cases) for headline metrics, per `run_plan`.
3. Bootstrap CIs treat rows as exchangeable; 3 repetitions are identical by
   determinism (CI width reflects case sampling, not run-to-run variance).
4. Confidence/phrasing manipulation affects only baselines that read the
   anchor confidence (§5).
5. Figures gitignored (regenerable); all metric JSONs/CSVs are committed.
