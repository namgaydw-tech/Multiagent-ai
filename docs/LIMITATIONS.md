# Limitations

## Data

- **Single centre, small n.** 782 patients (2016–2021, Regensburg) for ~53 candidate features;
  any performance estimate carries wide confidence intervals; no external validation cohort exists
  for this dataset in this project.
- **Informative missingness.** 30.9% of cells missing overall; ultrasound complication fields are
  70–98% missing because they were only recorded when visualised. Missingness is a signal about
  clinician behaviour, not only about pathology.
- **Label circularity.** `Diagnosis` for conservatively managed patients is partly *defined* by
  Alvarado/PAS ≥ 4 and appendix diameter ≥ 6 mm. Models using those columns partially predict the
  labelling rule. Sensitivity analyses excluding them are mandatory.
- **Anchor variable is noisy by nature.** `Diagnosis_Presumptive` disagrees with the final
  diagnosis in ~31% of two-class cases — that is a feature for bias experiments, but it means
  "incorrect anchor" injection must be done carefully (synthetic anchors, not just the raw column).
- **Auxiliary datasets are independent, not poolable.** Neonatal sepsis (n=1,946) differs in
  population, disease, targets and features; the pneumonia dataset has pre-augmentation leakage
  hazards (see docs/DATASETS.md). Nothing is concatenated to inflate row counts.

## Method

- **Association, not causation.** SHAP attributions explain the model's function, not disease
  mechanisms (see Ghassemi et al. 2021, DOI:10.1016/S2589-7500(21)00208-9).
- **LLM variability.** Agent outputs are stochastic; stability is measured, not assumed. Some
  experiments will require repeated runs per condition, increasing cost/time.
- **Bias metrics depend on ground truth definitions.** CBR/AOR/BCR/HFR assume the injected anchor's
  correctness is known; pre-registered anchor construction rules are required.
- **Power.** With n≈782 cases (and subsets eligible for paired anchor experiments), rare-event
  metrics such as Critical Miss Rate will have low power; CIs must always accompany point estimates.
- **No clinical validation.** Nothing here has been evaluated prospectively; DECIDE-AI-style
  evaluation is future work and explicitly out of scope for Phase 1.

## Legal / ethical

- Primary dataset is **CC-BY-NC-4.0**: non-commercial research only.
- Mendeley neonatal-sepsis license not exposed via the public API → confirm on landing page before
  reuse beyond inspection.
- The pneumonia dataset's license (stated CC BY 4.0 in the task brief) must be re-verified on the
  landing page, and its pre-augmented file structure audited, before Phase C.
- No copyrighted paper figures/images are used as training data; no paywalls or access controls are
  bypassed; no PHI is used.

## Claims discipline

- No accuracy, bias-reduction, or generalization claim may be made until produced by a run of the
  actual experiment code and stored in `outputs/`.
- No dataset, paper, file, result, metric or model exists in this repository unless it was actually
  verified or created — Phase 1 produced audits and inventories only.
