# Safety

**This is a research system. It is not a medical device, not clinically validated, and must never
be represented as a replacement for qualified pediatric clinicians or used for patient care.**

## Three distinct signal types (never conflated)

| Signal | Source | Meaning in this project |
|---|---|---|
| **MODEL PREDICTION** | LightGBM (calibrated probability) | Statistical association in a single-centre dataset; not a diagnosis, not causation |
| **RULE-BASED SAFETY ALERT** | `knowledge/*.yaml` rules with citations | Deterministic, age-aware thresholds from cited pediatric references |
| **LLM INTERPRETATION** | Agents 1–5 | Textual reasoning over supplied evidence only; can be wrong; must carry provenance for safety claims |

The dashboard, audit JSON, and report figures must visually and structurally separate these three.

## Hard rules

1. No autonomous treatment or triage decisions; output is decision *support for research*.
2. No fabricated laboratory values, vital signs, history, or imaging findings: agents must quote
   from the record; anything absent goes to `missing_critical_information` / `cannot_assess`.
3. No invented reference ranges. Every pediatric threshold originates from
   `knowledge/pediatric_thresholds.yaml` / `sepsis_rules.yaml` / `emergency_red_flags.yaml`, each
   entry carrying `source`, `publication_year`, `citation`, `applicable_age_group`, `units`, `rule`.
4. Adult thresholds are never silently substituted for pediatric ones; age-band lookup is mandatory
   and an unmatched age band must produce an "unable to apply rule" state, not a default.
5. The watchdog must not fabricate catastrophes: a high-acuity condition may only be raised when
   patient evidence provides a plausible trigger; absent evidence is recorded in
   `red_flags_absent` or `cannot_assess_due_to_missing_data`.
6. No PHI, no re-identification, no restricted data, no paywall/auth bypass.
7. SHAP wording rule (enforced in prompts): "feature X contributed positively/negatively to this
   model prediction" — never "feature X caused the diagnosis".
8. Disclaimer field (`research_only_disclaimer`) is mandatory in every Arbitrator output.

## Model probability ≠ multi-agent confidence ≠ clinical urgency

- ML probability: calibrated on validation data (Platt/isotonic), reports discrimination +
  calibration metrics.
- Multi-agent confidence: produced by the Arbitrator from evidence quality, contradiction strength,
  completeness, calibration and guidelines — it must **not** copy the LightGBM probability, and the
  transformation from model probability to final estimate is documented in the audit trail
  (docs/ARCHITECTURE.md Stage 7).
- Clinical urgency: derived from rule-based red flags (watchdog), independent of both.

## High-acuity coverage (evidence-grounded, cited)

Configured in `knowledge/sepsis_rules.yaml` and `knowledge/emergency_red_flags.yaml`, sourced from
(at minimum): Phoenix pediatric sepsis criteria (DOI:10.1001/jama.2024.0179 and
DOI:10.1001/jama.2024.0196), IPSCC 2005 definitions (DOI:10.1097/01.pcc.0000149131.72248.e6),
AHA Kawasaki statement (DOI:10.1161/CIR.0000000000000484). Rules that cannot be sourced to a
citable reference are not implemented.

## Failure handling

- Invalid agent JSON → retry with schema error feedback, then fail the stage loudly (no silent
  fallback that could mask a missing safety check).
- LLM outage → experiments pause; no partial audit files are marked complete.
- A watchdog alert is never suppressed by the Arbitrator; disagreement is recorded in
  `model_vs_agents_disagreement`.
