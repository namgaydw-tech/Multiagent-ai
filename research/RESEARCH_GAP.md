# Research Gap

## What already exists

1. **Clinical bias literature is mature.** Anchoring, confirmation bias and premature closure are
   well documented in clinicians (Croskerry 2003, DOI:10.1097/00001888-200308000-00003; Graber 2005,
   DOI:10.1001/archinte.165.13.1493), and automation bias under clinical decision support is
   systematically characterised (Goddard 2012, DOI:10.1136/amiajnl-2011-000089). Large-scale
   observational evidence shows an initial framing distorts real physician testing decisions
   (Ly 2023, DOI:10.1001/jamainternmed.2023.2366, n=108,019).

2. **Multi-agent LLM reasoning is established as a general technique** — debate improves factuality
   and reasoning (Du 2023, DOI:10.48550/arXiv.2305.14325), self-reflection demonstrably fails once a
   model is confident in a wrong answer (Liang 2024, "Degeneration-of-Thought",
   DOI:10.18653/v1/2024.emnlp-main.992), and debate lets weaker judges reach much higher accuracy
   (Khan 2024, 48%→76% LLM / 60%→88% human, DOI:10.48550/arXiv.2402.06782).

3. **Medical multi-agent frameworks exist but are cooperative and unevaluated for bias.**
   MedAgents (DOI:10.18653/v1/2024.findings-acl.33) gives every agent the same information;
   MedChat (DOI:10.1109/MIPR67560.2025.00078) assigns roles but reports reliability claims without
   bias metrics. None of these measures confirmation bias, anchoring, or premature closure.

4. **Pediatric appendicitis ML benchmarks exist** on the same cohort family used here
   (Marcinkevičs 2021 AUPRC 0.94/0.92/0.70, DOI:10.3389/fped.2021.662183; Marcinkevičs 2023
   concept-bottleneck ultrasound AUROC 0.80, DOI:10.1016/j.media.2023.103042), but they optimise
   accuracy, not debiasing behaviour.

5. **Reporting and safety frameworks are ready to use**: TRIPOD+AI (DOI:10.1136/bmj-2023-078378),
   STARD-AI (DOI:10.1038/s41591-025-03953-8), DECIDE-AI (DOI:10.1038/s41591-022-01772-9),
   TRIPOD-LLM (DOI:10.1038/s41591-024-03425-5), plus Phoenix pediatric sepsis criteria
   (DOI:10.1001/jama.2024.0179; DOI:10.1001/jama.2024.0196) as an authoritative rule source.

## The gap

No verified study in this review does all of the following simultaneously:

| Capability | MedAgents | MedChat | Debate papers | This project |
|---|---|---|---|---|
| Adversarial (proponent/opponent) roles in a clinical case | no | partial | general-domain | **yes** |
| Information isolation (anchor hidden from independent differential) | no | no | no | **yes** |
| Controlled anchor injection (correct/incorrect/high-confidence/low-confidence) | no | no | no | **yes** |
| Quantitative bias metrics (CBR, AOR, BCR, HFR, CRR, diagnostic stability) | no | no | no | **yes** |
| High-acuity watchdog grounded in cited pediatric criteria | no | no | no | **yes** |
| Separation of calibrated model probability vs multi-agent confidence vs urgency | no | partial | no | **yes** |
| Component ablation (11 conditions) with statistical comparison | no | no | partial | **yes** |
| Per-stage JSON audit trail for every case | no | no | no | **yes** |

Secondary gaps this project addresses:

- **Anchoring from classifiers is assumed but rarely measured in AI-assisted diagnosis.** The
  `Diagnosis_Presumptive` column in the primary dataset gives a rare opportunity to run *paired*
  experiments where patient evidence is identical and only the anchor differs.
- **Calibration is under-reported in pediatric ML.** Prior appendicitis work reports discrimination
  (AUROC/AUPRC) but rarely Brier/ECE or calibration curves (cf. Van Calster 2019,
  DOI:10.1186/s12916-019-1466-7).
- **LLM study reporting is unsettled** — TRIPOD-LLM (2025) only now defines what must be disclosed
  about prompts, models and stability; most multi-agent medical papers predate it.

## Refuted hypotheses to avoid (from the literature)

- "Adding explanations removes overreliance" — contradicted by Buçinca 2021 and qualified by
  Vasconcelos 2023; therefore explanations are *not* assumed to debias.
- "Self-reflection corrects a wrong initial diagnosis" — contradicted by Liang 2024's
  Degeneration-of-Thought result; hence the self-reflection condition is an ablation baseline, not
  the proposed solution.
- "More cooperating agents are automatically better" — no evidence of bias reduction; cooperative
  multi-agent systems may instead amplify a shared anchor. This is the central experimental question.
- "SHAP explains causation" — rejected by Ghassemi 2021; SHAP wording in agent prompts is
  explicitly non-causal.

## Contribution claim (narrow, falsifiable)

> Explicit adversarial diagnostic reasoning with **information isolation** reduces measured
> confirmation bias (CBR) and increases beneficial corrections (BCR) relative to a single-agent
> baseline and to a cooperative multi-agent baseline, **without** raising the harmful flip rate
> (HFR) beyond a pre-registered non-inferiority margin.

Everything else in the repository exists to test that claim honestly — including the possibility
that the claim fails.
