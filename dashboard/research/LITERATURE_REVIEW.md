# Literature Review

Scope: confirmation bias, anchoring and automation bias in diagnosis; adversarial / multi-agent LLM
reasoning; pediatric machine learning; explainability; calibration; clinical-AI safety; medical
imaging; reporting guidelines.

**Verification policy.** Every entry below is machine-verified against OpenAlex/Crossref by DOI
(see `research/_scripts/verify_literature.py` and its cache in
`research/_scripts/openalex_cache.json`). Titles, authors and years in
`research/literature_review.csv` were regenerated from that verified metadata, not hand-typed.
No citation appears here that could not be resolved to a DOI or a publisher URL on 2026-10-02.
Scopus itself is an index (not an open repository) and IEEE Xplore / IAENG are publisher/literature
platforms; their items are included only where a resolvable DOI or publisher-hosted PDF exists.
**No paper is treated as a patient dataset** — datasets are tracked separately in
`research/dataset_inventory.csv`.

Full records (research question, method, sample size, results, limitations, relevance) are in
[`literature_review.csv`](literature_review.csv).

---

## 1. Clinical confirmation bias and diagnostic error

- Croskerry (2003) — canonical taxonomy of cognitive biases in diagnosis, incl. anchoring and
  confirmation bias; debiasing via metacognition. DOI:10.1097/00001888-200308000-00003
- Graber, Franklin & Gordon (2005) — 100 diagnostic error cases; cognitive factors dominate the
  preventable component (faulty synthesis, premature closure). DOI:10.1001/archinte.165.13.1493
- Singh et al. (2013) — how diagnostic-error incidence can (and cannot) be measured.
  DOI:10.1136/bmjqs-2012-001615
- Singh et al. (2016) — global burden of diagnostic error in primary care; no single "magic bullet".
  DOI:10.1136/bmjqs-2016-005401
- AHRQ (2022) — systematic review of emergency-department diagnostic errors: 279 studies,
  top-15 high-risk conditions, harm rates with 95% CIs. DOI:10.23970/ahrqepccer258

## 2. Automation and anchoring bias (human-AI decision making)

- Goddard, Roudsari & Wyatt (2012) — systematic review of automation bias in clinical decision
  support: 74 studies; mediators (trust, workload, time pressure) and mitigators (training,
  confidence display, accountability). DOI:10.1136/amiajnl-2011-000089
- Ly, Shekelle & Song (2023) — 108,019 VA ED encounters; an initial triage framing (CHF) reduced
  subsequent PE testing: large-scale evidence of anchoring. DOI:10.1001/jamainternmed.2023.2366
- Buçinca, Malaya & Gajos (2021) — N=199: cognitive forcing functions reduce overreliance on AI
  suggestions (at some efficiency cost); explanations alone do not. DOI:10.48550/arXiv.2102.09692
- Vasconcelos et al. (2023) — overreliance is partly strategic; explanations reduce it when
  engaging with them is worthwhile. DOI:10.1145/3579605

## 3. Multi-agent and debate-based LLM reasoning

- Du et al. (2023) — multi-agent debate improves factuality/reasoning and reduces hallucination.
  DOI:10.48550/arXiv.2305.14325
- Liang et al. (EMNLP 2024) — self-reflection suffers the "Degeneration-of-Thought" problem: a
  model confident in a wrong answer cannot self-correct; structured debate restores divergence.
  DOI:10.18653/v1/2024.emnlp-main.992  ← central motivation for the Opponent agent.
- Khan et al. (2024) — debate lifts non-expert LLM judges from 48%→76% and humans 60%→88%.
  DOI:10.48550/arXiv.2402.06782
- Tang et al. (ACL Findings 2024) — MedAgents: cooperative multi-agent medical QA, but with
  **uniform information across agents** (the baseline this project ablates).
  DOI:10.18653/v1/2024.findings-acl.33
- Liu et al. (IEEE MIPR 2025) — MedChat: role-specific agents + director for multimodal diagnosis.
  DOI:10.1109/MIPR67560.2025.00078

## 4. Pediatric machine learning

- Marcinkevičs et al. (2023) — associated paper of the primary dataset; multiview concept-bottleneck
  ultrasound models, 579 patients / 1,709 images, AUROC 0.80 (diagnosis).
  DOI:10.1016/j.media.2023.103042
- Marcinkevičs et al. (2021) — ML for pediatric appendicitis on 430 children; random forest
  AUPRC 0.94/0.92/0.70 (diagnosis/management/severity). DOI:10.3389/fped.2021.662183
- Samuel (2002) — derivation of the Pediatric Appendicitis Score.
  DOI:10.1053/jpsu.2002.32893
- Goldman et al. (2008) — prospective validation of PAS. DOI:10.1016/j.jpeds.2008.01.033
- Alvarado (1986) — the Alvarado score. DOI:10.1016/S0196-0644(86)80993-3
- Horwitz et al. (2004) — selective imaging strategies for pediatric appendicitis.
  DOI:10.1542/peds.113.1.24
- Ostapenko et al. (2019) — dataset descriptor for the neonatal-sepsis auxiliary dataset.
  DOI:10.1016/j.dib.2019.104788

## 5. Explainable AI

- Lundberg & Lee (2017) — SHAP. DOI:10.48550/arXiv.1705.07874
- Ghassemi, Oakden-Rayner & Beam (2021) — current XAI is a "false hope" for patient-level trust;
  rigorous validation matters more. DOI:10.1016/S2589-7500(21)00208-9
- Adadi & Berrada (2019) — XAI concepts/taxonomies survey. DOI:10.1016/j.inffus.2019.12.012
- Arun et al. (2022) — XAI in deep-learning medical image analysis. DOI:10.1016/j.media.2022.102470
- Wanyonyi et al. (2026, IAENG) — empirical interpretability-vs-accuracy trade-off for clinical
  risk prediction (IAENG source category). URL:https://www.iaeng.org/IJCS/issues_v53/issue_3/IJCS_53_3_19.pdf

## 6. Calibration and uncertainty

- Guo et al. (2017) — modern networks are miscalibrated; temperature/Platt-style scaling works.
  DOI:10.48550/arXiv.1706.04599 (PMLR: https://proceedings.mlr.press/v70/guo17a.html)
- Van Calster et al. (2019) — calibration is the "Achilles heel" of predictive analytics;
  mandatory external calibration assessment. DOI:10.1186/s12916-019-1466-7
- Niculescu-Mizil & Caruana (2005) — Platt vs isotonic calibration across classifiers.
  DOI:10.1145/1102351.1102430

## 7. Clinical AI safety

- Challen et al. (2019) — AI, bias and clinical safety. DOI:10.1136/bmjqs-2018-008370
- Topol (2019) — human-AI convergence in medicine (augmentation, not replacement).
  DOI:10.1038/s41591-018-0300-7
- Singhal et al. (2023) — Med-PaLM 2: capability and evaluation methodology for medical LLM output.
  DOI:10.1038/s41586-023-06291-2

## 8. Medical imaging

- Rajpurkar et al. (2017) — CheXNet/DenseNet-121 pneumonia detection (transfer-learning reference).
  DOI:10.48550/arXiv.1711.05225
- Marcinkevičs et al. (2023) — pediatric ultrasound concept bottlenecks (see §4).
- Arun et al. (2022) — XAI for medical imaging (see §5).

## 9. Reporting guidelines

- TRIPOD+AI (Collins et al., 2024) — DOI:10.1136/bmj-2023-078378
- TRIPOD (2015) — DOI:10.1136/bmj.g7594
- STARD-AI (Sounderajah et al., 2025) — DOI:10.1038/s41591-025-03953-8
- STARD 2015 E&E — DOI:10.1136/bmjopen-2016-012799
- DECIDE-AI (Vasey et al., 2022) — DOI:10.1038/s41591-022-01772-9
- CONSORT-AI (2020) — DOI:10.1038/s41591-020-1034-x
- SPIRIT-AI (2020) — DOI:10.1038/s41591-020-1037-7
- TRIPOD-LLM (2025) — DOI:10.1038/s41591-024-03425-5  ← governs LLM/agent study reporting.

## 10. Pediatric safety criteria (knowledge sources for the watchdog)

- IPSCC pediatric sepsis definitions (Goldstein et al., 2005). DOI:10.1097/01.pcc.0000149131.72248.e6
- Phoenix Sepsis Criteria consensus (Schlapbach et al., 2024). DOI:10.1001/jama.2024.0179
- Phoenix criteria development/validation (Sanchez-Pinto et al., 2024; 3.05M dev / 0.58M external).
  DOI:10.1001/jama.2024.0196
- AHA Kawasaki disease scientific statement (McCrindle et al., 2017). DOI:10.1161/CIR.0000000000000484

---

## Method note

1. Candidate papers were identified from domain knowledge and targeted searches.
2. Each candidate was resolved **by title** against OpenAlex (`research/_scripts/verify_literature.py`);
   wrong guessed DOIs were rejected (several initial guesses failed 404 or resolved to a different
   paper — e.g. an early guess for the Marcinkevičs MedIA paper returned 404 until corrected to
   10.1016/j.media.2023.103042).
3. Abstracts were cached to ground the "major results" column in actual paper content.
4. IEEE Xplore and IAENG items were confirmed through publisher metadata (IEEE DOIs via OpenAlex;
   IAENG via the journal issue index) — not scraped from third-party mirrors.
5. Paywalled full texts were **not** bypassed; where a fact was not extractable from the abstract,
   the CSV says "not extracted" rather than guessing.
