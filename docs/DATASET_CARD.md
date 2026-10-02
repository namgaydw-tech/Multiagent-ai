# DATASET CARD: Regensburg Pediatric Appendicitis Dataset

| Field | Value |
|---|---|
| **Title** | Regensburg Pediatric Appendicitis Dataset |
| **Authors** | Ričards Marcinkevičs, Patricia Reis Wolfertstetter, Ugne Klimiene, Ece Ozkan, Kieran Chin-Cheong, Alyssia Paschke, Julia Zerres, Markus Denzinger, David Niederberger, S. Wellmann, C. Knorr, Julia E. Vogt |
| **Sources** | UCI ML Repository ID 938 — https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis · Zenodo record 7669442 — https://zenodo.org/records/7669442 |
| **DOI** | 10.5281/zenodo.7669442 (concept DOI 10.5281/zenodo.7669214) |
| **Associated publication** | Marcinkevičs et al., *Interpretable and intervenable ultrasonography-based machine learning models for pediatric appendicitis*, Medical Image Analysis, 2023. DOI:10.1016/j.media.2023.103042 (arXiv:2302.14460) |
| **License** | **CC-BY-NC-4.0** (verified from the Zenodo record API on 2026-10-02) — non-commercial research use only |
| **Patient count** | 782 rows = 782 patients (one row per patient), Children's Hospital St. Hedwig, Regensburg, Germany, 2016–2021 |
| **Age distribution** | mean 11.35 ± 3.53 y; median 11.44 y (IQR 9.20–14.10); range 0–18.36 y; bins: 0–1 y: 5, 2–5 y: 64, 6–12 y: 457, 13–18 y: 255; 1 missing |
| **Sex** | male 403, female 377, missing 2 |
| **Target variables** | `Diagnosis` (appendicitis 463 / no appendicitis 317 / missing 2), `Management` (conservative 483, primary surgical 270, secondary surgical 27, simultaneous appendectomy 1, missing 1), `Severity` (uncomplicated 662, complicated 119, missing 1) |
| **Tabular features** | 58 columns total including demographics (Age, Sex, Height, Weight, BMI), scoring (Alvarado, PAS), symptoms/examination (migratory pain, RLQ tenderness, rebound, psoas, cough pain, nausea, anorexia, peritonitis, dysuria, stool), vitals (temperature), labs (WBC, RBC, hemoglobin, RDW, thrombocytes, neutrophils, CRP, urine), ultrasound findings (appendix visibility/diameter, free fluids, wall layers, target sign, perfusion, perforation, abscess, lymph nodes, bowel wall thickening, ileus, coprostasis, meteorism, enteritis, appendicolith, gynecological findings), workflow (US_Performed, US_Number), plus `Diagnosis_Presumptive` (free-text presumptive diagnosis, German) |
| **Image modality** | B-mode abdominal ultrasound, BMP, multi-view per patient (`US_Pictures.zip`, 523,034,730 bytes, **not downloaded in Phase 1**); linked to rows via `US_Number` |
| **Class distribution** | Diagnosis: 59.2% appendicitis / 40.5% no appendicitis (neg:pos ≈ 0.685); Severity: 15.2% complicated |
| **Missingness** | 30.9% of all cells missing; every column has ≥1 missing value; ultrasound complication fields 70–98% missing (recorded only when visualised → **informative missingness**, not MCAR) |
| **Duplicates** | 0 exact duplicate rows; 0 duplicates over all non-target feature columns |
| **Potential biases** | Single-centre, German tertiary referral cohort; referral/verification bias (histology only for operated patients); `Diagnosis_Presumptive` is a clinician anchor (correct in only ~69% of the two-class values); label definition for conservative patients uses Alvarado/PAS + appendix diameter (circularity risk); age distribution skewed to school-age children (only 5 infants) |
| **Known limitations** | Small n (782) for ~53 candidate features; heavy missingness; free-text German columns; no external validation cohort available; ultrasound images not yet downloaded; no independent ethics/DPIA review performed for this project |
| **Ethical considerations** | De-identified research release; CC-BY-NC-4.0 prohibits commercial use; do not attempt re-identification; results are for research only and must not be used for patient care; no PHI is used anywhere in this repository |
| **Citation** | Marcinkevičs R, Reis Wolfertstetter P, Klimiene U, et al. (2023). Interpretable and intervenable ultrasonography-based machine learning models for pediatric appendicitis. *Medical Image Analysis*. DOI:10.1016/j.media.2023.103042. Dataset DOI:10.5281/zenodo.7669442 |

## Files as downloaded

| File | Bytes | Source | Note |
|---|---:|---|---|
| `data/raw/regensburg_pediatric_appendicitis/app_data.xlsx` | 212,771 | Zenodo `app_data.xlsx` content endpoint | Sheets: `All cases` (782×58), `Data Summary` (58×6 data dictionary) |
| `data/raw/regensburg_pediatric_appendicitis/ZENODO_README.md` | 2,235 | Zenodo `README.md` | Upstream readme incl. copyright and license statement |
| `data/manifests/uci_dataset_938_metadata.json` | — | UCI API `/api/dataset?id=938` | Machine-readable variable roles/descriptions |
| `data/manifests/regensburg_data_summary.tsv` | — | extracted from `Data Summary` sheet | Variable dictionary as flat text |

## Handling rules fixed for this project

1. `Diagnosis_Presumptive` is **never** a baseline predictor; it is the experimental anchor variable.
2. `Management` and `Severity` are co-targets, never features.
3. `Length_of_Stay` (recorded at discharge) is excluded as post-outcome.
4. `US_Number` is an identifier; any image split is by patient, never by image.
5. `Alvarado_Score`, `Paedriatic_Appendicitis_Score`, `Appendix_Diameter` are retained only with a
   circularity caveat plus a sensitivity analysis excluding them.
6. Missing values are never silently dropped; missingness indicators are explicit features.
