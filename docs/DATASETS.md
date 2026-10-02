# Datasets

Machine-readable inventory: [`research/dataset_inventory.csv`](../research/dataset_inventory.csv).

## Design rule

> One disease/domain = one model = one pipeline. Datasets are **never** concatenated.
> Every model emits the same normalized prediction interface, which is what the shared
> multi-agent debiasing engine consumes.

```
Regensburg appendicitis  → LightGBM appendicitis model   ─┐
Neonatal sepsis registry → sepsis-specific model         ─┼→ normalized prediction interface ─→ common multi-agent debiasing engine
Child pneumonia X-rays   → transfer-learning vision model ─┘
```

## Phase A — primary (audited in Phase 1)

**Regensburg Pediatric Appendicitis** — UCI 938 / Zenodo 10.5281/zenodo.7669442, CC-BY-NC-4.0,
n=782 patients, tabular + optional ultrasound images. Downloaded and audited; see
[`DATASET_AUDIT.md`](DATASET_AUDIT.md) and [`DATASET_CARD.md`](DATASET_CARD.md).
Verdict: **conditional PASS** for tabular model development.

## Phase B — auxiliary (downloaded, audit deferred)

**Neonatal Sepsis Registry: Time to Antibiotic Dataset** — Mendeley 10.17632/5vdz5cftz7.1,
n=1,946 sepsis evaluations (CHOP NICU, 2014–2018), tabular. Downloaded to
`data/raw/neonatal_sepsis/deid-nicu-sepsis-tta.csv` (175,549 bytes). Associated descriptor:
DOI:10.1016/j.dib.2019.104788.

- Population, disease, targets, and features all differ from appendicitis → **separate
  preprocessing, separate model, separate experiments**.
- License field is not exposed by the public Mendeley API; the license must be confirmed on the
  dataset landing page before any use beyond local inspection (tracked in docs/LIMITATIONS.md).
- Role: test whether the multi-agent anti-confirmation-bias architecture generalises beyond
  appendicitis.

## Phase C — imaging (metadata only, not downloaded)

**Child Pneumonia Dataset** — Mendeley 10.17632/3tx8xymdsv.1, pediatric chest X-rays.

**Audit flag (2026-10-02, via the public files API):** the v1 file listing contains **2,926 files**,
of which **2,644 carry an `aug_` (pre-augmented) prefix**, and **279 filename entries are exact
duplicates** — while the dataset description is commonly cited as "282 images". Consequences:

1. Random image-level splitting would leak augmented twins of the same radiograph across
   train/test → wildly optimistic accuracy. Patient-level grouping is not available from filenames.
2. The provenance of the pre-augmented files must be established before any training.
3. Label schema and license must be re-confirmed on the landing page.

Therefore Phase C is **blocked pending a dedicated image audit**, and will use transfer learning
(DenseNet121/EfficientNet/ResNet/ConvNeXt), training-set-only augmentation, and explicit
uncertainty reporting due to small size.

## Sources searched (per the data-discovery brief)

- UCI ML Repository — used (dataset 938, API metadata cached).
- Mendeley Data — used (two datasets; public API for metadata/files).
- Zenodo — used (authoritative source of the primary dataset).
- IEEE Xplore / IAENG / Scopus — these are **literature platforms, not patient-data repositories**;
  they contributed papers to `research/literature_review.csv`, not rows of training data.
- No paywalls, authentication walls, data-use agreements, or restricted clinical databases were
  bypassed; no copyrighted paper figures were scraped.
