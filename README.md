# Mitigating Confirmation Bias in Automated Pediatric Diagnostics via Adversarial Multi-Agent AI Frameworks

> **Research prototype. Not a medical device. Not clinically validated. Not a replacement for
> qualified pediatric clinicians. No output of this system may be used for patient care.**

## What this project actually is

An academic clinical-AI research platform whose **primary objective is not accuracy**. It
experimentally tests whether an adversarial, information-isolated multi-agent reasoning framework
reduces:

1. confirmation bias,
2. diagnostic anchoring,
3. premature diagnostic closure,
4. inappropriate adherence to an incorrect preliminary diagnosis,
5. failure to respond to contradictory clinical evidence,

— compared with a conventional single-model diagnostic pipeline and with conventional (cooperative)
multi-agent pipelines.

Core hypothesis (falsifiable): *explicit adversarial diagnostic reasoning with information
isolation reduces confirmation bias (CBR) and increases beneficial corrections (BCR) without
raising the harmful flip rate (HFR) beyond a pre-registered margin.* See
[research/RESEARCH_GAP.md](research/RESEARCH_GAP.md).

## Status: Phase 1 complete — no model trained yet

Phase 1 (per the project's execution plan) delivered:

- Repository structure per the specification
- Dataset discovery + verification (UCI, Zenodo, Mendeley; licenses recorded)
- **Primary dataset downloaded and audited** → [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md)
  (conditional PASS; training gated on the audit conditions)
- Dataset inventory → [research/dataset_inventory.csv](research/dataset_inventory.csv)
- Literature review: **46 entries, every one DOI- or publisher-URL-verified**
  → [research/literature_review.csv](research/literature_review.csv),
  [research/LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md),
  [research/RESEARCH_GAP.md](research/RESEARCH_GAP.md)
- Citable knowledge base for pediatric safety rules
  → [knowledge/](knowledge/) (fail-closed verification status per rule)
- Pre-registered configs: [config/model.yaml](config/model.yaml),
  [config/agents.yaml](config/agents.yaml), [config/experiment.yaml](config/experiment.yaml)
- Docs: [architecture](docs/ARCHITECTURE.md) · [datasets](docs/DATASETS.md) ·
  [model card](docs/MODEL_CARD.md) · [safety](docs/SAFETY.md) ·
  [limitations](docs/LIMITATIONS.md) · [reproducibility](docs/REPRODUCIBILITY.md) ·
  [reporting checklist](docs/REPORTING_CHECKLIST.md) (TRIPOD+AI / STARD-AI / DECIDE-AI / TRIPOD-LLM)

**Nothing in this repository claims a result that was not produced by actually running code.**

## Quick start (Phase 1 reproduction)

```bash
pip install -r requirements.txt

# dataset audit -> outputs/metrics/regensburg_audit.json + docs/DATASET_AUDIT.md
python -m src.data.audit_regensburg

# citation verification against OpenAlex (network)
python research/_scripts/verify_literature.py

# tests: audit integrity, inventory, citation completeness, YAML validity
python -m pytest tests/ -q
```

Raw data lives under `data/raw/` (git-ignored) and is re-downloadable from the URLs + checksums in
`data/manifests/download_manifest.json`.

## Architecture (target)

```
Disease-specific model (LightGBM / sepsis model / vision model)
        ↓  normalized prediction interface (JSON schema)
Common multi-agent debiasing engine
        Stage 1 data cleansing → 2 independent differential (anchor hidden) → 3 proponent
        → 4 opponent → 5 high-acuity watchdog → 6 evidence retrieval → 7 arbitration → 8 bias audit
        (each stage persisted separately; information isolation is config-driven)
```

Five agents: Data Cleanser, Proponent, Opponent/Devil's Advocate, High-Acuity Watchdog,
Pediatric Consultant/Arbitrator — see [config/agents.yaml](config/agents.yaml).

## Datasets

| Dataset | Role | Status |
|---|---|---|
| Regensburg Pediatric Appendicitis (UCI 938 / Zenodo 10.5281/zenodo.7669442, CC-BY-NC-4.0) | primary, Phase A | audited, conditional PASS |
| Neonatal Sepsis Registry (Mendeley 10.17632/5vdz5cftz7.1) | auxiliary, Phase B | downloaded, audit pending |
| Child Pneumonia Dataset (Mendeley 10.17632/3tx8xymdsv.1) | imaging, Phase C | metadata only — **blocked**: 2,644/2,926 files are pre-augmented, 279 duplicate filenames |

Datasets are never concatenated; each domain gets its own model behind a shared interface.

## License & ethics notes

- Primary dataset is **CC-BY-NC-4.0** (non-commercial research use).
- No PHI, no restricted data, no paywall/auth bypass, no copyrighted figures as training data.
- Citations are machine-verified; no invented references; no invented thresholds.

## Repository map

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full layout
(`src/`, `config/`, `knowledge/`, `research/`, `experiments/`, `outputs/`, `tests/`, `docs/`).
