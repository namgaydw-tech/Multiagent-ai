# Experiments

**Status: scaffolding only — no experiment has been executed yet.** Directories here will hold
run configurations and pointers to outputs; all results live under `outputs/`.

| Directory | Purpose | Planned conditions |
|---|---|---|
| `baseline/` | LightGBM alone, single LLM, LightGBM + single LLM, self-reflection | ablation matrix items 1–4 (`config/experiment.yaml`) |
| `anchoring/` | Anchor-injection conditions (control, correct, incorrect, high/low confidence, model anchor, no-model anchor) | paired conditions with identical patient features |
| `confirmation_bias/` | Headline bias experiments: CBR, AOR, BCR, HFR, CRR, DDR, CMR, stability | single-agent vs cooperative multi-agent vs adversarial framework |
| `ablations/` | The 11-condition ablation matrix incl. RAG on/off and isolation variants A/B/C | statistical comparison with bootstrap CIs + McNemar |
| `external_validation/` | Phase B neonatal-sepsis replication (separate pipeline) | generalisation of the debiasing architecture |

Rules:

1. No run may start before `docs/DATASET_AUDIT.md` conditions are satisfied (phase gate).
2. Every run writes per-case audit JSON (`outputs/audits/`) before aggregate metrics.
3. Figures are generated only from stored metrics — never hand-authored numbers.
4. Test-set labels are touched exactly once, at final evaluation.
