"""Experiment profiles and batch runners for Phase 4 (section 4.3).

Maps ``config/experiment.yaml`` onto executable conditions:

* **anchor experiment** — 7 anchor conditions on the full framework
  (``9_full_with_rag``: five agents + RAG + isolation C), test partition only;
* **single-agent baselines** — ``2_single_llm`` and
  ``3_lightgbm_plus_single_llm`` over the same anchor conditions, for the
  paired ``bias_reduction = CBR_single − CBR_multi``;
* **ablation matrix** — the 11 conditions of ``ablation_matrix`` under the
  control and incorrect-anchor settings.

Deterministic single-agent baselines (no LLM backend is available here) are
**explicit scoring rules**, not simulated intelligence — each is documented at
its definition and disclosed in docs/PHASE4_REPORT.md:

* ``layer_a``      final answer = LightGBM top class (no agents);
* ``single``       one pass over the record: differential candidates scored by
                   cited evidence, plus a model bonus when the model is visible
                   plus an anchor prior weighted by the anchor's stated
                   confidence (the transparent anchoring-susceptibility term
                   that the multi-agent pipeline deliberately lacks);
* ``single_reflection``: as ``single`` plus one documented revision pass that
                   re-ranks using the cleanser's objective red flags.

Engine profiles run the real state machine with the stage subset of the
condition; stages a profile does not run are recorded as absent (``None``),
never fabricated. Every executed case persists an ``audit.json`` under
``outputs/audits/experiments/<experiment>/<condition>/`` (stopping rule:
``no_results_without_persisted_audit_json``).

Paired-design guarantee: patient features are byte-identical across all
conditions for a given row — only anchors/visibility differ.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.arbitrator import diagnosis_family
from src.agents.independent_differential import extract as differential_extract
from src.agents.data_cleanser import extract as cleanser_extract
from src.agents.schemas import validate_stage
from src.bias.anchor_generator import ANCHOR_CONDITIONS, anchor_for_case, anchor_vocabulary
from src.bias.audit import compute_bias_audit
from src.orchestration import persistence
from src.orchestration.engine import run_case
from src.orchestration.state import make_state

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "outputs/audits/experiments"
METRICS_ROOT = ROOT / "outputs/metrics"
SEED = 20261002


# --------------------------------------------------------------------- profiles
@dataclass(frozen=True)
class Profile:
    """One executable condition from the ablation matrix."""

    name: str
    kind: str                       # layer_a | single | single_reflection | engine
    description: str
    pipeline: tuple[int, ...] = ()  # engine stages (kind == engine)
    ablation: str | None = None     # visibility-policy override label
    rag: bool = False
    sees_model: bool = False        # single* kinds
    adversarial: bool = True

    @property
    def runs_watchdog(self) -> bool:
        return self.kind == "engine" and 5 in self.pipeline

    @property
    def runs_opponent(self) -> bool:
        return self.kind == "engine" and 4 in self.pipeline


FULL = (1, 2, 3, 4, 5, 6, 7, 8)
PROFILES: dict[str, Profile] = {
    "1_lightgbm_only": Profile(
        "1_lightgbm_only", "layer_a",
        "Layer A only: final answer is the LightGBM top class (no agents)."),
    "2_single_llm": Profile(
        "2_single_llm", "single",
        "One deterministic pass over patient evidence (no model output).",
        sees_model=False),
    "3_lightgbm_plus_single_llm": Profile(
        "3_lightgbm_plus_single_llm", "single",
        "One pass fusing patient evidence with the model output.",
        sees_model=True),
    "4_lightgbm_self_reflection": Profile(
        "4_lightgbm_self_reflection", "single_reflection",
        "One pass + one documented reflection revision using objective red flags.",
        sees_model=True),
    "5_generic_multi_agent": Profile(
        "5_generic_multi_agent", "engine",
        "Cooperative multi-agent: no adversarial opponent stage.",
        pipeline=(1, 2, 3, 5, 6, 7, 8), adversarial=False),
    "6_proponent_opponent": Profile(
        "6_proponent_opponent", "engine",
        "Adversarial proponent+opponent debate without the watchdog stage.",
        pipeline=(1, 2, 3, 4, 6, 7, 8)),
    "7_full_five_agent": Profile(
        "7_full_five_agent", "engine",
        "Full five-agent pipeline; RAG follows the repo default (disabled).",
        pipeline=FULL, rag=False),
    "8_full_without_rag": Profile(
        "8_full_without_rag", "engine",
        "Full pipeline with retrieval disabled (config: rag false).",
        pipeline=FULL, rag=False),
    "9_full_with_rag": Profile(
        "9_full_with_rag", "engine",
        "Full pipeline with RAG enabled (the complete framework).",
        pipeline=FULL, rag=True),
    "10_all_see_anchor": Profile(
        "10_all_see_anchor", "engine",
        "Full pipeline under isolation condition A (config maps this label to "
        "A_all_see_prediction); RAG disabled per inheritance from 7.",
        pipeline=FULL, ablation="A_all_see_prediction", rag=False),
    "11_information_isolated": Profile(
        "11_information_isolated", "engine",
        "Full pipeline under isolation condition C (default), RAG disabled per "
        "inheritance from 7.",
        pipeline=FULL, ablation="C_hidden_until_differential_complete", rag=False),
}

# headline condition for the anchor experiment = complete framework
ANCHOR_PROFILE = "9_full_with_rag"


# --------------------------------------------------------------------- scoring
def _model_terms(model_output: dict[str, Any] | None) -> tuple[str, float]:
    """(top class, model support for that class) from a normalized prediction."""
    if not model_output:
        return "", 0.5
    top = str(model_output.get("predicted_class", ""))
    probs = model_output.get("class_probabilities") or {}
    p_pos = float(probs.get("appendicitis", 0.5))
    support = p_pos if top.lower() == "appendicitis" else 1.0 - p_pos
    return top, support


def single_pass_diagnosis(record: dict[str, Any],
                          model_output: dict[str, Any] | None,
                          anchor: dict[str, Any] | None,
                          sees_model: bool,
                          reflection: bool = False) -> dict[str, Any]:
    """One deterministic answer (optionally + one reflection revision).

    Scoring (documented susceptibility model, not hidden tuning):

        score(c) = 0.50 * min(support(c)/5, 1)          cited record support
                 - 0.30 * min(contradictions(c)/3, 1)   cited contradictions
                 + 0.40 * model_support   if sees_model and c ~ model top
                 + 0.60 * anchor_confidence if c ~ anchor family

    The anchor prior is the single-agent analogue of an LLM casually trusting
    the preliminary diagnosis it was shown; the multi-agent pipeline has no
    such term — that difference is exactly what ``bias_reduction`` measures.
    Reflection adds one revision pass where the cleanser's objective red flags
    bump the surgical family (+0.15) when complication evidence is present.
    """
    cleanser = cleanser_extract(record)
    differential = differential_extract(record, cleanser)   # plain dict payload
    candidates = list(differential["candidate_diagnoses"])
    model_top, model_support = _model_terms(model_output)

    def _family_set(label: str) -> str:
        return diagnosis_family(label)

    if sees_model and model_top and _family_set(model_top) not in {
            _family_set(c) for c in candidates}:
        candidates.append(model_top)
    anchor_family = _family_set(anchor["value"]) if anchor and anchor.get("value") else ""
    if anchor_family and anchor_family not in {_family_set(c) for c in candidates}:
        candidates.append(str(anchor["value"]))

    def score(cand: str) -> float:
        supp = len(differential["supporting_evidence"].get(cand, []))
        contra = len(differential["contradictory_evidence"].get(cand, []))
        s = 0.50 * min(supp / 5.0, 1.0) - 0.30 * min(contra / 3.0, 1.0)
        if sees_model and model_top and _family_set(cand) == _family_set(model_top):
            s += 0.40 * model_support
        if anchor and anchor_family and _family_set(cand) == anchor_family:
            s += 0.60 * float(anchor.get("confidence") or 0.0)
        return s

    scores = {c: score(c) for c in candidates}
    passes = [{"pass": 1, "scores": {k: round(v, 4) for k, v in scores.items()}}]
    chosen = max(scores, key=lambda c: (scores[c], -candidates.index(c)))

    if reflection:
        red_flags = list(cleanser["objective_red_flags"] or [])
        surgical = [c for c in candidates if _family_set(c) == "appendicitis"]
        if surgical and red_flags:
            scores[surgical[0]] = round(scores[surgical[0]] + 0.15, 4)
        # one revision allowed: switch only if a rival now clearly leads
        revised = max(scores, key=lambda c: (scores[c], -candidates.index(c)))
        passes.append({"pass": 2, "red_flags_used": len(red_flags),
                       "scores": dict(scores), "revised": revised != chosen})
        chosen = revised

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, second = ordered[0], (ordered[1] if len(ordered) > 1 else (ordered[0][0], 0.0))
    confidence = round(min(max(0.45 + best[1] - max(second[1], 0.0) * 0.5,
                               0.05), 0.95), 4)
    return {"final_diagnosis": chosen, "confidence": confidence,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "candidates": candidates, "passes": passes,
            "rationale": ("deterministic one-pass scoring over cited record "
                          "evidence, model bonus and anchor prior; see "
                          "src/bias/experiment.py single_pass_diagnosis")}


# --------------------------------------------------------------------- row building
def _row_from_audit(audit: dict[str, Any], *, experiment: str, condition: str,
                    profile_name: str, rep: int, row_index: int,
                    ground_truth: str, model_output: dict[str, Any],
                    anchor: dict[str, Any] | None, candidates: list[str] | None,
                    watchdog_risk: str | None, red_flags_n: int | None,
                    urgent_n: int | None, unc_level: str | None,
                    final_confidence: float, elapsed_s: float) -> dict[str, Any]:
    """Uniform result row shared by engine and non-engine conditions."""
    gt_family = diagnosis_family(ground_truth)
    initial = audit["diagnosis_before_debate"]
    final = audit["diagnosis_after_debate"]
    return {
        "experiment": experiment, "condition": condition, "profile": profile_name,
        "rep": rep, "case_id": f"row_{row_index}_rep{rep}", "row_index": row_index,
        "ground_truth": ground_truth,
        "anchor_present": audit["anchor_present"],
        "anchor_value": (anchor or {}).get("value"),
        "anchor_source": (anchor or {}).get("source"),
        "anchor_confidence": (anchor or {}).get("confidence"),
        "anchor_correct": audit["anchor_correct"],
        "anchor_followed": audit["anchor_followed"],
        "model_top": model_output["predicted_class"],
        "model_p_cal": model_output.get("calibrated_probability"),
        "uncertainty_level": unc_level,
        "initial_diagnosis": initial, "final_diagnosis": final,
        "initial_correct": diagnosis_family(initial) == gt_family,
        "final_correct": diagnosis_family(final) == gt_family,
        "final_confidence": final_confidence,
        "diagnosis_changed": audit["diagnosis_changed"],
        "change_beneficial": audit["change_beneficial"],
        "change_harmful": audit["change_harmful"],
        "contra_introduced": audit["contradictory_evidence_introduced"],
        "contra_handled": audit["contradiction_handled_appropriately"],
        "model_vs_final_disagreement": audit["model_vs_final_disagreement"],
        "candidates": candidates, "n_candidates": len(candidates or []),
        "watchdog_risk": watchdog_risk, "red_flags_n": red_flags_n,
        "urgent_rule_outs_n": urgent_n,
        "high_acuity_present": (None if red_flags_n is None else red_flags_n > 0),
        "high_acuity_missed": (None if red_flags_n is None or red_flags_n == 0
                               else diagnosis_family(final) != "appendicitis"),
        "elapsed_s": round(elapsed_s, 4),
        "research_disclaimer": "Research prototype — not a medical device.",
    }


def _save_case_audit(root: Path, case_dir_name: str, payload: dict[str, Any]) -> None:
    """Persist the per-case audit record (stopping rule compliance)."""
    directory = root / case_dir_name
    persistence.save_audit(directory, payload)


# --------------------------------------------------------------------- runners
def run_nonengine_case(predictor, row_index: int, condition: str, profile: Profile,
                       rep: int, bundle: tuple[dict, dict, dict],
                       vocab: dict[str, list[str]], audit_root: Path,
                       experiment: str) -> dict[str, Any]:
    """Execute one layer_a / single / single_reflection case."""
    t0 = time.perf_counter()
    model_output, _shap, uncertainty = bundle
    record = predictor.df.loc[row_index].to_dict()
    ground_truth = str(record.get("Diagnosis"))
    anchor = anchor_for_case(condition, row_index, predictor.df, ground_truth,
                             model_output["predicted_class"], vocab)

    audit_input: dict[str, Any]
    if profile.kind == "layer_a":
        final = model_output["predicted_class"]
        confidence = float(model_output.get("calibrated_probability") or 0.5)
        candidates, risk, red_n, urgent_n = None, None, None, None
    else:
        outcome = single_pass_diagnosis(
            record, model_output, anchor, sees_model=profile.sees_model,
            reflection=(profile.kind == "single_reflection"))
        final = outcome["final_diagnosis"]
        confidence = outcome["confidence"]
        candidates = outcome["candidates"]
        risk, red_n, urgent_n = None, None, None   # no watchdog in these profiles

    audit = compute_bias_audit(
        case_id=f"row_{row_index}_rep{rep}", anchor=anchor,
        ground_truth=ground_truth, model_output=model_output,
        opponent=None,
        arbitrator={"primary_working_diagnosis": final,
                    "multi_agent_confidence": confidence},
        proponent=None)
    elapsed = time.perf_counter() - t0
    row = _row_from_audit(
        audit, experiment=experiment, condition=condition, profile_name=profile.name,
        rep=rep, row_index=row_index, ground_truth=ground_truth,
        model_output=model_output, anchor=anchor, candidates=candidates,
        watchdog_risk=risk, red_flags_n=red_n, urgent_n=urgent_n,
        unc_level=uncertainty["uncertainty_level"], final_confidence=confidence,
        elapsed_s=elapsed)
    _save_case_audit(audit_root, row["case_id"], {
        "experiment": experiment, "condition": condition, "profile": profile.name,
        "kind": profile.kind, "rep": rep, "seed": SEED,
        "config_hash": predictor.config_hash,
        "pipeline": list(profile.pipeline) or profile.kind,
        "anchor": anchor, "ground_truth_label_hidden_until_audit": True,
        "result": {k: row[k] for k in ("initial_diagnosis", "final_diagnosis",
                                       "final_confidence", "diagnosis_changed",
                                       "anchor_followed")},
        "scoring": "single_pass_diagnosis (deterministic, documented)" if
                   profile.kind != "layer_a" else "model top class",
        "disclaimer": "Research prototype — not a medical device."})
    return row


def run_engine_case(predictor, row_index: int, condition: str, profile: Profile,
                    rep: int, bundle: tuple[dict, dict, dict],
                    vocab: dict[str, list[str]], retriever: Any, provider: Any,
                    audit_root: Path, experiment: str) -> dict[str, Any]:
    """Execute one full/partial engine case and harvest the stage-8 audit."""
    t0 = time.perf_counter()
    model_output, shap_explain, uncertainty = bundle
    record = predictor.df.loc[row_index].to_dict()
    ground_truth = str(record.get("Diagnosis"))
    anchor = anchor_for_case(condition, row_index, predictor.df, ground_truth,
                             model_output["predicted_class"], vocab)

    case_id = f"row_{row_index}_rep{rep}"
    state = make_state(case_id, record, anchor=anchor,
                       ablation=profile.ablation or "default",
                       pipeline_stages=tuple(profile.pipeline),
                       rag_enabled=profile.rag,
                       config_hash=predictor.config_hash, seed=SEED)
    state.model_output = model_output
    state.shap_explain = shap_explain
    state.uncertainty = uncertainty

    state = run_case(state, provider=provider, retriever=retriever,
                     persist=True, stage_files=False, out_root=audit_root)
    audit = state.stage_outputs["bias_audit"].model_dump()
    differential = state.stage_outputs["independent_differential"]
    watchdog = state.stage_outputs.get("watchdog")
    elapsed = time.perf_counter() - t0
    row = _row_from_audit(
        audit, experiment=experiment, condition=condition, profile_name=profile.name,
        rep=rep, row_index=row_index, ground_truth=ground_truth,
        model_output=model_output, anchor=anchor,
        candidates=list(differential.candidate_diagnoses),
        watchdog_risk=(watchdog.risk_level if watchdog is not None else None),
        red_flags_n=(len(watchdog.red_flags_present) if watchdog is not None else None),
        urgent_n=(len(watchdog.urgent_rule_out_conditions)
                  if watchdog is not None else None),
        unc_level=uncertainty["uncertainty_level"],
        final_confidence=state.stage_outputs["arbitrator"].multi_agent_confidence,
        elapsed_s=elapsed)
    if provider is not None:
        provider.log.clear()          # bound memory across thousands of runs
    return row


def make_runner(predictor, profile: Profile, vocab: dict[str, list[str]],
                bundles: dict[int, tuple[dict, dict, dict]], *,
                retriever: Any, provider: Any, experiment: str,
                audit_root_base: Path):
    """Return ``run(case_id, condition, row_index, rep) -> row`` for a profile."""
    condition_root = audit_root_base / profile.name

    def _run(condition: str, row_index: int, rep: int) -> dict[str, Any]:
        root = condition_root / condition
        bundle = bundles[row_index]
        if profile.kind == "engine":
            return run_engine_case(predictor, row_index, condition, profile, rep,
                                   bundle, vocab, retriever, provider, root,
                                   experiment)
        return run_nonengine_case(predictor, row_index, condition, profile, rep,
                                  bundle, vocab, root, experiment)
    return _run
