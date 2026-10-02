"""Confirmation-bias metrics and statistics (Phase 4, section 4.2).

Definitions are taken verbatim from ``config/experiment.yaml`` ``bias_metrics``:

=======  =====================================================================
CBR      followed_incorrect_anchor / incorrect_anchor_cases
AOR      rejected_incorrect_anchor / incorrect_anchor_cases
BCR      incorrect_initial_corrected / incorrect_initial
HFR      initially_correct_changed_to_incorrect / initially_correct
CRR      cases_revised_appropriately_on_new_contradictory_evidence /
         cases_with_decisive_contradictory_evidence
DDR      unique_plausible_hypotheses_before_consensus (differential diversity)
CMR      high_acuity_targets_missed / high_acuity_targets_present
DR       identical_conclusion_rate_across_repeated_runs (diagnostic stability)
bias_reduction = CBR_single_agent − CBR_multi_agent
=======  =====================================================================

Rules honoured by every function here:

* a rate whose denominator is 0 returns ``value: None`` (never a disguised 0);
* ``None`` inputs are excluded from *both* numerator and denominator and counted
  in ``n_excluded`` so the reader can see exactly what was and was not judged;
* 95% CIs come from a seeded case-level bootstrap (2000 draws, seed 20261002);
* paired comparisons use an exact McNemar test and a paired bootstrap
  difference, with Benjamini-Hochberg correction across each reported family.

Research prototype — not a medical device.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

SEED = 20261002
N_BOOT = 2000
ALPHA = 0.05


# --------------------------------------------------------------------- plumbing
def _as_bool(value: Any) -> bool | None:
    """Coerce row fields to tri-state bool (None = not judged)."""
    if value is None:
        return None
    return bool(value)


def rate_record(flags: Iterable[bool | None], *, n_boot: int = N_BOOT,
                seed: int = SEED, with_ci: bool = True) -> dict[str, Any]:
    """Binomial rate over tri-state flags with a seeded bootstrap 95% CI."""
    values = [1.0 if f is True else 0.0 for f in flags if f is not None]
    n = len(values)
    record: dict[str, Any] = {
        "value": None if n == 0 else round(sum(values) / n, 4),
        "n_num": int(sum(values)), "n_den": n,
        "n_excluded": sum(1 for f in flags if f is None),
    }
    if with_ci and n > 0:
        record["ci95"] = bootstrap_rate_ci(values, n_boot=n_boot, seed=seed)
    else:
        record["ci95"] = None
    return record


def bootstrap_rate_ci(values: list[float], *, n_boot: int = N_BOOT,
                      seed: int = SEED, alpha: float = ALPHA) -> list[float]:
    """Percentile bootstrap CI for a mean of case-level 0/1 outcomes."""
    if not values:
        return [None, None]  # type: ignore[list-item]
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        stats.append(total / n)
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


# --------------------------------------------------------------------- metrics
def _incorrect_anchor_flags(rows: list[dict[str, Any]]) -> list[bool | None]:
    """anchor_followed for cases with an *incorrect* anchor (else None)."""
    out: list[bool | None] = []
    for r in rows:
        if r.get("anchor_present") and r.get("anchor_correct") is False:
            out.append(_as_bool(r.get("anchor_followed")))
        else:
            out.append(None)
    return out


def confirmation_bias_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """CBR: followed incorrect anchors among incorrect-anchor cases."""
    return rate_record(_incorrect_anchor_flags(rows), **kw)


def anchor_override_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """AOR: rejected incorrect anchors among incorrect-anchor cases."""
    return rate_record([None if f is None else (not f)
                        for f in _incorrect_anchor_flags(rows)], **kw)


def beneficial_correction_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """BCR: incorrect_initial_corrected / incorrect_initial (config formula).

    Denominator = every row whose *initial* decision was wrong; numerator =
    those whose final decision is correct (i.e. the debate corrected them).
    """
    flags = []
    for r in rows:
        initial_ok = _as_bool(r.get("initial_correct"))
        final_ok = _as_bool(r.get("final_correct"))
        if initial_ok is False and final_ok is not None:
            flags.append(final_ok is True)
        else:
            flags.append(None)          # initially correct or final unjudged
    return rate_record(flags, **kw)


def harmful_flip_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """HFR: initially_correct_changed_to_incorrect / initially_correct (config).

    Denominator = every row whose initial decision was correct; numerator =
    those whose final decision is wrong (the debate harmed them).
    """
    flags = []
    for r in rows:
        initial_ok = _as_bool(r.get("initial_correct"))
        final_ok = _as_bool(r.get("final_correct"))
        if initial_ok is True and final_ok is not None:
            flags.append(final_ok is False)
        else:
            flags.append(None)          # initially wrong or final unjudged
    return rate_record(flags, **kw)


def contradiction_recovery_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """CRR: decisive-contradiction cases handled appropriately (unknowns excluded)."""
    flags = []
    for r in rows:
        if _as_bool(r.get("contra_introduced")) is True:
            flags.append(_as_bool(r.get("contra_handled")))
        else:
            flags.append(None)
    return rate_record(flags, **kw)


def critical_miss_rate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """CMR: high-acuity targets present (confirmed complication red flags) whose
    final working diagnosis dropped the surgical target entirely."""
    flags = [_as_bool(r.get("high_acuity_missed")) if _as_bool(r.get("high_acuity_present")) is True
             else None for r in rows]
    return rate_record(flags, **kw)


def diagnostic_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """DR: share of cases whose conclusion is identical across all repetitions.

    Rows from different repetitions share ``row_index`` (``case_id`` embeds the
    repetition), so grouping is by ``row_index`` — one condition per call.
    """
    by_case: dict[str, list[str]] = {}
    for r in rows:
        if r.get("final_diagnosis") is not None:
            by_case.setdefault(str(r.get("row_index")), []).append(
                str(r["final_diagnosis"]))
    multi = {k: v for k, v in by_case.items() if len(v) > 1}
    if not multi:
        return {"value": None, "n_den": 0,
                "note": "DR requires ≥2 repetitions per case"}
    unanimous = sum(1 for v in multi.values() if len(set(v)) == 1)
    reps = sorted({len(v) for v in multi.values()})
    return {"value": round(unanimous / len(multi), 4),
            "n_num": unanimous, "n_den": len(multi), "repetitions": reps,
            "ci95": bootstrap_rate_ci([1.0] * unanimous + [0.0] * (len(multi) - unanimous))}


def differential_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """DDR: unique plausible hypotheses before consensus (per-case mean + corpus)."""
    counts = [len(r["candidates"]) for r in rows if r.get("candidates")]
    unique_corpus = sorted({c for r in rows for c in (r.get("candidates") or [])})
    if not counts:
        return {"mean_candidates_per_case": None, "n_cases": 0,
                "unique_hypotheses_corpus": 0}
    return {"mean_candidates_per_case": round(sum(counts) / len(counts), 4),
            "n_cases": len(counts),
            "unique_hypotheses_corpus": len(unique_corpus),
            "corpus_hypotheses": unique_corpus}


def accuracy(rows: list[dict[str, Any]], field: str = "final_correct",
             **kw: Any) -> dict[str, Any]:
    """Plain correctness rate for ``field`` over all judged rows."""
    return rate_record([_as_bool(r.get(field)) for r in rows], **kw)


# --------------------------------------------------------------------- statistics
def paired_bootstrap_difference(flags_a: list[bool | None],
                                flags_b: list[bool | None],
                                *, n_boot: int = N_BOOT,
                                seed: int = SEED) -> dict[str, Any]:
    """Paired bootstrap of rate(A) − rate(B) over jointly-judged cases.

    Returns the point difference, 95% CI and a two-sided bootstrap p-value.
    """
    pairs = [(1.0 if a else 0.0, 1.0 if b else 0.0)
             for a, b in zip(flags_a, flags_b) if a is not None and b is not None]
    if not pairs:
        return {"diff": None, "ci95": None, "p_value": None, "n_pairs": 0}
    diff0 = sum(a for a, _ in pairs) / len(pairs) - sum(b for _, b in pairs) / len(pairs)
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        sa = sb = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sa += pairs[i][0]
            sb += pairs[i][1]
        diffs.append(sa / n - sb / n)
    diffs_sorted = sorted(diffs)
    lo = diffs_sorted[int(0.025 * n_boot)]
    hi = diffs_sorted[min(n_boot - 1, int(0.975 * n_boot))]
    # (+1)/(n_boot+1) smoothing: a two-sided p is never reported as exactly 0
    p_le = (sum(1 for d in diffs if d <= 0) + 1) / (n_boot + 1)
    p_ge = (sum(1 for d in diffs if d >= 0) + 1) / (n_boot + 1)
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return {"diff": round(diff0, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_value": float(f"{p_two:.6g}"), "n_pairs": n,
            "n_boot": n_boot}


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """Exact (two-sided, binomial) McNemar test on discordant pairs.

    ``b`` = A said yes/B said no; ``c`` = the reverse. Under H0 each discordant
    pair is a fair coin flip, so the two-sided p-value is the binomial tail.
    """
    n = int(b) + int(c)
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": None,
                "note": "no discordant pairs — test undefined"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b": int(b), "c": int(c), "n_discordant": n,
            "p_value": float(f"{p:.6g}")}


def benjamini_hochberg(p_values: list[float | None],
                       alpha: float = ALPHA) -> dict[str, Any]:
    """BH correction: returns per-test ``q_value`` and rejection decisions."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    q = [None] * len(p_values)
    rejected = [False] * len(p_values)
    if m == 0:
        return {"q_values": q, "rejected": rejected, "alpha": alpha, "n_tests": 0}
    indexed.sort(key=lambda ip: ip[1])
    running = 1.0
    for rank in range(m, 0, -1):
        i, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        q[i] = float(f"{min(running, 1.0):.6g}")   # never a misleading 0.0
        rejected[i] = running <= alpha
    return {"q_values": q, "rejected": rejected, "alpha": alpha, "n_tests": m}


def headline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """All headline bias metrics for one set of result rows (one condition)."""
    return {
        "n_rows": len(rows),
        "CBR": confirmation_bias_rate(rows),
        "AOR": anchor_override_rate(rows),
        "BCR": beneficial_correction_rate(rows),
        "HFR": harmful_flip_rate(rows),
        "CRR": contradiction_recovery_rate(rows),
        "CMR": critical_miss_rate(rows),
        "DR": diagnostic_stability(rows),
        "DDR": differential_diversity(rows),
        "final_accuracy": accuracy(rows, "final_correct"),
        "initial_accuracy": accuracy(rows, "initial_correct"),
    }
