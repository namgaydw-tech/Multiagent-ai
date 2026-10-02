"""Publication figures for Phase 4 (section 4.5).

Every plotted number is read from the persisted metrics JSON files produced by
real experiment runs — no value is hard-coded, and a missing/undefined metric
(e.g. CBR under the control condition, whose denominator is 0) is drawn as an
explicit gap or omitted with a note, never as a fabricated bar.

Figures produced by :func:`generate_all_figures`:

1. ``fig_anchor_cbr_aor.png``      — CBR under incorrect-anchor conditions,
                                      single-agent vs multi-agent, bootstrap CIs
2. ``fig_bias_reduction.png``      — CBR_single − CBR_multi with CI and p-value
3. ``fig_ablation_matrix.png``     — final accuracy / HFR / CBR across the
                                      11 ablation profiles
4. ``fig_error_taxonomy.png``      — error taxonomy counts (executed rows)
5. ``fig_anchor_outcomes.png``     — initial vs final accuracy per anchor
                                      condition + diagnostic stability (DR)

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless reproducible output
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = ROOT / "outputs/metrics"
FIG_ROOT = ROOT / "outputs/figures"

DISCLAIMER = "Research prototype — not a medical device"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing — run `python scripts/run_phase4.py analyze` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _val(metric: dict[str, Any] | None, key: str = "value") -> float | None:
    if not metric:
        return None
    v = metric.get(key)
    return None if v is None else float(v)


def _err(metric: dict[str, Any] | None) -> list[float] | None:
    if not metric:
        return None
    ci = metric.get("ci95")
    if not ci or ci[0] is None or metric.get("value") is None:
        return None
    v = float(metric["value"])
    return [v - float(ci[0]), float(ci[1]) - v]


def fig_anchor_cbr_aor(bias: dict[str, Any], out: Path) -> Path:
    """CBR for the three incorrect-anchor conditions: single vs multi-agent."""
    conds = ["incorrect_anchor", "high_confidence_incorrect_anchor",
             "low_confidence_incorrect_anchor"]
    multi = bias["anchor_experiment"]["conditions"]
    single = bias["single_baselines"].get("3_lightgbm_plus_single_llm", {})
    x = range(len(conds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (label, source) in ((-w / 2, ("multi-agent (9_full_with_rag)", multi)),
                                    (w / 2, ("single-agent (3_lightgbm_plus_single_llm)", single))):
        vals = [_val((source.get(c) or {}).get("CBR")) for c in conds]
        errs = [_err((source.get(c) or {}).get("CBR")) for c in conds]
        ax.bar([i + offset for i in x], [v if v is not None else 0 for v in vals],
               w, label=label)
        # CI error bars drawn manually (asymmetric percentiles)
        for i, (v, err) in enumerate(zip(vals, errs)):
            if v is not None and err:
                ax.errorbar(i + offset, v, yerr=[[err[0]], [err[1]]], fmt="none",
                            ecolor="black", capsize=4, linewidth=1)
        for i, v in enumerate(vals):
            if v is None:
                ax.text(i + offset, 0.02, "n/a", ha="center", fontsize=8, color="red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["incorrect_anchor", "high_confidence\nincorrect",
                        "low_confidence\nincorrect"], fontsize=9)
    ax.set_ylabel("Confirmation Bias Rate (followed incorrect anchor)")
    ax.set_ylim(0, 1)
    ax.set_title("CBR under incorrect-anchor conditions (2000-draw bootstrap 95% CI)\n"
                 f"{DISCLAIMER}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_bias_reduction(bias: dict[str, Any], out: Path) -> Path:
    """Paired bias-reduction comparison with CI and significance annotation."""
    br = bias.get("bias_reduction", {})
    labels = ["single (2_single_llm)", "single (3_+model)", "multi-agent (9)"]
    keys = ["cbr_single_2", "cbr_single_3", "cbr_multi"]
    records = [br.get(k) for k in keys]
    vals = [_val(r) for r in records]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, [v if v is not None else 0 for v in vals], 0.55,
                  color=["#8da0cb", "#66c2a5", "#fc8d62"])
    for bar, r, v in zip(bars, records, vals):
        err = _err(r)
        if v is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "n/a",
                    ha="center", color="red", fontsize=9)
        elif err:
            ax.errorbar(bar.get_x() + bar.get_width() / 2, v,
                        yerr=[[err[0]], [err[1]]], fmt="none", ecolor="black",
                        capsize=5)
        if r and r.get("n_den"):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min((v or 0) + 0.04, 0.97), f"n={r['n_den']}",
                    ha="center", fontsize=8)
    diff = br.get("paired_diff") or {}
    mcn = br.get("mcnemar") or {}
    note = (f"CBR_single − CBR_multi = {diff.get('diff')} "
            f"(95% CI {diff.get('ci95')}, p={diff.get('p_value')}); "
            f"McNemar exact p={mcn.get('p_value')} "
            f"(b={mcn.get('b')}, c={mcn.get('c')})")
    ax.set_ylabel("Confirmation Bias Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Anchoring bias reduction: single- vs multi-agent\n"
                 f"{note}\n{DISCLAIMER}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ablation_matrix(bias: dict[str, Any], out: Path) -> Path:
    """Final accuracy / HFR / CBR across the 11 ablation profiles."""
    ablations = bias.get("ablations", {})
    profiles = sorted(ablations)
    setting = bias.get("ablation_anchor_setting", "incorrect_anchor")
    accuracy, hfr, cbr = [], [], []
    for p in profiles:
        subset = ablations[p].get(setting) or {}
        accuracy.append(_val(subset.get("final_accuracy")))
        hfr.append(_val(subset.get("HFR")))
        cbr.append(_val(subset.get("CBR")))
    x = range(len(profiles))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - w for i in x], [v or 0 for v in accuracy], w, label="final accuracy")
    ax.bar(list(x), [v or 0 for v in hfr], w, label="HFR (harmful flip rate)")
    ax.bar([i + w for i in x], [v or 0 for v in cbr], w, label="CBR")
    for i, (a, h, c) in enumerate(zip(accuracy, hfr, cbr)):
        for off, v in ((-w, a), (0, h), (w, c)):
            if v is None:
                ax.text(i + off, 0.015, "n/a", ha="center", fontsize=7,
                        color="red", rotation=90)
    ax.set_xticks(list(x))
    ax.set_xticklabels([p.replace("_", "\n", 1) for p in profiles],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title(f"Ablation matrix under the {setting} condition "
                 f"(N cases from executed runs)\n{DISCLAIMER}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_error_taxonomy(error_summary: dict[str, Any], out: Path) -> Path:
    """Tag counts from the executed error taxonomy."""
    counts = error_summary.get("tag_counts", {})
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4c72b0")
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("number of test cases (rep 0)")
    ax.set_title(f"Error taxonomy — test partition ({error_summary.get('n_cases')} cases)\n"
                 f"{DISCLAIMER}", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_anchor_outcomes(bias: dict[str, Any], out: Path) -> Path:
    """Initial vs final accuracy per anchor condition + DR annotation."""
    conds = list(bias["anchor_experiment"]["conditions"])
    multi = bias["anchor_experiment"]["conditions"]
    initial = [_val((multi[c]).get("initial_accuracy")) for c in conds]
    final = [_val((multi[c]).get("final_accuracy")) for c in conds]
    x = range(len(conds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar([i - w / 2 for i in x], [v or 0 for v in initial], w,
           label="initial (anchor present else model top)")
    ax.bar([i + w / 2 for i in x], [v or 0 for v in final], w,
           label="final (multi-agent arbitration)")
    for i, c in enumerate(conds):
        dr = (multi[c].get("DR") or {}).get("value")
        if dr is not None:
            ax.text(i, 1.01, f"DR={dr}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy vs ground truth")
    ax.set_title("Diagnostic accuracy before vs after the debate, per anchor "
                 f"condition (DR = stability across repetitions)\n{DISCLAIMER}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all_figures(metrics_root: Path | None = None,
                         fig_root: Path | None = None) -> list[Path]:
    """Generate every publication figure from persisted metrics; return paths."""
    mroot = metrics_root or METRICS_ROOT
    froot = fig_root or FIG_ROOT
    froot.mkdir(parents=True, exist_ok=True)
    bias = _load(mroot / "bias_metrics.json")
    error = _load(mroot / "error_analysis.json")
    written = [
        fig_anchor_cbr_aor(bias, froot / "fig_anchor_cbr_aor.png"),
        fig_bias_reduction(bias, froot / "fig_bias_reduction.png"),
        fig_ablation_matrix(bias, froot / "fig_ablation_matrix.png"),
        fig_error_taxonomy(error, froot / "fig_error_taxonomy.png"),
        fig_anchor_outcomes(bias, froot / "fig_anchor_outcomes.png"),
    ]
    return written
