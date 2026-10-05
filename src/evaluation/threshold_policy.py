"""Phase 5 threshold-selection policy (section G).

PRIMARY OBJECTIVE   maximize validation Macro F0.5
SUBJECT TO          positive-class sensitivity >= ``sensitivity_floor`` (default 0.90)
INFEASIBLE          maximize sensitivity first, Macro F0.5 as tie-breaker

Rules enforced here (and by ``tests/test_phase5.py``):

* selection consumes **validation predictions only** — no test label can reach
  this function by construction (it accepts only ``y_val``/``p_val``);
* the returned record is the persisted lock artifact: threshold, objective,
  floor, feasibility, and the ``fitted_on: validation`` marker;
* the floor is a **research operating point**, not a clinically validated
  threshold (disclaimer persisted with every artifact).

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.phase5_metrics import macro_f05

DEFAULT_SENSITIVITY_FLOOR = 0.90


def select_threshold_macro_f05(y_val: np.ndarray, p_val: np.ndarray, *,
                               sensitivity_floor: float = DEFAULT_SENSITIVITY_FLOOR,
                               beta: float = 0.5,
                               grid: np.ndarray | None = None) -> dict[str, Any]:
    """Locked threshold from validation data (section G policy).

    Parameters
    ----------
    y_val, p_val:
        validation labels and positive-class probabilities (never test data).
    sensitivity_floor:
        minimum positive-class sensitivity required for feasibility.
    """
    if beta != 0.5:
        raise ValueError("Phase 5 pre-registers beta=0.5 (Macro F0.5) as the objective")
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    if len(y) == 0:
        raise ValueError("empty validation partition — cannot select a threshold")
    if (y == 1).sum() == 0:
        raise ValueError("validation partition has no positive cases — sensitivity undefined")

    grid = np.asarray(grid) if grid is not None else np.round(np.arange(0.01, 1.00, 0.01), 4)
    rows = []
    for t in grid:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({"threshold": float(t), "sensitivity": float(sens),
                     "macro_f0_5": macro_f05(y, yhat)})

    feasible = [r for r in rows if r["sensitivity"] >= sensitivity_floor]
    if feasible:
        best = max(feasible, key=lambda r: (round(r["macro_f0_5"], 6), -r["threshold"]))
        feasible_flag, note = True, (
            f"max validation Macro F0.5 subject to sensitivity >= {sensitivity_floor}")
    else:
        best = max(rows, key=lambda r: (round(r["sensitivity"], 6),
                                        round(r["macro_f0_5"], 6), -r["threshold"]))
        feasible_flag, note = False, (
            f"no threshold reached sensitivity >= {sensitivity_floor}; fell back to "
            "max sensitivity with Macro F0.5 as tie-breaker")
    return {
        "threshold": round(best["threshold"], 4),
        "val_macro_f0_5": round(best["macro_f0_5"], 6),
        "val_sensitivity": round(best["sensitivity"], 6),
        "sensitivity_floor": float(sensitivity_floor),
        "floor_satisfied": bool(feasible_flag),
        "objective": "maximize validation Macro F0.5 s.t. sensitivity >= floor",
        "note": note,
        "fitted_on": "validation",
        "locked_before_test": True,
        "disclaimer": ("Research operating point, NOT a clinically validated threshold. "
                       "Selected on validation predictions only; the test partition is "
                       "evaluated once after locking."),
        "beta": 0.5,
        "grid_n": int(len(grid)),
    }


def sweep_curve(y_val: np.ndarray, p_val: np.ndarray,
                grid: np.ndarray | None = None) -> list[dict[str, float]]:
    """Full (threshold, sensitivity, Macro F0.5) sweep for figure 3."""
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    grid = np.asarray(grid) if grid is not None else np.round(np.arange(0.01, 1.00, 0.01), 4)
    out = []
    for t in grid:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        out.append({"threshold": float(t),
                    "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                    "macro_f0_5": macro_f05(y, yhat)})
    return out
