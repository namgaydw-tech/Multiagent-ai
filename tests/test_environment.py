"""Environment sanity checks: Python version, required imports, env template completeness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_version():
    assert sys.version_info >= (3, 11), "project requires Python 3.11+"


def test_core_imports():
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import yaml  # noqa: F401
    import openpyxl  # noqa: F401


def test_env_example_has_required_keys():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ["LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "RANDOM_SEED"]:
        assert key in env, key
    # no real secrets committed
    assert "sk-" not in env


def test_requirements_has_phase1_core():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ["pandas", "numpy", "openpyxl", "PyYAML", "pytest", "lightgbm", "shap"]:
        assert pkg in req, pkg


def test_project_structure_exists():
    for d in [
        "config", "data/raw", "data/manifests", "research", "src/data", "src/agents",
        "src/orchestration", "src/bias", "src/evaluation", "knowledge", "experiments",
        "notebooks", "tests", "outputs/metrics", "outputs/audits", "docs",
    ]:
        assert (ROOT / d).exists(), d
