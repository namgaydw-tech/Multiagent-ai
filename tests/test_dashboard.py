"""Dashboard tests: data bundle integrity, asset wiring, and honest-empty-state guards.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
DATA = DASH / "data/dashboard.json"


def _build():
    """Rebuild the dashboard bundle from repo artifacts."""
    import runpy
    import sys

    old_argv = sys.argv
    sys.argv = ["build_data.py"]
    try:
        runpy.run_path(str(DASH / "build_data.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv


@pytest.fixture(scope="module", autouse=True)
def build_bundle():
    _build()


class TestBundle:
    def test_data_exists_and_parses(self):
        assert DATA.exists()
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        assert payload["audit"]["dimensions"]["n_rows"] == 782
        assert payload["audit"]["verdict"].startswith("PASS")
        assert payload["literature"]["count"] == len(payload["literature"]["entries"]) >= 40
        assert payload["project"]["disclaimer"].lower().startswith("research prototype")

    def test_assets_exist(self):
        for f in ["index.html", "styles.css", "app.js"]:
            assert (DASH / f).exists(), f

    def test_index_references_assets_and_fetches_bundle(self):
        html = (DASH / "index.html").read_text(encoding="utf-8")
        assert 'href="styles.css"' in html
        assert 'src="app.js"' in html
        assert "Not a medical device" in html
        js = (DASH / "app.js").read_text(encoding="utf-8")
        assert 'fetch("data/dashboard.json"' in js

    def test_no_external_resources(self):
        """Offline-safe + Vercel-safe: no CDN or third-party origins."""
        for f in ["index.html", "styles.css", "app.js"]:
            text = (DASH / f).read_text(encoding="utf-8")
            externals = re.findall(r"https?://[^\s\"')]+", text)
            bad = [u for u in externals if "data:image" not in u and "w3.org" not in u]
            # allow only documented, non-loaded references (schemas/comments) - none expected here
            assert not bad, f"{f}: unexpected external URL(s): {bad}"

    def test_docs_copied_into_bundle(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        for d in payload["docs"]:
            assert (DASH / d["path"]).exists(), d["path"]


class TestHonestyGuards:
    def test_metrics_are_pending_not_fabricated(self):
        js = (DASH / "app.js").read_text(encoding="utf-8")
        assert "pending experiment" in js
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        # no numeric bias-metric results exist yet anywhere in the bundle
        assert all("value" not in m for m in payload["bias_metrics"])

    def test_counts_match_source_artifacts(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        with (ROOT / "research/literature_review.csv").open(encoding="utf-8", newline="") as f:
            lit_rows = list(csv.DictReader(f))
        assert payload["literature"]["count"] == len(lit_rows)
        with (ROOT / "research/dataset_inventory.csv").open(encoding="utf-8", newline="") as f:
            inv_rows = list(csv.DictReader(f))
        assert len(payload["datasets"]) == len(inv_rows)

    def test_models_trained_is_zero(self):
        payload = json.loads(DATA.read_text(encoding="utf-8"))
        html = (DASH / "index.html").read_text(encoding="utf-8")
        assert "no model trained" in payload["project"]["status"].lower()
        assert "not a medical device" in html.lower()


class TestVercelConfig:
    def test_vercel_json(self):
        cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
        assert cfg["framework"] == "vite"
        assert cfg["buildCommand"] == "npm run build"
        assert cfg["outputDirectory"] == "frontend/dist"

    def test_bundle_is_static_only(self):
        """Deployed bundle must contain no server code needing Python at runtime."""
        served = [p for p in (ROOT / "frontend" / "dist").rglob("*") if p.is_file()]
        py_runtime = [p for p in served if p.suffix == ".py"]
        assert not py_runtime
        assert any(p.name == "index.html" for p in served)
