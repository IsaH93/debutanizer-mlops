"""
Tests for drift detector — verifies clean data returns no drift,
drifted data returns drift on the correct features.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from monitoring.drift_detector import compute_psi, PSI_THRESHOLD


def make_series(mean=0.0, std=1.0, n=500, seed=0):
    np.random.seed(seed)
    return np.random.normal(mean, std, n)


class TestComputePSI:
    def test_identical_distributions_zero_psi(self):
        x = make_series(mean=70, std=2, n=1000, seed=1)
        psi = compute_psi(x, x.copy())
        assert psi < 0.01, f"Identical distributions should have PSI ≈ 0, got {psi:.4f}"

    def test_small_shift_below_threshold(self):
        ref = make_series(mean=70.0, std=2.0, n=1000, seed=2)
        cur = make_series(mean=70.3, std=2.0, n=500,  seed=3)  # tiny shift
        psi = compute_psi(ref, cur)
        assert psi < PSI_THRESHOLD, f"Small shift should be < threshold, got PSI={psi:.4f}"

    def test_large_shift_above_threshold(self):
        ref = make_series(mean=70.0, std=2.0, n=1000, seed=4)
        cur = make_series(mean=75.0, std=2.0, n=500,  seed=5)  # 2.5σ shift
        psi = compute_psi(ref, cur)
        assert psi > PSI_THRESHOLD, f"Large shift should exceed threshold, got PSI={psi:.4f}"

    def test_psi_non_negative(self):
        ref = make_series(seed=6)
        cur = make_series(mean=1.0, seed=7)
        psi = compute_psi(ref, cur)
        assert psi >= 0.0, "PSI must be non-negative"


class TestDriftFiles:
    def test_stable_weeks_low_psi(self):
        """Weeks 1 should show no drift vs reference."""
        import json
        report = json.loads((ROOT / "results/reports/drift_report_week_01.json").read_text())
        max_psi = max(v["psi"] for v in report["features"].values())
        assert max_psi < PSI_THRESHOLD * 2, \
            f"Week 1 max PSI should be low, got {max_psi:.4f}"

    def test_severe_drift_week_12(self):
        """Week 12 should show significant drift."""
        import json
        report = json.loads((ROOT / "results/reports/drift_report_week_12.json").read_text())
        assert report["drift_detected"] is True, "Week 12 should show drift"
        assert report["n_drifted_features"] >= 3, \
            f"Week 12 should have ≥3 drifted features, got {report['n_drifted_features']}"

    def test_escalating_drift(self):
        """Max PSI should generally increase from week 7 to week 12."""
        import json
        psis = []
        for w in [7, 9, 11, 12]:
            r = json.loads((ROOT / f"results/reports/drift_report_week_{w:02d}.json").read_text())
            psis.append(max(v["psi"] for v in r["features"].values()))
        assert psis[-1] > psis[0], \
            f"PSI should escalate from week 7 to 12, got {psis}"
