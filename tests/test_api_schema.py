"""
Schema and logic tests for the FastAPI prediction endpoint.
Tests the request/response structure without needing a running server.
"""
import pytest
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestFeatureEngineering:
    def test_lag_features_created(self):
        import pandas as pd
        from data.feature_engineering import engineer_features, get_feature_names
        df = pd.read_csv(ROOT / "data/raw/debutanizer.csv", parse_dates=["timestamp"])
        out = engineer_features(df)
        feats = get_feature_names(out)
        assert "u1_top_tray_temp_lag1" in feats
        assert "u6_bottom_temp_lag3"   in feats
        assert "reflux_ratio"          in feats
        assert "delta_top_bottom"      in feats

    def test_no_nan_after_engineering(self):
        import pandas as pd
        from data.feature_engineering import engineer_features
        df = pd.read_csv(ROOT / "data/raw/debutanizer.csv", parse_dates=["timestamp"])
        out = engineer_features(df)
        assert out.isnull().sum().sum() == 0, "No NaNs expected after dropna"

    def test_feature_count(self):
        import pandas as pd
        from data.feature_engineering import engineer_features, get_feature_names
        df = pd.read_csv(ROOT / "data/raw/debutanizer.csv", parse_dates=["timestamp"])
        out = engineer_features(df)
        feats = get_feature_names(out)
        assert len(feats) >= 30, f"Expected ≥30 features, got {len(feats)}"


class TestModelMetrics:
    def test_model_r2_acceptable(self):
        import json
        metrics = json.loads((ROOT / "results/registry/latest_metrics.json").read_text())
        assert metrics["test_r2"] > 0.60, f"R² below acceptable threshold: {metrics['test_r2']:.4f}"

    def test_model_rmse_reasonable(self):
        import json
        metrics = json.loads((ROOT / "results/registry/latest_metrics.json").read_text())
        assert metrics["test_rmse"] < 0.20, f"RMSE too high: {metrics['test_rmse']:.4f}"
