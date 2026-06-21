"""
Feature engineering for the debutanizer soft sensor.
Adds lag features and rolling statistics — standard in process soft sensing.
"""
import pandas as pd
import numpy as np
from pathlib import Path


FEATURE_COLS = [
    "u1_top_tray_temp", "u2_top_temp", "u3_reflux_flow",
    "u4_feed_flow", "u5_6th_tray_temp", "u6_bottom_temp", "u7_pressure"
]
TARGET = "y_butane_content"
LAG_STEPS = [1, 2, 3]
ROLLING_WINDOWS = [6, 12]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # Lag features (process dynamics: current output depends on past inputs)
    for col in FEATURE_COLS:
        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    # Rolling mean (smooth out measurement noise)
    for col in ["u1_top_tray_temp", "u5_6th_tray_temp", "u6_bottom_temp"]:
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll{w}"] = df[col].rolling(w).mean()
    # Temperature differentials (physically meaningful)
    df["delta_top_bottom"] = df["u1_top_tray_temp"] - df["u6_bottom_temp"]
    df["delta_temp_56"]    = df["u5_6th_tray_temp"] - df["u6_bottom_temp"]
    df["reflux_ratio"]     = df["u3_reflux_flow"] / (df["u4_feed_flow"] + 1e-6)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_names(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ["timestamp", TARGET]]


if __name__ == "__main__":
    raw = pd.read_csv(Path(__file__).parent.parent.parent / "data/raw/debutanizer.csv",
                      parse_dates=["timestamp"])
    out = engineer_features(raw)
    dest = Path(__file__).parent.parent.parent / "data/processed/debutanizer_features.parquet"
    out.to_parquet(dest, index=False)
    features = get_feature_names(out)
    print(f"Feature matrix: {out.shape} | {len(features)} features")
    print("Features:", features[:10], "...")
