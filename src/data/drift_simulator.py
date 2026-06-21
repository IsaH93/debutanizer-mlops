"""
Simulate realistic distribution shift across 12 weekly batches.
Weeks 1-6: stable (reference). Weeks 7-12: gradual drift in feed conditions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.parent / "data/simulated"
PROC_DIR = Path(__file__).parent.parent.parent / "data/processed"


def simulate_drift(df: pd.DataFrame, week: int) -> pd.DataFrame:
    df = df.copy()
    if week <= 6:
        return df
    severity = (week - 6) * 0.4   # 0.4 sigma shift per extra week
    np.random.seed(week * 7)
    # Drift: feed flow increases (heavier feed coming in)
    df["u4_feed_flow"]      += severity * 0.15
    df["u4_feed_flow_lag1"] += severity * 0.15
    df["u4_feed_flow_lag2"] += severity * 0.12
    # Drift: bottom temperature creeps up (fouling effect)
    df["u6_bottom_temp"]      += severity * 0.9
    df["u6_bottom_temp_lag1"] += severity * 0.9
    df["u6_bottom_temp_roll6"]+= severity * 0.7
    # Reflux ratio degrades slightly
    df["reflux_ratio"]        -= severity * 0.03
    # Add extra noise to represent measurement degradation
    noise_cols = ["u1_top_tray_temp", "u5_6th_tray_temp", "u7_pressure"]
    for col in noise_cols:
        if col in df.columns:
            df[col] += np.random.randn(len(df)) * (0.1 * severity)
    return df


if __name__ == "__main__":
    df = pd.read_parquet(PROC_DIR / "debutanizer_features.parquet")
    n_per_week = len(df) // 12
    for week in range(1, 13):
        chunk = df.iloc[(week-1)*n_per_week : week*n_per_week].copy()
        chunk = simulate_drift(chunk, week)
        out = SIM_DIR / f"week_{week:02d}.parquet"
        chunk.to_parquet(out, index=False)
        tag = "DRIFT" if week > 6 else "stable"
        print(f"Week {week:02d} [{tag}]: {len(chunk)} rows → {out.name}")
