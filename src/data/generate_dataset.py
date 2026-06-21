"""
Generate synthetic debutanizer dataset with realistic process correlations.
Based on the structure of the Fortuna & MacGregor (1995) debutanizer benchmark.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 2500

# Time index (hourly samples, ~104 days)
t = np.arange(N)

# Simulate correlated process inputs
# u1: Top tray temperature (°C) ~ 65–75°C
u1_base = 70 + 3*np.sin(2*np.pi*t/168) + np.random.randn(N)*0.8
# u2: Top temperature (°C) ~ 60–70°C
u2_base = 65 + 2*np.sin(2*np.pi*t/168 + 0.5) + np.random.randn(N)*0.7
# u3: Reflux flow (m³/h)
u3_base = 1.8 + 0.3*np.sin(2*np.pi*t/72) + np.random.randn(N)*0.1
# u4: Flow (m³/h)
u4_base = 3.2 + 0.2*np.sin(2*np.pi*t/96 + 1.0) + np.random.randn(N)*0.12
# u5: 6th tray temperature (°C) ~ 80–95°C
u5_base = 88 + 4*np.sin(2*np.pi*t/120 + 0.3) + np.random.randn(N)*1.1
# u6: Bottom temperature (°C) ~ 90–105°C
u6_base = 97 + 5*np.sin(2*np.pi*t/144 + 0.8) + np.random.randn(N)*1.3
# u7: Pressure (kPa) ~ 300–350
u7_base = 325 + 10*np.sin(2*np.pi*t/200) + np.random.randn(N)*2.5

# Target: butane (C4) content in bottom product (wt%)
# Physics-inspired: lower top temp => more butane leaks through
y = (
    0.8
    - 0.045 * (u1_base - 70)
    - 0.030 * (u2_base - 65)
    + 0.060 * (u3_base - 1.8)
    - 0.020 * (u4_base - 3.2)
    + 0.008 * (u5_base - 88)
    - 0.012 * (u6_base - 97)
    + 0.001 * (u7_base - 325)
    + 0.15*np.sin(2*np.pi*t/500)
    + np.random.randn(N)*0.05
)
y = np.clip(y, 0.05, 2.5)

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=N, freq="h"),
    "u1_top_tray_temp":   np.round(u1_base, 3),
    "u2_top_temp":        np.round(u2_base, 3),
    "u3_reflux_flow":     np.round(u3_base, 4),
    "u4_feed_flow":       np.round(u4_base, 4),
    "u5_6th_tray_temp":   np.round(u5_base, 3),
    "u6_bottom_temp":     np.round(u6_base, 3),
    "u7_pressure":        np.round(u7_base, 2),
    "y_butane_content":   np.round(y, 4),
})

out = Path(__file__).parent.parent.parent / "data" / "raw" / "debutanizer.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} rows → {out}")
print(df.describe().round(3).to_string())
