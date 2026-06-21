"""
Drift detection using PSI (Population Stability Index) and KS test.
PSI is the industry-standard metric in process industries.
Reference distribution = training data (first 80% of processed dataset).
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
SIM_DIR    = ROOT / "data/simulated"
PROC_DIR   = ROOT / "data/processed"
REPORT_DIR = ROOT / "results/reports"

PSI_THRESHOLD = 0.20
KS_ALPHA      = 0.05

MONITORED_FEATURES = [
    "u1_top_tray_temp", "u4_feed_flow", "u6_bottom_temp",
    "u3_reflux_flow", "reflux_ratio", "delta_top_bottom"
]


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins=10) -> float:
    eps = 1e-6
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] -= eps; bins[-1] += eps
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current,   bins=bins)[0]
    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def get_reference() -> pd.DataFrame:
    df = pd.read_parquet(PROC_DIR / "debutanizer_features.parquet")
    split = int(len(df) * 0.80)
    return df.iloc[:split]


def run_drift_check(week: int) -> dict:
    ref_data = get_reference()
    current  = pd.read_parquet(SIM_DIR / f"week_{week:02d}.parquet")

    results  = {}
    any_drift = False
    for feat in MONITORED_FEATURES:
        if feat not in ref_data.columns or feat not in current.columns:
            continue
        ref_vals = ref_data[feat].dropna().values
        cur_vals = current[feat].dropna().values
        psi = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        feat_drift = bool(psi > PSI_THRESHOLD or ks_pval < KS_ALPHA)
        if feat_drift:
            any_drift = True
        results[feat] = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "drift_detected": feat_drift,
        }

    report = {
        "week": int(week),
        "drift_detected": bool(any_drift),
        "features": results,
        "n_drifted_features": int(sum(1 for v in results.values() if v["drift_detected"])),
        "psi_threshold": PSI_THRESHOLD,
        "ks_alpha": KS_ALPHA,
    }
    out = REPORT_DIR / f"drift_report_week_{week:02d}.json"
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(f"\n{'Week':>6} | {'Drift?':>8} | {'#Drifted':>8} | {'Max PSI':>8}")
    print("-" * 45)
    all_reports = []
    for w in range(1, 13):
        r = run_drift_check(w)
        max_psi = max(v["psi"] for v in r["features"].values())
        flag = "YES" if r["drift_detected"] else "no"
        print(f"  {w:>4d} | {flag:>8} | {r['n_drifted_features']:>8d} | {max_psi:>8.4f}")
        all_reports.append(r)
    summary = [{"week": r["week"], "drift_detected": bool(r["drift_detected"]),
                "n_drifted": r["n_drifted_features"],
                "max_psi": round(max(v["psi"] for v in r["features"].values()), 4)}
               for r in all_reports]
    (ROOT / "results/registry/drift_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDrift summary saved.")
