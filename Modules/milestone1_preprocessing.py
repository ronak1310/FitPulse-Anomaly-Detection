"""
=============================================================
MILESTONE 1: Data Collection and Preprocessing (Weeks 1-2)
=============================================================
- Import health data (CSV/JSON) from fitness trackers
- Clean and normalize timestamps
- Handle missing values via interpolation
- Resample to consistent time intervals
=============================================================
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ─────────────────────────────────────────────
# 1. Data Ingestion
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV or JSON fitness data into a DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        with open(filepath) as f:
            raw = json.load(f)
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
    else:
        raise ValueError(f"Unsupported format: {ext}. Use CSV or JSON.")
    print(f"✅ Loaded {len(df)} records from {filepath}")
    return df


# ─────────────────────────────────────────────
# 2. Timestamp Normalization
# ─────────────────────────────────────────────
def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Parse and sort timestamps; set as DatetimeIndex."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    df = df.sort_values(ts_col).reset_index(drop=True)
    df.set_index(ts_col, inplace=True)
    print(f"✅ Timestamps normalized. Range: {df.index.min()} → {df.index.max()}")
    return df


# ─────────────────────────────────────────────
# 3. Missing Value Handling
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Inject some synthetic NaNs to demonstrate cleaning
    - Interpolate numeric columns
    - Forward-fill categorical columns
    """
    df = df.copy()

    # Simulate ~3% missing values for demo purposes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    before_nulls = df.isnull().sum().sum()
    print(f"   Missing values before cleaning : {before_nulls}")

    # Interpolate numeric columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")

    # Forward-fill remaining
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    after_nulls = df.isnull().sum().sum()
    print(f"✅ Missing values after cleaning  : {after_nulls}")
    return df


# ─────────────────────────────────────────────
# 4. Resampling to Consistent Intervals
# ─────────────────────────────────────────────
def resample_data(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample data to a fixed frequency using aggregation rules."""
    agg_rules = {
        "heart_rate_bpm": "mean",
        "steps": "sum",
        "sleeping": "max",
        "spo2_pct": "mean",
        "calories_burned": "sum",
        "is_anomaly": "max",
    }
    # Keep only columns that exist in the df
    existing_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    df_resampled = df[list(existing_rules.keys())].resample(freq).agg(existing_rules)
    df_resampled.ffill(inplace=True)

    # Round for readability
    for col in ["heart_rate_bpm", "spo2_pct", "calories_burned"]:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].round(2)

    print(f"✅ Data resampled to '{freq}' frequency. Shape: {df_resampled.shape}")
    return df_resampled


# ─────────────────────────────────────────────
# 5. Visualization - Cleaned Dataset Preview
# ─────────────────────────────────────────────
def plot_preprocessed_data(df: pd.DataFrame, save_path: str = "outputs/milestone1_preview.png"):
    """Plot cleaned heart rate, steps, and SpO2 for first 3 days."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = df.iloc[:864]  # 3 days at 5-min intervals

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Milestone 1: Cleaned & Normalized Fitness Data (First 3 Days)",
                 fontsize=15, fontweight="bold", y=1.01)

    # Heart Rate
    axes[0].plot(sample.index, sample["heart_rate_bpm"], color="#E74C3C", linewidth=0.8, alpha=0.85)
    axes[0].axhline(100, color="orange", linestyle="--", linewidth=1, label="Threshold 100 bpm")
    axes[0].set_ylabel("Heart Rate (bpm)", fontsize=11)
    axes[0].set_title("Heart Rate Over Time", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Steps
    axes[1].bar(sample.index, sample["steps"], color="#2ECC71", alpha=0.7, width=0.003)
    axes[1].set_ylabel("Steps (per 5 min)", fontsize=11)
    axes[1].set_title("Step Count", fontsize=12)
    axes[1].grid(alpha=0.3)

    # SpO2
    axes[2].plot(sample.index, sample["spo2_pct"], color="#3498DB", linewidth=0.8)
    axes[2].axhline(95, color="red", linestyle="--", linewidth=1, label="Low SpO2 threshold (95%)")
    axes[2].set_ylabel("SpO2 (%)", fontsize=11)
    axes[2].set_title("Blood Oxygen Saturation", fontsize=12)
    axes[2].set_ylim(85, 101)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone1(csv_path: str = r"fitness_data_raw.csv") -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  MILESTONE 1: Data Collection & Preprocessing")
    print("=" * 55)

    df_raw = load_data(csv_path)
    df_norm = normalize_timestamps(df_raw)
    df_clean = handle_missing_values(df_norm)
    df_resampled = resample_data(df_clean, freq="5min")

    # Save cleaned data
    os.makedirs("outputs", exist_ok=True)
    df_resampled.to_csv("outputs/cleaned_data.csv")
    print(f"💾 Cleaned data saved → outputs/cleaned_data.csv")

    plot_preprocessed_data(df_resampled)

    print("\n✅ Milestone 1 Complete!")
    print(f"   Output records : {len(df_resampled)}")
    print(f"   Columns        : {list(df_resampled.columns)}")
    return df_resampled


if __name__ == "__main__":
    df = run_milestone1()
    print(df.head(10).to_string())
