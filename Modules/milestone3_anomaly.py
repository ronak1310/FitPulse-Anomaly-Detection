"""
=============================================================
MILESTONE 3: Anomaly Detection & Visualization (Weeks 5-6)
=============================================================
- Rule-based threshold detection (HR, SpO2, steps)
- Prophet residual-based anomaly detection
- DBSCAN clustering outlier flagging
- Combined anomaly scoring
- Interactive annotated visualizations
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Rule-Based Threshold Detection
# ─────────────────────────────────────────────
THRESHOLDS = {
    "heart_rate_bpm": {
        "high": 120,       # Tachycardia threshold
        "low": 45,         # Bradycardia threshold
        "sleep_high": 90,  # HR too high during sleep
    },
    "spo2_pct": {
        "low": 94,         # Low blood oxygen
    },
    "steps": {
        "sleep_active": 50,  # Steps during supposed sleep
    },
}


def rule_based_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Flag anomalies based on medical/fitness thresholds."""
    print("\n[Rule-Based] Applying threshold detection...")
    df = df.copy()

    df["rule_tachycardia"]    = (df["heart_rate_bpm"] > THRESHOLDS["heart_rate_bpm"]["high"]).astype(int)
    df["rule_bradycardia"]    = (df["heart_rate_bpm"] < THRESHOLDS["heart_rate_bpm"]["low"]).astype(int)
    df["rule_low_spo2"]       = (df["spo2_pct"] < THRESHOLDS["spo2_pct"]["low"]).astype(int)

    # Steps during sleep
    if "sleeping" in df.columns:
        df["rule_sleep_steps"] = (
            (df["sleeping"] == 1) & (df["steps"] > THRESHOLDS["steps"]["sleep_active"])
        ).astype(int)
        df["rule_sleep_hr"] = (
            (df["sleeping"] == 1) & (df["heart_rate_bpm"] > THRESHOLDS["heart_rate_bpm"]["sleep_high"])
        ).astype(int)
    else:
        df["rule_sleep_steps"] = 0
        df["rule_sleep_hr"]    = 0

    rule_cols = ["rule_tachycardia", "rule_bradycardia", "rule_low_spo2", "rule_sleep_steps", "rule_sleep_hr"]
    df["rule_anomaly"] = df[rule_cols].any(axis=1).astype(int)

    total = df["rule_anomaly"].sum()
    print(f"   ✅ Rule-based anomalies flagged: {total} ({total/len(df)*100:.2f}%)")
    for col in rule_cols:
        print(f"      {col}: {df[col].sum()}")

    return df


# ─────────────────────────────────────────────
# 2. Residual-Based (Prophet) Anomaly Detection
# ─────────────────────────────────────────────
def add_prophet_anomalies(df: pd.DataFrame, prophet_results: dict) -> pd.DataFrame:
    """Merge Prophet-detected anomalies back into the main dataframe."""
    print("\n[Prophet Residuals] Merging anomaly flags...")
    df = df.copy()
    df["prophet_anomaly"] = 0

    for metric, result in prophet_results.items():
        residuals = result["residuals"]
        residuals_indexed = residuals.set_index("ds")
        residuals_resampled = residuals_indexed["is_anomaly_prophet"].resample("5min").max().ffill()
        df["prophet_anomaly"] = df["prophet_anomaly"].add(
            residuals_resampled.reindex(df.index, fill_value=0), fill_value=0
        )

    df["prophet_anomaly"] = (df["prophet_anomaly"] > 0).astype(int)
    print(f"   ✅ Prophet anomalies merged: {df['prophet_anomaly'].sum()}")
    return df


# ─────────────────────────────────────────────
# 3. DBSCAN Outlier Flags
# ─────────────────────────────────────────────
def add_cluster_anomalies(df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add DBSCAN outlier flags from feature clustering."""
    print("\n[DBSCAN] Merging outlier cluster flags...")
    df = df.copy()

    if "is_outlier_dbscan" in feat_df.columns:
        df["dbscan_anomaly"] = feat_df["is_outlier_dbscan"].reindex(df.index, fill_value=0)
        print(f"   ✅ DBSCAN outliers merged: {df['dbscan_anomaly'].sum()}")
    else:
        df["dbscan_anomaly"] = 0
    return df


# ─────────────────────────────────────────────
# 4. Combined Anomaly Score
# ─────────────────────────────────────────────
def compute_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine signals into a single anomaly score (0–3)."""
    df = df.copy()
    score_cols = ["rule_anomaly", "prophet_anomaly", "dbscan_anomaly"]
    existing = [c for c in score_cols if c in df.columns]
    df["anomaly_score"] = df[existing].sum(axis=1)
    df["final_anomaly"] = (df["anomaly_score"] >= 1).astype(int)

    # Severity label
    def severity(score):
        if score == 0: return "normal"
        elif score == 1: return "low"
        elif score == 2: return "medium"
        else: return "high"

    df["severity"] = df["anomaly_score"].apply(severity)

    total = df["final_anomaly"].sum()
    print(f"\n✅ Combined anomaly score computed.")
    print(f"   Total flagged: {total} ({total/len(df)*100:.2f}%)")
    print(f"   Severity breakdown:\n{df['severity'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────
SEVERITY_COLORS = {"low": "#F39C12", "medium": "#E67E22", "high": "#C0392B"}


def plot_heart_rate_anomalies(df: pd.DataFrame, days: int = 7,
                               save_path: str = "outputs/milestone3_heartrate.png"):
    """Heart rate chart with color-coded anomaly highlights."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = df.iloc[:days * 288]  # 5-min intervals

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(sample.index, sample["heart_rate_bpm"], color="#95A5A6", linewidth=0.6, alpha=0.8, label="Heart Rate")
    ax.axhline(THRESHOLDS["heart_rate_bpm"]["high"], color="#E74C3C", linestyle="--", linewidth=1, label="Tachy threshold (120)")
    ax.axhline(THRESHOLDS["heart_rate_bpm"]["low"],  color="#3498DB", linestyle="--", linewidth=1, label="Brady threshold (45)")

    # Shade anomalies by severity
    for sev, color in SEVERITY_COLORS.items():
        mask = sample["severity"] == sev
        if mask.any():
            ax.scatter(sample.index[mask], sample["heart_rate_bpm"][mask],
                       color=color, s=25, zorder=5, label=f"Anomaly ({sev})", alpha=0.9)

    ax.set_title(f"Heart Rate — Anomaly Detection (First {days} Days)", fontsize=14, fontweight="bold")
    ax.set_ylabel("BPM", fontsize=12)
    ax.set_xlabel("Date/Time")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Heart rate anomaly plot → {save_path}")


def plot_sleep_anomalies(df: pd.DataFrame, days: int = 7,
                          save_path: str = "outputs/milestone3_sleep.png"):
    """Sleep pattern with HR + step anomalies during sleep."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if "sleeping" not in df.columns:
        print("   ⚠️  No 'sleeping' column found; skipping sleep plot.")
        return

    sample = df.iloc[:days * 288]
    sleep_data = sample[sample["sleeping"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Sleep Pattern Analysis — Anomaly Detection (First {days} Days)",
                 fontsize=14, fontweight="bold")

    # Sleeping HR
    axes[0].plot(sleep_data.index, sleep_data["heart_rate_bpm"], color="#9B59B6", linewidth=0.7)
    anomalies_sleep = sleep_data[sleep_data["rule_sleep_hr"] == 1]
    axes[0].scatter(anomalies_sleep.index, anomalies_sleep["heart_rate_bpm"],
                    color="#E74C3C", s=30, zorder=5, label=f"Elevated HR during sleep ({len(anomalies_sleep)})")
    axes[0].axhline(90, color="orange", linestyle="--", linewidth=1, label="Sleep HR threshold")
    axes[0].set_ylabel("Heart Rate (bpm)")
    axes[0].set_title("Heart Rate During Sleep")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Steps during sleep
    axes[1].bar(sleep_data.index, sleep_data["steps"], color="#2ECC71", alpha=0.6, width=0.003)
    step_anomalies = sleep_data[sleep_data["rule_sleep_steps"] == 1]
    axes[1].bar(step_anomalies.index, step_anomalies["steps"], color="#E74C3C", alpha=0.8, width=0.003,
                label=f"Sleep disturbance - steps ({len(step_anomalies)})")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps During Sleep (should be ~0)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Sleep anomaly plot → {save_path}")


def plot_steps_and_spo2(df: pd.DataFrame, days: int = 7,
                         save_path: str = "outputs/milestone3_steps_spo2.png"):
    """Step count + SpO2 with anomalies highlighted."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = df.iloc[:days * 288]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Step Count & SpO2 — Anomaly Detection (First {days} Days)",
                 fontsize=14, fontweight="bold")

    # Steps
    axes[0].bar(sample.index, sample["steps"], color="#3498DB", alpha=0.6, width=0.003, label="Steps")
    sev_mask = sample["severity"].isin(["medium", "high"])
    if sev_mask.any():
        axes[0].bar(sample.index[sev_mask], sample["steps"][sev_mask],
                    color="#E74C3C", alpha=0.8, width=0.003, label="High-severity anomaly")
    axes[0].set_ylabel("Steps (per 5 min)")
    axes[0].set_title("Step Count Trend")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # SpO2
    axes[1].plot(sample.index, sample["spo2_pct"], color="#1ABC9C", linewidth=0.7, label="SpO2")
    axes[1].axhline(THRESHOLDS["spo2_pct"]["low"], color="#E74C3C", linestyle="--", linewidth=1.2,
                    label=f"Low SpO2 threshold ({THRESHOLDS['spo2_pct']['low']}%)")
    low_spo2 = sample[sample["rule_low_spo2"] == 1]
    axes[1].scatter(low_spo2.index, low_spo2["spo2_pct"], color="#E74C3C", s=40, zorder=5,
                    label=f"Low SpO2 alert ({len(low_spo2)})")
    axes[1].set_ylabel("SpO2 (%)")
    axes[1].set_ylim(85, 101)
    axes[1].set_title("Blood Oxygen Saturation")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Steps & SpO2 plot → {save_path}")


def plot_anomaly_summary(df: pd.DataFrame, save_path: str = "outputs/milestone3_summary.png"):
    """Daily anomaly summary heatmap-style bar chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date
    daily = df_copy.groupby("date").agg(
        total=("final_anomaly", "count"),
        anomalies=("final_anomaly", "sum"),
        high_sev=("severity", lambda x: (x == "high").sum()),
    ).reset_index()
    daily["pct"] = daily["anomalies"] / daily["total"] * 100

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(range(len(daily)), daily["pct"],
                  color=["#E74C3C" if x > 5 else "#F39C12" if x > 2 else "#2ECC71" for x in daily["pct"]],
                  alpha=0.8, edgecolor="white")

    ax.set_xticks(range(0, len(daily), 5))
    ax.set_xticklabels([str(daily["date"].iloc[i]) for i in range(0, len(daily), 5)], rotation=30, ha="right")
    ax.set_ylabel("% Anomalous Readings")
    ax.set_xlabel("Date")
    ax.set_title("Daily Anomaly Rate (Green < 2% | Orange < 5% | Red ≥ 5%)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    green_p  = mpatches.Patch(color="#2ECC71", label="Normal (< 2%)")
    orange_p = mpatches.Patch(color="#F39C12", label="Warning (2–5%)")
    red_p    = mpatches.Patch(color="#E74C3C", label="Alert (≥ 5%)")
    ax.legend(handles=[green_p, orange_p, red_p], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Anomaly summary plot → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone3(df: pd.DataFrame, feat_df: pd.DataFrame = None,
                   prophet_results: dict = None) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  MILESTONE 3: Anomaly Detection & Visualization")
    print("=" * 55)

    df = rule_based_detection(df)

    if prophet_results:
        df = add_prophet_anomalies(df, prophet_results)
    else:
        df["prophet_anomaly"] = 0

    if feat_df is not None:
        df = add_cluster_anomalies(df, feat_df)
    else:
        df["dbscan_anomaly"] = 0

    df = compute_anomaly_score(df)
    df.to_csv("outputs/anomaly_results.csv")
    print("\n💾 Anomaly results → outputs/anomaly_results.csv")

    # Plots
    plot_heart_rate_anomalies(df)
    plot_sleep_anomalies(df)
    plot_steps_and_spo2(df)
    plot_anomaly_summary(df)

    print("\n✅ Milestone 3 Complete!")
    return df


if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
    result_df = run_milestone3(df)
