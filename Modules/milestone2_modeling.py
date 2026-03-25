"""
=============================================================
MILESTONE 2: Feature Extraction and Modeling (Weeks 3-4)
=============================================================
- Extract statistical time-series features (TSFresh-style)
- Facebook Prophet trend modeling for HR / Steps / SpO2
- KMeans & DBSCAN clustering for behavioral pattern detection
- PCA visualization of clusters
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Feature Extraction (TSFresh-style manual)
# ─────────────────────────────────────────────
def extract_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Extract statistical features from rolling windows.
    Mimics TSFresh: mean, std, min, max, kurtosis, skewness, energy.
    """
    print("\n[Feature Extraction] Computing rolling statistical features...")

    numeric_cols = ["heart_rate_bpm", "steps", "spo2_pct", "calories_burned"]
    cols = [c for c in numeric_cols if c in df.columns]

    features = {}
    for col in cols:
        series = df[col]
        roll = series.rolling(window=window, min_periods=1)
        features[f"{col}_mean"]     = roll.mean()
        features[f"{col}_std"]      = roll.std().fillna(0)
        features[f"{col}_min"]      = roll.min()
        features[f"{col}_max"]      = roll.max()
        features[f"{col}_range"]    = features[f"{col}_max"] - features[f"{col}_min"]
        features[f"{col}_skew"]     = series.rolling(window=window, min_periods=3).skew().fillna(0)
        features[f"{col}_kurt"]     = series.rolling(window=window, min_periods=4).kurt().fillna(0)
        features[f"{col}_energy"]   = roll.apply(lambda x: np.sum(x**2), raw=True)

    feat_df = pd.DataFrame(features, index=df.index)

    # Add time features
    feat_df["hour"]      = df.index.hour
    feat_df["day_of_week"] = df.index.dayofweek
    feat_df["is_night"]  = ((feat_df["hour"] < 6) | (feat_df["hour"] >= 23)).astype(int)

    print(f"   ✅ Feature matrix shape: {feat_df.shape}")
    return feat_df


# ─────────────────────────────────────────────
# 2. Prophet Trend Modeling
# ─────────────────────────────────────────────
def prophet_model(df: pd.DataFrame, metric: str = "heart_rate_bpm", periods: int = 48) -> dict:
    """Fit Facebook Prophet on a metric and forecast next `periods` intervals."""
    print(f"\n[Prophet] Modeling '{metric}'...")

    # Prophet expects 'ds' and 'y' columns
    prophet_df = df[[metric]].resample("1h").mean().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    prophet_df.dropna(inplace=True)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq="h")
    forecast = model.predict(future)

    # Compute residuals for anomaly detection
    merged = prophet_df.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
    merged["residual"] = merged["y"] - merged["yhat"]
    merged["is_anomaly_prophet"] = (
        (merged["y"] > merged["yhat_upper"]) | (merged["y"] < merged["yhat_lower"])
    ).astype(int)

    anomaly_pct = merged["is_anomaly_prophet"].mean() * 100
    print(f"   ✅ Prophet anomalies detected: {merged['is_anomaly_prophet'].sum()} ({anomaly_pct:.1f}%)")
    return {"model": model, "forecast": forecast, "residuals": merged}


def plot_prophet(result: dict, metric: str, save_path: str):
    """Plot Prophet forecast with confidence interval and actuals."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    forecast = result["forecast"]
    residuals = result["residuals"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Prophet Trend Model — {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")

    # Forecast plot
    ax = axes[0]
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.25, color="#3498DB", label="95% Confidence Interval")
    ax.plot(forecast["ds"], forecast["yhat"], color="#2980B9", linewidth=1.5, label="Forecast")
    ax.scatter(residuals["ds"], residuals["y"], s=4, color="#E74C3C", alpha=0.6, label="Actual")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Residuals
    ax2 = axes[1]
    ax2.bar(residuals["ds"], residuals["residual"],
            color=["#E74C3C" if x else "#2ECC71" for x in residuals["is_anomaly_prophet"]],
            alpha=0.7, width=0.04)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date")
    ax2.set_title("Residuals (Red = outside confidence band)", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Prophet plot → {save_path}")


# ─────────────────────────────────────────────
# 3. Clustering (KMeans + DBSCAN)
# ─────────────────────────────────────────────
def run_clustering(feat_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Apply KMeans and DBSCAN clustering on feature matrix."""
    print(f"\n[Clustering] Running KMeans (k={n_clusters}) and DBSCAN...")

    cluster_cols = [c for c in feat_df.columns
                    if any(c.startswith(m) for m in ["heart_rate", "steps", "spo2"])]
    X = feat_df[cluster_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df = feat_df.copy()
    feat_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    feat_df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
    feat_df["is_outlier_dbscan"] = (feat_df["dbscan_cluster"] == -1).astype(int)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    feat_df["pca1"] = pca_result[:, 0]
    feat_df["pca2"] = pca_result[:, 1]

    print(f"   ✅ KMeans clusters: {sorted(feat_df['kmeans_cluster'].unique())}")
    print(f"   ✅ DBSCAN outliers: {feat_df['is_outlier_dbscan'].sum()}")
    return feat_df


def plot_clusters(feat_df: pd.DataFrame, save_path: str):
    """Visualize clusters in PCA space."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Milestone 2: Clustering Results (PCA Projection)", fontsize=14, fontweight="bold")

    # KMeans clusters
    colors_km = plt.cm.Set1(np.linspace(0, 1, feat_df["kmeans_cluster"].nunique()))
    for i, (cluster, group) in enumerate(feat_df.groupby("kmeans_cluster")):
        axes[0].scatter(group["pca1"], group["pca2"], s=4, alpha=0.5,
                        color=colors_km[i], label=f"Cluster {cluster}")
    axes[0].set_title("KMeans Clusters", fontsize=12)
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    axes[0].legend(fontsize=8, markerscale=3)
    axes[0].grid(alpha=0.3)

    # DBSCAN outliers
    normal = feat_df[feat_df["is_outlier_dbscan"] == 0]
    outliers = feat_df[feat_df["is_outlier_dbscan"] == 1]
    axes[1].scatter(normal["pca1"], normal["pca2"], s=4, alpha=0.4,
                    color="#3498DB", label=f"Normal ({len(normal)})")
    axes[1].scatter(outliers["pca1"], outliers["pca2"], s=20, alpha=0.8,
                    color="#E74C3C", marker="x", label=f"Outliers ({len(outliers)})")
    axes[1].set_title("DBSCAN Outlier Detection", fontsize=12)
    axes[1].set_xlabel("PCA Component 1")
    axes[1].set_ylabel("PCA Component 2")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Cluster plot → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone2(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  MILESTONE 2: Feature Extraction & Modeling")
    print("=" * 55)

    feat_df = extract_features(df)
    feat_df.to_csv("outputs/feature_matrix.csv")
    print("   💾 Feature matrix → outputs/feature_matrix.csv")

    # Prophet for 3 metrics
    prophet_results = {}
    for metric in ["heart_rate_bpm", "steps", "spo2_pct"]:
        result = prophet_model(df, metric=metric)
        prophet_results[metric] = result
        plot_prophet(result, metric, save_path=f"outputs/prophet_{metric}.png")

    # Clustering
    feat_clustered = run_clustering(feat_df)
    feat_clustered.to_csv("outputs/feature_clustered.csv")
    plot_clusters(feat_clustered, save_path="outputs/milestone2_clusters.png")

    print("\n✅ Milestone 2 Complete!")
    return {"features": feat_clustered, "prophet": prophet_results}


if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
    results = run_milestone2(df)
