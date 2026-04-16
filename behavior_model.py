#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GroupShuffleSplit

TARGET_COL = "avg_Tokens_per_s"

########################################################
# 1. Feature engineering: runtime behavior + config
########################################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavior + config features from your columns.
    Uses avg_GPU{i}_* metrics only; ignores avg_avg_* layers.
    """

    # GPUs are 0..3 from your schema
    gpu_ids = [0, 1, 2, 3]

    # Per-GPU base column templates
    clock_cols = [f"avg_GPU{i}_GfxClock_MHz" for i in gpu_ids]
    mem_clock_cols = [f"avg_GPU{i}_MemClock_MHz" for i in gpu_ids]
    util_cols = [f"avg_GPU{i}_Util_%" for i in gpu_ids]
    mem_util_cols = [f"avg_GPU{i}_MemUtil_%" for i in gpu_ids]
    pwr_cols = [f"avg_GPU{i}_Power_W" for i in gpu_ids]
    pcie_rx_cols = [f"avg_GPU{i}_PCIe_RX_MBps" for i in gpu_ids]
    pcie_tx_cols = [f"avg_GPU{i}_PCIe_TX_MBps" for i in gpu_ids]

    # Basic sanity check
    for cols in [clock_cols, mem_clock_cols, util_cols, mem_util_cols,
                 pwr_cols, pcie_rx_cols, pcie_tx_cols]:
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' missing from dataframe.")

    # Aggregate behavior features
    df_feat = df.copy()

    df_feat["mean_clock_MHz"] = df_feat[clock_cols].mean(axis=1)
    df_feat["mean_mem_clock_MHz"] = df_feat[mem_clock_cols].mean(axis=1)
    df_feat["mean_util_pct"] = df_feat[util_cols].mean(axis=1)
    df_feat["mean_mem_util_pct"] = df_feat[mem_util_cols].mean(axis=1)

    df_feat["total_power_W"] = df_feat[pwr_cols].sum(axis=1)
    df_feat["total_pcie_MBps"] = (
        df_feat[pcie_rx_cols].sum(axis=1) +
        df_feat[pcie_tx_cols].sum(axis=1)
    )

    # Some variation indicators
    df_feat["util_std"] = df_feat[util_cols].std(axis=1)
    df_feat["pcie_std"] = (
        df_feat[pcie_rx_cols + pcie_tx_cols].std(axis=1)
    )

    # Ensure config knobs are numeric
    df_feat["tp"] = df_feat["tp"].astype(float)
    df_feat["batch"] = df_feat["batch"].astype(float)
    df_feat["cap_w"] = df_feat["cap_w"].astype(float)

    # Optionally log-transform some features to compress dynamic range
    df_feat["log_total_pcie_MBps"] = np.log1p(df_feat["total_pcie_MBps"])
    df_feat["log_total_power_W"] = np.log1p(df_feat["total_power_W"])
    df_feat["log_mean_util_pct"] = np.log1p(df_feat["mean_util_pct"])

    return df_feat


def get_feature_columns() -> List[str]:
    """
    Features we feed into the global RF model.
    You can tweak this list as needed.
    """
    return [
        "tp",
        "batch",
        "cap_w",
        "mean_clock_MHz",
        "mean_mem_clock_MHz",
        "mean_util_pct",
        "mean_mem_util_pct",
        "total_power_W",
        "total_pcie_MBps",
        "util_std",
        "pcie_std",
        "log_total_pcie_MBps",
        "log_total_power_W",
        "log_mean_util_pct",
    ]

########################################################
# 2. Training & evaluation with group split by model
########################################################

def train_global_model(df_feat: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Train a single global RF model using runtime behavior + config.
    Uses GroupShuffleSplit grouped by 'model' to simulate unseen benchmark generalization.
    Returns:
      - trained model
      - per-model metrics on the test split
    """

    feature_cols = get_feature_columns()

    for c in feature_cols + [TARGET_COL, "model"]:
        if c not in df_feat.columns:
            raise ValueError(f"Required column '{c}' not in dataframe.")

    X = df_feat[feature_cols].values
    y = df_feat[TARGET_COL].values
    groups = df_feat["model"].values

    gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model_train = df_feat.iloc[train_idx]["model"].values
    model_test = df_feat.iloc[test_idx]["model"].values

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_test = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print("=== Global model evaluation (test split grouped by model) ===")
    print(f"Test samples: {len(y_test)}")
    print(f"R^2:  {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print("============================================================\n")

    # Per-model metrics on test set
    metrics_rows = []
    test_df = pd.DataFrame({
        "model": model_test,
        "y_true": y_test,
        "y_pred": y_pred_test
    })

    for m in sorted(test_df["model"].unique()):
        sub = test_df[test_df["model"] == m]
        if len(sub) < 3:
            continue
        r2_m = r2_score(sub["y_true"], sub["y_pred"])
        mape_m = mean_absolute_percentage_error(sub["y_true"], sub["y_pred"])
        metrics_rows.append({"model": m.strip(), "R2": r2_m, "MAPE": mape_m, "n": len(sub)})

    metrics_df = pd.DataFrame(metrics_rows)
    print("=== Per-model metrics on test set ===")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    else:
        print("[No per-model metrics; too few samples per model in test split]")
    print("=====================================\n")

    return rf, metrics_df, (X_test, y_test, model_test)

########################################################
# 3. Plotting helpers
########################################################

def plot_pred_vs_true(X_test, y_test, model_test, rf, feature_cols, out_path: str):
    y_pred = rf.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("True tokens/s")
    plt.ylabel("Predicted tokens/s")
    plt.title("Predicted vs True tokens/s (test set)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved global pred-vs-true plot → {out_path}")


def plot_r2_bar(metrics_df: pd.DataFrame, out_path: str):
    if metrics_df.empty:
        print("[WARN] No per-model metrics to plot R² bar.")
        return
    df_plot = metrics_df.sort_values("R2", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df_plot["model"], df_plot["R2"])
    plt.ylim(0, 1.05)
    plt.ylabel("R² (test)")
    plt.xlabel("Model")
    plt.title("Per-model R² on test split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved per-model R² bar plot → {out_path}")


def plot_feature_importance(rf: RandomForestRegressor, feature_cols: List[str], out_path: str):
    importances = rf.feature_importances_
    idx = np.argsort(importances)

    plt.figure(figsize=(7, 5))
    plt.barh([feature_cols[i] for i in idx], importances[idx])
    plt.xlabel("Importance")
    plt.title("Global feature importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot → {out_path}")

########################################################
# 4. Config recommendation for a benchmark
########################################################

def recommend_best_config_for_model(
    df_feat: pd.DataFrame,
    rf: RandomForestRegressor,
    feature_cols: List[str],
    model_name: str,
    power_cap: float,
) -> Tuple[pd.Series, float]:
    """
    Given the full dataframe (with behavior features) and a trained RF,
    pick the best config for a given benchmark (model_name) under cap_w <= power_cap.
    Uses RF to smooth noise and predict tokens/s.
    """
    sub = df_feat[(df_feat["model"] == model_name) & (df_feat["cap_w"] <= power_cap)]
    if sub.empty:
        raise ValueError(f"No rows for model='{model_name}' with cap_w <= {power_cap}")
    X_sub = sub[feature_cols].values
    y_pred = rf.predict(X_sub)
    best_idx = np.argmax(y_pred)
    best_row = sub.iloc[best_idx].copy()
    best_perf = float(y_pred[best_idx])
    return best_row, best_perf

########################################################
# 5. Main
########################################################

def main():
    parser = argparse.ArgumentParser(
        description="Global behavior+config model with R² plots and feature importance."
    )
    parser.add_argument("csv_path", type=str,
                        help="Path to gpu_timeseries_summaries.csv")
    parser.add_argument("--out-dir", type=str,
                        default="behavior_model_plots",
                        help="Directory to save plots and metrics.")
    parser.add_argument("--sample-power-cap", type=float,
                        default=300.0,
                        help="Power cap (W) for example recommendations.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)

    # Engineer behavior + config features
    df_feat = engineer_features(df)

    # Train global RF with group split by model
    rf, metrics_df, test_pack = train_global_model(df_feat)
    X_test, y_test, model_test = test_pack
    feature_cols = get_feature_columns()

    # Save metrics CSV
    metrics_path = os.path.join(args.out_dir, "per_model_metrics_test.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved per-model metrics (test) → {metrics_path}")

    # Plots
    plot_pred_vs_true(
        X_test, y_test, model_test, rf, feature_cols,
        out_path=os.path.join(args.out_dir, "pred_vs_true_test.png"),
    )
    plot_r2_bar(
        metrics_df,
        out_path=os.path.join(args.out_dir, "per_model_R2_bar.png"),
    )
    plot_feature_importance(
        rf, feature_cols,
        out_path=os.path.join(args.out_dir, "feature_importance.png"),
    )

    # Example: recommend best config per model at specified power cap
    print(f"\n=== Example best configs at cap_w <= {args.sample_power_cap} W (using RF predictions) ===")
    for m in sorted(df_feat["model"].unique()):
        try:
            best_row, perf = recommend_best_config_for_model(
                df_feat, rf, feature_cols, m, args.sample_power_cap
            )
            print(
                f"{m:15s} → tp={int(best_row['tp'])}, "
                f"batch={int(best_row['batch'])}, "
                f"cap_w={best_row['cap_w']:.0f}, "
                f"pred_tokens/s={perf:.1f}"
            )
        except ValueError as e:
            print(f"{m:15s} → {e}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
