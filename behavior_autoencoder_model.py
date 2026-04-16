#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

TARGET_COL = "avg_Tokens_per_s"
DEVICE = torch.device("cpu")  # CPU is fine


########################################################
# 1. Feature engineering: runtime behavior + config
########################################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavior + config features from your columns.
    Uses avg_GPU{i}_* metrics only; ignores avg_avg_* layers.
    Includes normalized behavior-intensity features (no tokens/s).
    """

    gpu_ids = [0, 1, 2, 3]

    clock_cols = [f"avg_GPU{i}_GfxClock_MHz" for i in gpu_ids]
    mem_clock_cols = [f"avg_GPU{i}_MemClock_MHz" for i in gpu_ids]
    util_cols = [f"avg_GPU{i}_Util_%" for i in gpu_ids]
    mem_util_cols = [f"avg_GPU{i}_MemUtil_%" for i in gpu_ids]
    pwr_cols = [f"avg_GPU{i}_Power_W" for i in gpu_ids]
    pcie_rx_cols = [f"avg_GPU{i}_PCIe_RX_MBps" for i in gpu_ids]
    pcie_tx_cols = [f"avg_GPU{i}_PCIe_TX_MBps" for i in gpu_ids]

    # basic checks
    for cols in [clock_cols, mem_clock_cols, util_cols, mem_util_cols,
                 pwr_cols, pcie_rx_cols, pcie_tx_cols]:
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' missing from dataframe.")

    df_feat = df.copy()

    # Base aggregate behavior features
    df_feat["mean_clock_MHz"] = df_feat[clock_cols].mean(axis=1)
    df_feat["mean_mem_clock_MHz"] = df_feat[mem_clock_cols].mean(axis=1)
    df_feat["mean_util_pct"] = df_feat[util_cols].mean(axis=1)
    df_feat["mean_mem_util_pct"] = df_feat[mem_util_cols].mean(axis=1)

    df_feat["total_power_W"] = df_feat[pwr_cols].sum(axis=1)
    df_feat["total_pcie_MBps"] = (
        df_feat[pcie_rx_cols].sum(axis=1) +
        df_feat[pcie_tx_cols].sum(axis=1)
    )

    df_feat["util_std"] = df_feat[util_cols].std(axis=1)
    df_feat["pcie_std"] = df_feat[pcie_rx_cols + pcie_tx_cols].std(axis=1)

    # Config knobs
    df_feat["tp"] = df_feat["tp"].astype(float)
    df_feat["batch"] = df_feat["batch"].astype(float)
    df_feat["cap_w"] = df_feat["cap_w"].astype(float)

    # Normalized / intensity features (no tokens/s used)
    eps = 1e-6

    # Compute intensity proxy
    df_feat["compute_intensity"] = df_feat["mean_clock_MHz"] * df_feat["mean_util_pct"]

    # Memory intensity proxy
    df_feat["mem_intensity"] = df_feat["mean_mem_clock_MHz"] * df_feat["mean_mem_util_pct"]

    # Communication per power / util
    df_feat["comm_per_power"] = df_feat["total_pcie_MBps"] / (df_feat["total_power_W"] + eps)
    df_feat["comm_per_util"] = df_feat["total_pcie_MBps"] / (df_feat["mean_util_pct"] + eps)

    # Per-GPU power
    df_feat["power_per_gpu"] = df_feat["total_power_W"] / 4.0

    # Balance metrics
    df_feat["util_balance"] = df_feat["util_std"] / (df_feat["mean_util_pct"] + eps)
    df_feat["pcie_balance"] = df_feat["pcie_std"] / (df_feat["total_pcie_MBps"] + eps)

    return df_feat


def get_behavior_columns() -> List[str]:
    """
    Runtime behavior features that go into the autoencoder.
    Includes base + normalized/intensity features.
    """
    return [
        # base aggregates
        "mean_clock_MHz",
        "mean_mem_clock_MHz",
        "mean_util_pct",
        "mean_mem_util_pct",
        "total_power_W",
        "total_pcie_MBps",
        "util_std",
        "pcie_std",
        # normalized / intensity
        "compute_intensity",
        "mem_intensity",
        "comm_per_power",
        "comm_per_util",
        "power_per_gpu",
        "util_balance",
        "pcie_balance",
    ]


def get_config_columns() -> List[str]:
    """Config knobs for the RF model."""
    return ["tp", "batch", "cap_w"]


########################################################
# 2. Autoencoder definition & training
########################################################

class BehaviorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        hidden = max(16, latent_dim * 4)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(
    X_beh: np.ndarray,
    latent_dim: int = 4,
    epochs: int = 400,
    lr: float = 1e-3,
) -> Tuple[BehaviorAutoencoder, StandardScaler, np.ndarray]:
    """
    Train an autoencoder on behavior features.
    Returns:
      - trained AE (on CPU)
      - fitted StandardScaler
      - latent_repr (numpy array, shape [N, latent_dim])
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_beh.astype(np.float32))

    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)

    input_dim = X_scaled.shape[1]
    model = BehaviorAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(X_t)
        loss = loss_fn(recon, X_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[AE] Epoch {epoch+1:3d}/{epochs}, recon loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        _, latent = model(X_t)
    latent_np = latent.cpu().numpy()

    return model, scaler, latent_np


########################################################
# 3. RF training & evaluation with group split by filename
########################################################

def train_rf_with_latent_group_by_file(
    df_feat: pd.DataFrame,
    latent: np.ndarray,
    latent_dim: int,
) -> Tuple[RandomForestRegressor, pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Train a global RF model with features: [latent_z0..zK, tp, batch, cap_w].
    Grouped split by 'filename' (each run) instead of 'model'.

    Returns:
      - rf
      - per-model metrics (on test split)
      - (X_test, y_test, model_test, feature_cols)
    """

    if "filename" not in df_feat.columns:
        raise ValueError("Column 'filename' is required for group-by-file splitting.")

    # Attach latent dims
    for i in range(latent_dim):
        df_feat[f"z{i}"] = latent[:, i]

    feature_cols = [f"z{i}" for i in range(latent_dim)] + get_config_columns()
    needed = feature_cols + [TARGET_COL, "model", "filename"]
    for c in needed:
        if c not in df_feat.columns:
            raise ValueError(f"Required column '{c}' not found in df_feat.")

    X = df_feat[feature_cols].values
    y = df_feat[TARGET_COL].values
    groups = df_feat["filename"].values  # group per run
    models_all = df_feat["model"].values

    # Grouped split by filename
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model_test = models_all[test_idx]

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_test = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print("\n=== Global RF evaluation (AE + normalized behavior, grouped by filename) ===")
    print(f"Test samples: {len(y_test)}")
    print(f"R^2:  {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print("==========================================================================\n")

    # Per-model metrics on test split (now every model appears in both train & test)
    metrics_rows = []
    test_df = pd.DataFrame({
        "model": model_test,
        "y_true": y_test,
        "y_pred": y_pred_test,
    })
    for m in sorted(test_df["model"].unique()):
        sub = test_df[test_df["model"] == m]
        if len(sub) < 3:
            continue
        r2_m = r2_score(sub["y_true"], sub["y_pred"])
        mape_m = mean_absolute_percentage_error(sub["y_true"], sub["y_pred"])
        metrics_rows.append({"model": m.strip(), "R2": r2_m, "MAPE": mape_m, "n": len(sub)})

    metrics_df = pd.DataFrame(metrics_rows)
    print("=== Per-model metrics on test set (grouped by filename) ===")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    else:
        print("[No per-model metrics; too few samples per model in test split]")
    print("==========================================================\n")

    return rf, metrics_df, (X_test, y_test, model_test, feature_cols)


########################################################
# 4. Plotting helpers
########################################################

def plot_pred_vs_true(X_test, y_test, rf, out_path: str):
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
    plt.ylim(-0.5, 1.05)
    plt.ylabel("R² (test)")
    plt.xlabel("Model")
    plt.title("Per-model R² on test split (grouped by filename)")
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
    plt.title("Feature importance (latent dims + config knobs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot → {out_path}")


########################################################
# 5. Config recommendation under power cap
########################################################

def recommend_best_config_for_model(
    df_feat: pd.DataFrame,
    rf: RandomForestRegressor,
    feature_cols: List[str],
    model_name: str,
    power_cap: float,
) -> Tuple[pd.Series, float]:
    """
    Pick the best config for a given benchmark (model_name) under cap_w <= power_cap.
    Uses AE+RF predictions to smooth noise and pick max tokens/s.
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
# 6. Main
########################################################

def main():
    parser = argparse.ArgumentParser(
        description="AE + behavior-normalized features, grouped by filename."
    )
    parser.add_argument("csv_path", type=str,
                        help="Path to gpu_timeseries_summaries.csv")
    parser.add_argument("--out-dir", type=str,
                        default="behavior_ae_groupfile_plots",
                        help="Directory to save plots and metrics.")
    parser.add_argument("--latent-dim", type=int,
                        default=4,
                        help="Latent dimension for behavior autoencoder.")
    parser.add_argument("--sample-power-cap", type=float,
                        default=300.0,
                        help="Power cap (W) for example recommendations.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)

    # 1) Engineer behavior + config features
    df_feat = engineer_features(df)

    # 2) Clean NaNs / infs and require filename
    behavior_cols = get_behavior_columns()
    cols_needed = behavior_cols + [TARGET_COL, "model", "filename"] + get_config_columns()

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    before = len(df_feat)
    df_feat = df_feat.dropna(subset=cols_needed).copy()
    after = len(df_feat)
    print(f"Filtered rows with NaN/inf in behavior/target/filename: {before - after} removed, {after} remain.")
    if after == 0:
        raise RuntimeError("No valid rows left after cleaning. Check your CSV / summaries.")

    # 3) Train autoencoder on behavior features
    X_beh = df_feat[behavior_cols].values
    print(f"Training autoencoder on behavior features (dim={X_beh.shape[1]}) "
          f"→ latent_dim={args.latent_dim}")
    ae_model, scaler, latent = train_autoencoder(
        X_beh,
        latent_dim=args.latent_dim,
        epochs=400,
        lr=1e-3,
    )

    # 4) Train RF on [latent, config] with group split by filename
    rf, metrics_df, (X_test, y_test, model_test, feature_cols) = train_rf_with_latent_group_by_file(
        df_feat, latent, latent_dim=args.latent_dim
    )

    # 5) Save per-model metrics
    metrics_path = os.path.join(args.out_dir, "per_model_metrics_test.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved per-model metrics (test) → {metrics_path}")

    # 6) Plots
    plot_pred_vs_true(
        X_test, y_test, rf,
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

    # 7) Example: best configs at a given power cap
    print(f"\n=== Example best configs at cap_w <= {args.sample_power_cap} W (AE+groupfile RF) ===")
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
