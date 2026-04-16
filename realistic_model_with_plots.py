#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error


FEATURE_COLS = ["tp", "cap_w", "batch"]
TARGET_COL = "avg_Tokens_per_s"


@dataclass
class ModelSurrogate:
    name: str
    reg: RandomForestRegressor
    feature_cols: List[str]
    y_min: float
    y_max: float


def load_data(csv_path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["model"] + FEATURE_COLS + [target_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"Required column '{c}' not in CSV. "
                f"Available: {list(df.columns)}"
            )
    df = df.dropna(subset=required).copy()
    return df


def train_per_model_surrogates(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.25,
) -> Tuple[Dict[str, ModelSurrogate], pd.DataFrame]:
    """
    Train one RandomForest regressor per LLM model on log(tokens/s+1).
    Returns:
      - dict: model_name -> ModelSurrogate
      - metrics_df: DataFrame with per-model R2 and MAPE
    """
    surrogates: Dict[str, ModelSurrogate] = {}
    rows = []

    print("=== Training per-model surrogates ===")
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m].copy()
        X = sub[FEATURE_COLS].astype(float)
        y = sub[target_col].astype(float)

        # log-transform target
        y_log = np.log1p(y)

        if len(sub) < 10:
            print(f"[WARN] model '{m}' has only {len(sub)} samples; results may be noisy.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )

        reg = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        reg.fit(X_train, y_train)

        # Evaluate in original units
        y_test_pred_log = reg.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        y_test_true = np.expm1(y_test)

        r2 = r2_score(y_test_true, y_test_pred)
        mape = mean_absolute_percentage_error(y_test_true, y_test_pred)

        print(f"Model '{m:15s}': n={len(sub):4d}, R^2={r2:6.3f}, MAPE={mape:6.3f}")

        y_min = float(y.min())
        y_max = float(y.max())
        surrogates[m] = ModelSurrogate(
            name=m,
            reg=reg,
            feature_cols=FEATURE_COLS,
            y_min=y_min,
            y_max=y_max,
        )
        rows.append({"model": m.strip(), "R2": r2, "MAPE": mape, "n": len(sub)})

    print("=====================================\n")
    metrics_df = pd.DataFrame(rows)
    return surrogates, metrics_df


def predict_tokens_per_s(surrogate: ModelSurrogate, X: pd.DataFrame) -> np.ndarray:
    y_log_pred = surrogate.reg.predict(X[surrogate.feature_cols])
    y_pred = np.expm1(y_log_pred)
    y_pred = np.clip(y_pred, surrogate.y_min, surrogate.y_max)
    return y_pred


def recommend_config_for_model(
    df: pd.DataFrame,
    surrogate: ModelSurrogate,
    power_cap: float,
) -> Tuple[pd.Series, float]:
    sub = df[(df["model"] == surrogate.name) & (df["cap_w"] <= power_cap)]
    if sub.empty:
        raise ValueError(
            f"No configs for model='{surrogate.name}' with cap_w <= {power_cap}"
        )

    cand = sub[FEATURE_COLS].drop_duplicates().copy()
    cand_pred = predict_tokens_per_s(surrogate, cand)
    cand["pred_tokens_per_s"] = cand_pred

    best = cand.sort_values("pred_tokens_per_s", ascending=False).iloc[0]
    return best, float(best["pred_tokens_per_s"])


def plot_recommendation_curve(
    df: pd.DataFrame,
    surrogate: ModelSurrogate,
    out_dir: str,
    n_points: int = 10,
):
    sub = df[df["model"] == surrogate.name]
    if sub.empty:
        return

    caps = np.linspace(sub["cap_w"].min(), sub["cap_w"].max(), num=n_points)
    eff_caps = []
    best_perf = []

    for pc in caps:
        try:
            best_cfg, perf = recommend_config_for_model(df, surrogate, pc)
            eff_caps.append(pc)
            best_perf.append(perf)
        except ValueError:
            continue

    if not eff_caps:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(eff_caps, best_perf, marker="o")
    plt.xlabel("Power cap (W)")
    plt.ylabel("Predicted tokens/s")
    plt.title(f"Recommended performance vs cap – {surrogate.name}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"reco_curve_{surrogate.name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved recommendation curve for {surrogate.name} → {out_path}")


def plot_r2_bar(metrics_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df_plot = metrics_df.sort_values("R2", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df_plot["model"], df_plot["R2"])
    plt.ylim(0, 1.05)
    plt.ylabel("R²")
    plt.xlabel("Model")
    plt.title("Per-model R² (log-RF surrogate)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_model_R2_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved R² bar plot → {out_path}")


def plot_feature_importance(surrogates: Dict[str, ModelSurrogate], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for name, surr in surrogates.items():
        importances = surr.reg.feature_importances_
        idx = np.argsort(importances)

        plt.figure(figsize=(6, 4))
        plt.barh(
            [surr.feature_cols[i] for i in idx],
            importances[idx]
        )
        plt.xlabel("Importance")
        plt.title(f"Feature importance – {name}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"feat_importance_{name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved feature importance plot for {name} → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-LLM realistic config→tokens/s modeling with R² + feature importances."
    )
    parser.add_argument("csv_path", type=str,
                        help="Path to gpu_timeseries_summaries.csv")
    parser.add_argument("--target-col", type=str,
                        default=TARGET_COL,
                        help=f"Performance column (default: {TARGET_COL}).")
    parser.add_argument("--out-dir", type=str,
                        default="realistic_model_plots",
                        help="Directory to save plots.")
    parser.add_argument("--sample-power-cap", type=float,
                        default=300.0,
                        help="Power cap (W) for example recommendations.")
    args = parser.parse_args()

    print(f"Loading data from {args.csv_path} ...")
    df = load_data(args.csv_path, target_col=args.target_col)

    # Train surrogates
    surrogates, metrics_df = train_per_model_surrogates(df, target_col=args.target_col)

    # Save metrics CSV
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "per_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved per-model metrics → {metrics_path}\n")

    # Plot R² bar chart
    plot_r2_bar(metrics_df, args.out_dir)

    # Plot feature importance per model
    plot_feature_importance(surrogates, args.out_dir)

    # Example recommendations
    print(f"=== Example recommendations at {args.sample_power_cap} W ===")
    for name, surr in surrogates.items():
        try:
            best_cfg, perf = recommend_config_for_model(df, surr, args.sample_power_cap)
            print(
                f"{name:15s} → tp={int(best_cfg['tp'])}, "
                f"cap_w={best_cfg['cap_w']:.0f}, "
                f"batch={int(best_cfg['batch'])}, "
                f"pred_tokens/s={perf:.1f}"
            )
        except ValueError as e:
            print(f"{name:15s} → {e}")
    print("==============================================\n")

    # Recommendation curves per model
    for name, surr in surrogates.items():
        plot_recommendation_curve(df, surr, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
