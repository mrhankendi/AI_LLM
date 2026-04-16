#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error


# =========================
# Data loading & training
# =========================

def load_data(csv_path: str,
              target_col: str = "avg_Tokens_per_s",
              feature_cols: List[str] = None
              ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the CSV and return (X_encoded, y, original_df_clean).

    feature_cols should include "model" plus numeric knobs like "tp", "cap_w", "batch".
    """
    if feature_cols is None:
        feature_cols = ["model", "tp", "cap_w", "batch"]

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    use_cols = feature_cols + [target_col]
    df_clean = df.dropna(subset=use_cols).copy()

    X_raw = df_clean[feature_cols].copy()
    y = df_clean[target_col].astype(float)

    # One-hot encode 'model'
    X_enc = pd.get_dummies(X_raw, columns=["model"])

    return X_enc, y, df_clean


def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor,
                                                        pd.DataFrame,
                                                        pd.Series,
                                                        pd.DataFrame,
                                                        pd.Series]:
    """
    Train a RandomForestRegressor and return model and split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except Exception:
        mape = float("nan")

    print("=== Global model evaluation ===")
    print(f"Samples (train+test): {len(X)}")
    print(f"R^2 on test set: {r2:.4f}")
    if mape == mape:  # not NaN
        print(f"MAPE on test set: {mape:.4f}")
    print("================================\n")

    return model, X_train, X_test, y_train, y_test


# =========================
# Plotting helpers
# =========================

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def plot_predicted_vs_true(model,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           out_path: str):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True Tokens/s")
    plt.ylabel("Predicted Tokens/s")
    plt.title("Predicted vs True Tokens/s")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Predicted vs True plot to {out_path}")


def plot_residuals(model,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   out_path: str):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.axvline(0, linestyle='--')
    plt.xlabel("Error (True - Predicted Tokens/s)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved residual distribution plot to {out_path}")


def plot_feature_importance(model,
                            feature_names: List[str],
                            out_path: str):
    importances = model.feature_importances_
    idx = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx], importances[idx])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")


def compute_per_model_metrics(df_clean: pd.DataFrame,
                              model: RandomForestRegressor,
                              X_enc: pd.DataFrame,
                              y: pd.Series) -> pd.DataFrame:
    """
    Compute R^2 and MAPE per 'model' (LLM benchmark).
    """
    df2 = df_clean.copy()
    df2["y_true"] = y.values
    df2["y_pred"] = model.predict(X_enc)

    results = []
    for m in df2["model"].unique():
        subset = df2[df2["model"] == m]
        true = subset["y_true"]
        pred = subset["y_pred"]
        if len(subset) < 3:
            # too few samples to be meaningful
            r2_val = np.nan
            mape_val = np.nan
        else:
            r2_val = r2_score(true, pred)
            # guard divide-by-zero
            with np.errstate(divide="ignore", invalid="ignore"):
                mape_val = np.mean(np.abs((true - pred) / true.replace(0, np.nan)))
        results.append({"model": m, "R2": r2_val, "MAPE": mape_val})

    res_df = pd.DataFrame(results)
    return res_df


def plot_per_model_metrics(metrics_df: pd.DataFrame,
                           out_path: str):
    """
    Plot R^2 per model as a bar chart.
    """
    metrics_df_sorted = metrics_df.sort_values("R2", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df_sorted["model"], metrics_df_sorted["R2"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R^2")
    plt.title("Per-model R^2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved per-model R^2 plot to {out_path}")


# =========================
# Recommendation logic
# =========================

def recommend_config(
    df_clean: pd.DataFrame,
    rf_model: RandomForestRegressor,
    feature_cols_encoded: List[str],
    model_name: str,
    power_cap: float,
    feature_cols_raw: List[str] = None,
) -> pd.Series:
    """
    Given a trained RF and the original df, recommend best config
    for a given model and power_cap (interpreted as max allowed cap_w).

    Returns a row with chosen configuration and predicted tokens/s.
    """
    if feature_cols_raw is None:
        feature_cols_raw = ["model", "tp", "cap_w", "batch"]

    cand = (
        df_clean[(df_clean["model"] == model_name) & (df_clean["cap_w"] <= power_cap)]
        [feature_cols_raw]
        .drop_duplicates()
    )

    if cand.empty:
        raise ValueError(
            f"No candidate configs for model='{model_name}' with cap_w <= {power_cap}. "
            f"Check available 'cap_w' values for this model."
        )

    # encode model like training
    cand_enc = pd.get_dummies(cand, columns=["model"])

    # align columns
    missing_cols = set(feature_cols_encoded) - set(cand_enc.columns)
    for c in missing_cols:
        cand_enc[c] = 0
    extra_cols = set(cand_enc.columns) - set(feature_cols_encoded)
    if extra_cols:
        cand_enc = cand_enc.drop(columns=list(extra_cols))

    cand_enc = cand_enc[feature_cols_encoded]

    cand["pred_tokens_per_s"] = rf_model.predict(cand_enc)

    best = cand.sort_values("pred_tokens_per_s", ascending=False).iloc[0]
    return best


def plot_recommendation_curve(
    df_clean: pd.DataFrame,
    rf_model: RandomForestRegressor,
    feature_cols_encoded: List[str],
    model_name: str,
    out_path: str,
    feature_cols_raw: List[str] = None,
):
    """
    For a range of power caps, plot the predicted best tokens/s for that model.
    """
    if feature_cols_raw is None:
        feature_cols_raw = ["model", "tp", "cap_w", "batch"]

    df_model = df_clean[df_clean["model"] == model_name]
    if df_model.empty:
        print(f"[WARN] No data for model '{model_name}'. Skipping curve.")
        return

    min_cap = df_model["cap_w"].min()
    max_cap = df_model["cap_w"].max()

    caps = np.linspace(min_cap, max_cap, num=10)  # 10 points across the range

    perf = []
    effective_caps = []

    for pc in caps:
        try:
            best = recommend_config(
                df_clean=df_clean,
                rf_model=rf_model,
                feature_cols_encoded=feature_cols_encoded,
                model_name=model_name,
                power_cap=pc,
                feature_cols_raw=feature_cols_raw,
            )
            perf.append(best["pred_tokens_per_s"])
            effective_caps.append(pc)
        except ValueError:
            # No config for this cap; skip
            continue

    if not perf:
        print(f"[WARN] No valid recommendations for '{model_name}' across caps. Skipping curve.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(effective_caps, perf, marker="o")
    plt.xlabel("Power cap (W)")
    plt.ylabel("Predicted Tokens/s")
    plt.title(f"Recommended performance vs power cap for {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved recommendation curve for {model_name} to {out_path}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Train config→tokens/s model, evaluate it, and generate plots + recommendations."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to gpu_timeseries_summaries.csv",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="avg_Tokens_per_s",
        help="Target performance column (default: avg_Tokens_per_s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="model_plots",
        help="Directory to save plots (default: model_plots).",
    )
    parser.add_argument(
        "--sample-power-cap",
        type=float,
        default=300.0,
        help="Power cap to use for printing example recommendations (default: 300 W).",
    )

    args = parser.parse_args()

    feature_cols_raw = ["model", "tp", "cap_w", "batch"]
    ensure_dir(args.out_dir)

    print(f"Loading data from {args.csv_path} ...")
    X_enc, y, df_clean = load_data(
        csv_path=args.csv_path,
        target_col=args.target_col,
        feature_cols=feature_cols_raw,
    )

    print("Training model ...")
    rf_model, X_train, X_test, y_train, y_test = train_model(X_enc, y)

    # ---- Global plots ----
    plot_predicted_vs_true(
        rf_model,
        X_test,
        y_test,
        out_path=os.path.join(args.out_dir, "pred_vs_true.png"),
    )

    plot_residuals(
        rf_model,
        X_test,
        y_test,
        out_path=os.path.join(args.out_dir, "residuals.png"),
    )

    plot_feature_importance(
        rf_model,
        feature_names=list(X_enc.columns),
        out_path=os.path.join(args.out_dir, "feature_importance.png"),
    )

    # ---- Per-model metrics ----
    per_model_df = compute_per_model_metrics(df_clean, rf_model, X_enc, y)
    print("=== Per-model metrics ===")
    print(per_model_df.to_string(index=False))
    print("=========================\n")

    per_model_metrics_path = os.path.join(args.out_dir, "per_model_metrics.csv")
    per_model_df.to_csv(per_model_metrics_path, index=False)
    print(f"Saved per-model metrics CSV to {per_model_metrics_path}")

    plot_per_model_metrics(
        per_model_df,
        out_path=os.path.join(args.out_dir, "per_model_R2.png"),
    )

    # ---- Recommendation curves and sample recommendations ----
    unique_models = sorted(df_clean["model"].unique())
    print(f"\nUnique models in data: {unique_models}\n")

    print(f"Example recommendations at power_cap={args.sample_power_cap} W:")
    for m in unique_models:
        try:
            best = recommend_config(
                df_clean=df_clean,
                rf_model=rf_model,
                feature_cols_encoded=list(X_enc.columns),
                model_name=m,
                power_cap=args.sample_power_cap,
                feature_cols_raw=feature_cols_raw,
            )
            print(f"- {m}: tp={best['tp']}, cap_w={best['cap_w']}, batch={best['batch']}, "
                  f"pred_tokens/s={best['pred_tokens_per_s']:.3f}")
        except ValueError as e:
            print(f"- {m}: no valid config under {args.sample_power_cap} W ({e})")

        # Recommendation curve per model
        curve_path = os.path.join(args.out_dir, f"reco_curve_{m}.png")
        plot_recommendation_curve(
            df_clean=df_clean,
            rf_model=rf_model,
            feature_cols_encoded=list(X_enc.columns),
            model_name=m,
            out_path=curve_path,
            feature_cols_raw=feature_cols_raw,
        )


if __name__ == "__main__":
    main()
