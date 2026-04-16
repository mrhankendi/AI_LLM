import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. LOAD DATA
# ============================
df = pd.read_excel("moe_summary_tp4.xlsx")

print("Columns:", df.columns.tolist())

# Your columns (for reference):
# model, tp, cap_w, batch,
# avg_Tokens_per_s, norm_tokens_per_s,
# avg_mem_util, avg_gpu_util, avg_gpu_power,
# avg_data_receive, avg_data_transmit, avg_total_comm,
# power_efficiency, total_gpu_power

TARGET_COLS = {
    "tokens": "avg_Tokens_per_s",
    "power":  "total_gpu_power",
    "eff":    "power_efficiency",
}

# ----------------------------
# Features we WANT to use at runtime:
#   - config knobs: model, tp, cap_w, batch
#   - runtime signals: util, mem, comm, (optionally power)
# ----------------------------
BASE_FEATURES = [
    "tp",
    "batch",
    "avg_mem_util",
    "avg_gpu_util",
    "avg_data_receive",
    "avg_data_transmit",
    "avg_total_comm",
   # we'll selectively drop this for some targets to avoid leakage
]

# ----------------------------
# Per-target forbidden columns (to avoid leakage)
# ----------------------------
FORBIDDEN_BY_TARGET = {
    # Predicting tokens/s: don't feed tokens or normalized tokens or efficiency
    "tokens": [
        "avg_Tokens_per_s",
        "norm_tokens_per_s",
        "power_efficiency",
        "total_gpu_power",   # don't feed power or eff into tokens model
    ],
    # Predicting total power: don't feed any power-like or direct algebraic combos
    "power": [
        "total_gpu_power",
        "avg_gpu_power",     # very close to target
        "avg_Tokens_per_s",
        "norm_tokens_per_s",
        "power_efficiency",
    ],
    # Predicting efficiency: don't feed tokens or power (since eff ~ tokens/power)
    "eff": [
        "power_efficiency",
        "avg_Tokens_per_s",
        "norm_tokens_per_s",
        "total_gpu_power",
        "avg_gpu_power",
    ],
}


def train_and_plot(target_key: str):
    """
    Train a RandomForestRegressor for a given target (tokens, power, eff),
    using config + runtime metrics, but excluding leaky columns.
    Returns (model, feature_names, allowed_raw_cols).
    """
    target_col = TARGET_COLS[target_key]
    forbidden = set(FORBIDDEN_BY_TARGET[target_key])

    # Pick allowed raw feature columns for this target
    allowed_raw_cols = [c for c in BASE_FEATURES if c not in forbidden]

    # Safety: keep only columns that actually exist in df
    allowed_raw_cols = [c for c in allowed_raw_cols if c in df.columns]

    print(f"\n=== Training model for {target_col} ===")
    print("Using raw feature columns:", allowed_raw_cols)

    # Build X, y
    X_raw = df[allowed_raw_cols].copy()
    y = df[target_col].values

    # One-hot encode categoricals (e.g., model)
    X = pd.get_dummies(X_raw)
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R²  : {r2:.4f}")
    print(f"MAE : {mae:.4f}")

    # -----------------------------
    # Plot 1: Predicted vs Actual
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual — {target_col}")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 2: Residuals
    # -----------------------------
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=20)
    plt.title(f"Residuals — {target_col}")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 3: Feature Importances
    # -----------------------------
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names)
    imp_top = imp_series.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp_top.values, y=imp_top.index)
    plt.title(f"Top Feature Importances — {target_col}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return model, feature_names, allowed_raw_cols


# ============================
# 2. TRAIN ALL THREE MODELS
# ============================
model_tokens, feat_tokens, raw_tokens = train_and_plot("tokens")
model_power,  feat_power,  raw_power  = train_and_plot("power")
model_eff,    feat_eff,    raw_eff    = train_and_plot("eff")

# At this point you have:
#   - model_tokens, model_power, model_eff
#   - and the corresponding feature name lists feat_tokens, feat_power, feat_eff
# which we'll use when turning this into a runtime predictor.

# ============================
# 3. CONFIG RECOMMENDER
# ============================

def recommend_config(
    target_tokens_per_s: float,
    model_name: str,
    tp_value: int = 4,
    candidate_caps=None,
    candidate_batches=None,
    verbose: bool = True,
):
    """
    Given a target throughput (tokens/s) and model name, find the configuration
    that meets the performance target and maximizes predicted power efficiency.
    
    Args:
        target_tokens_per_s: Desired performance (tokens/s)
        model_name: Model name (e.g., "mixtral", "deepseek")
        tp_value: Tensor parallelism value (default: 4)
        candidate_caps: List of power caps to consider (if None, uses all from data)
        candidate_batches: List of batch sizes to consider (if None, uses all from data)
        verbose: Print detailed results
    
    Returns:
        pandas.Series with the recommended configuration
    """
    
    # Get unique values from the dataset
    available_caps = sorted(df['cap_w'].unique()) if 'cap_w' in df.columns else []
    available_batches = sorted(df['batch'].unique()) if 'batch' in df.columns else []
    
    # Use provided candidates or defaults from data
    caps = candidate_caps if candidate_caps is not None else available_caps
    batches = candidate_batches if candidate_batches is not None else available_batches
    
    if not caps or not batches:
        raise ValueError("No candidate caps or batches available")
    
    # Check if model exists in dataset
    if model_name not in df['model'].unique():
        available_models = df['model'].unique()
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(available_models)}")
    
    # Get average runtime metrics for this model from the dataset
    model_data = df[df['model'] == model_name]
    avg_metrics = {
        'avg_mem_util': model_data['avg_mem_util'].mean(),
        'avg_gpu_util': model_data['avg_gpu_util'].mean(),
        'avg_data_receive': model_data['avg_data_receive'].mean(),
        'avg_data_transmit': model_data['avg_data_transmit'].mean(),
        'avg_total_comm': model_data['avg_total_comm'].mean(),
    }
    
    # Generate all candidate configurations
    results = []
    
    for cap in caps:
        for batch_size in batches:
            # Build feature vector for prediction
            config = {
                'tp': tp_value,
                'batch': batch_size,
                **avg_metrics
            }
            
            # Prepare for prediction (handle one-hot encoding)
            X_config = pd.DataFrame([config])
            X_config = pd.get_dummies(X_config)
            
            # Predict tokens/s
            X_tokens_aligned = X_config.reindex(columns=feat_tokens, fill_value=0)
            pred_tokens = model_tokens.predict(X_tokens_aligned)[0]
            
            # Predict power efficiency
            X_eff_aligned = X_config.reindex(columns=feat_eff, fill_value=0)
            pred_efficiency = model_eff.predict(X_eff_aligned)[0]
            
            # Predict total power
            X_power_aligned = X_config.reindex(columns=feat_power, fill_value=0)
            pred_power = model_power.predict(X_power_aligned)[0]
            
            results.append({
                'model': model_name,
                'tp': tp_value,
                'cap_w': cap,
                'batch': batch_size,
                'pred_tokens_per_s': pred_tokens,
                'pred_efficiency': pred_efficiency,
                'pred_total_power': pred_power,
                'tokens_error': abs(pred_tokens - target_tokens_per_s),
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter configurations that meet the throughput requirement
    feasible = results_df[results_df['pred_tokens_per_s'] >= target_tokens_per_s]
    
    if feasible.empty:
        if verbose:
            print("\n⚠️  No configuration meets the throughput requirement.")
            best = results_df.sort_values('pred_tokens_per_s', ascending=False).iloc[0]
            print("Returning the highest-throughput config instead:\n")
            print(f"Model           : {best['model']}")
            print(f"TP              : {best['tp']}")
            print(f"cap_w (W)       : {best['cap_w']}")
            print(f"batch           : {best['batch']}")
            print(f"Pred tokens/s   : {best['pred_tokens_per_s']:.2f} (target: {target_tokens_per_s:.2f})")
            print(f"Pred efficiency : {best['pred_efficiency']:.6f} tokens/J")
            print(f"Pred total power: {best['pred_total_power']:.2f} W")
        return best
    
    # Among feasible configs, pick the one with maximum predicted efficiency
    best = feasible.sort_values('pred_efficiency', ascending=False).iloc[0]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ RECOMMENDED CONFIG for {model_name} @ {target_tokens_per_s:.1f} tokens/s")
        print(f"{'='*70}")
        print(f"Model           : {best['model']}")
        print(f"TP              : {best['tp']}")
        print(f"cap_w (W)       : {best['cap_w']}")
        print(f"batch           : {best['batch']}")
        print(f"Pred tokens/s   : {best['pred_tokens_per_s']:.2f}")
        print(f"Pred efficiency : {best['pred_efficiency']:.6f} tokens/J")
        print(f"Pred total power: {best['pred_total_power']:.2f} W")
        print(f"{'='*70}")
        
        # Show top 5 alternatives
        print("\n📊 Top 5 Alternative Configurations:")
        print(f"{'='*70}")
        top5 = feasible.sort_values('pred_efficiency', ascending=False).head(5)
        for idx, row in top5.iterrows():
            print(f"Cap: {row['cap_w']:3.0f}W | Batch: {row['batch']:3.0f} | "
                  f"Tokens/s: {row['pred_tokens_per_s']:7.1f} | "
                  f"Efficiency: {row['pred_efficiency']:.6f} | "
                  f"Power: {row['pred_total_power']:6.1f}W")
        print(f"{'='*70}\n")
    
    return best


# ============================
# 4. EXAMPLE USAGE
# ============================
if __name__ == "__main__":
    # Example 1: Find optimal config for mixtral @ 400 tokens/s
    target_tps = 600.0
    model_name = "deepseek"
    
    best_config = recommend_config(
        target_tokens_per_s=target_tps,
        model_name=model_name,
        tp_value=4,
        # Optional: restrict search space
        # candidate_caps=[150, 175, 200],
        # candidate_batches=[1, 4, 8, 16],
    )
    
    # Example 2: Find optimal config for a different model
    # best_config2 = recommend_config(
    #     target_tokens_per_s=500.0,
    #     model_name="deepseek",
    #     tp_value=4,
    # )
