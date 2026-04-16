import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1) LOAD DATA
# ============================================================
df = pd.read_excel("moe_summary_tp4.xlsx")

# Columns in your file:
# model, tp, cap_w, batch, avg_Tokens_per_s, norm_tokens_per_s,
# avg_mem_util, avg_gpu_util, avg_gpu_power,
# avg_data_receive, avg_data_transmit, avg_total_comm,
# power_efficiency, total_gpu_power

target_columns = {
    "tokens": "avg_Tokens_per_s",
    "power":  "total_gpu_power",
    "eff":    "power_efficiency"
}

# ============================================================
# 2) BUILD FEATURE MATRIX (NO LEAKAGE)
#    Only use knobs + static info: model, tp, cap_w, batch
# ============================================================
base_features = ["model", "tp", "cap_w", "batch"]
X_raw = df[base_features]

# One-hot encode categorical columns (model)
X = pd.get_dummies(X_raw, columns=["model"])

# We'll reuse this list of feature names later when we make predictions
feature_columns = X.columns.tolist()

# Use the same X for all three targets; just change y
X_train, X_test, _, _ = train_test_split(
    X, df[target_columns["tokens"]],
    test_size=0.2, random_state=42
)

# ============================================================
# 3) TRAIN MODELS + EVALUATE + SIMPLE VISUALS
# ============================================================
def train_model_for_target(target_key):
    """
    Train a RandomForestRegressor for one target (tokens, power, eff).
    Reuses the same X_train/X_test split for comparability.
    """
    target_col = target_columns[target_key]
    y = df[target_col]

    # Align y with the train/test indices of X
    y_train = y.loc[X_train.index]
    y_test  = y.loc[X_test.index]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=12,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== Test performance for {target_col} ===")
    print(f"R² : {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # --------- Visual 1: Predicted vs Actual scatter ----------
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual — {target_col}")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.tight_layout()
    plt.show()

    # --------- Visual 2: Residuals histogram ----------
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals — {target_col}")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.show()

        # -----------------------------
    # VISUAL 3: Feature Importance
    # -----------------------------
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=X.columns)
    imp_top = imp_series.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8,6))
    sns.barplot(x=imp_top.values, y=imp_top.index)
    plt.title(f"Top Feature Importances — {target_columns[target_key]}")
    plt.xlabel("Importance")
    plt.show()


    return model

print("Training models (tokens, power, efficiency) without leakage...")
model_tokens = train_model_for_target("tokens")
model_power  = train_model_for_target("power")
model_eff    = train_model_for_target("eff")

# ============================================================
# FEATURE IMPORTANCE VISUALIZATION (GLOBAL)
# ============================================================

def plot_feature_importances(model, X, top_n=20):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Extract importances
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi_sorted = fi.sort_values(ascending=False)

    # -----------------------
    # Plot 1 — Top-N Features
    # -----------------------
    plt.figure(figsize=(10, 7))
    sns.barplot(x=fi_sorted.head(top_n), y=fi_sorted.head(top_n).index)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------
    # Plot 2 — Full Ranked Heatmap
    # ----------------------------------------
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        fi_sorted.to_frame(name="importance"),
        annot=False,
        cmap="viridis",
        cbar=True
    )
    plt.title("Full Feature Importance Ranking (All Features)")
    plt.tight_layout()
    plt.show()

    # Print top features
    print("\nTop Important Features:")
    print(fi_sorted.head(top_n))


# ============================================================
# CALL THE VISUALIZATION FOR YOUR CHOSEN MODEL
# ============================================================
plot_feature_importances(model_eff, X, top_n=25)

models = {
    "tokens": model_tokens,
    "power":  model_power,
    "eff":    model_eff,
}

# ============================================================
# 4) CONFIG RECOMMENDER
#    Pick (cap_w, batch) that meets throughput target and
#    maximizes predicted power_efficiency
# ============================================================

# Precompute all unique (model, tp, cap_w, batch) combos from the data
design_df = df[["model", "tp", "cap_w", "batch"]].drop_duplicates()

def _build_feature_matrix_from_design(design_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a subset of design_df into the feature matrix used by the models.
    Ensures columns match training-time feature_columns exactly.
    """
    X_cand = pd.get_dummies(design_subset, columns=["model"])
    # Align to training feature columns (add missing cols filled with 0)
    X_cand = X_cand.reindex(columns=feature_columns, fill_value=0)
    return X_cand

def recommend_config(
    target_tokens_per_s: float,
    model_name: str,
    tp_value: int = 4,
    candidate_caps=None,
    candidate_batches=None,
    verbose: bool = True,
):
    """
    Given a target throughput (tokens/s), model name (e.g. "deepseek"),
    and tp value (e.g. 4), scan candidate (cap_w, batch) configs that
    exist in the dataset and choose the one that:
        - satisfies predicted_tokens >= target_tokens_per_s
        - has maximum predicted efficiency among those.

    Optionally restrict caps / batches via candidate_caps / candidate_batches.
    """

    # Filter design space for this model & tp
    subset = design_df[
        (design_df["model"] == model_name) &
        (design_df["tp"] == tp_value)
    ]

    if candidate_caps is not None:
        subset = subset[subset["cap_w"].isin(candidate_caps)]
    if candidate_batches is not None:
        subset = subset[subset["batch"].isin(candidate_batches)]

    if subset.empty:
        raise ValueError("No candidate configs found for given model/tp/filters.")

    # Build feature matrix matching training columns
    X_cand = _build_feature_matrix_from_design(subset)

    # Predict metrics for each candidate config
    pred_tokens = models["tokens"].predict(X_cand)
    pred_power  = models["power"].predict(X_cand)
    pred_eff    = models["eff"].predict(X_cand)

    subset = subset.copy()
    subset["pred_tokens_per_s"] = pred_tokens
    subset["pred_total_power"]  = pred_power
    subset["pred_efficiency"]   = pred_eff

    # Filter configs that meet throughput requirement
    feasible = subset[subset["pred_tokens_per_s"] >= target_tokens_per_s]

    if feasible.empty:
        if verbose:
            print("\n[WARN] No config meets the throughput requirement.")
            best = subset.sort_values("pred_tokens_per_s", ascending=False).iloc[0]
            print("Returning the highest-throughput config instead.")
            print(best)
        return subset.sort_values("pred_tokens_per_s", ascending=False).iloc[0]

    # Among feasible configs, pick the one with maximum predicted efficiency
    best = feasible.sort_values("pred_efficiency", ascending=False).iloc[0]

    if verbose:
        print("\n=== Recommended config ===")
        print(f"Model           : {best['model']}")
        print(f"TP              : {best['tp']}")
        print(f"cap_w (W)       : {best['cap_w']}")
        print(f"batch           : {best['batch']}")
        print(f"Pred tokens/s   : {best['pred_tokens_per_s']:.2f}")
        print(f"Pred total power: {best['pred_total_power']:.2f} W")
        print(f"Pred efficiency : {best['pred_efficiency']:.6f} tokens/J")

    return best


# ============================================================
# 5) EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Example: want at least 300 tokens/s for deepseek with tp=4
    target_tps   = 400.0
    model_name   = "mixtral"
    tp_val       = 4

    best_cfg = recommend_config(
        target_tokens_per_s=target_tps,
        model_name=model_name,
        tp_value=tp_val,
        # you can restrict to certain caps/batches like:
        # candidate_caps=[150, 175, 200],
        # candidate_batches=[1, 4, 8, 16],
    )

    # best_cfg is a pandas Series with fields:
    # ['model', 'tp', 'cap_w', 'batch', 'pred_tokens_per_s', 'pred_total_power', 'pred_efficiency']
