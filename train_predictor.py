import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_excel("moe_summary_tp4.xlsx")

# ---------------------------
# REMOVE LEAKY TARGET-DEPENDENT COLUMNS
# ---------------------------
leaky_cols = [
    "avg_Tokens_per_s",
    "norm_tokens_per_s",
    "total_gpu_power",
    "avg_gpu_power",
    "power_efficiency",
    "cap_w",
    "model"

]

target_columns = {
    "tokens": "avg_Tokens_per_s",
    "power": "total_gpu_power",
    "eff": "power_efficiency"
}

# Keep only feature columns that are not leaky + not target
feature_cols = [c for c in df.columns if c not in leaky_cols]

# Optionally remove ID or timestamp-like columns
drop_noise = ["run_id", "timestamp"]
feature_cols = [c for c in feature_cols if c not in drop_noise and c in df.columns]

X = df[feature_cols]

# One-hot encode categorical columns
X = pd.get_dummies(X)

# ------------------------------------------
# FUNCTION: Train model for one target
# ------------------------------------------
def train_and_plot(target_name):
    y = df[target_columns[target_name]]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=12
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n=== Test performance for {target_columns[target_name]} ===")
    print(f"R² : {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # -----------------------------
    # VISUAL 1: Predicted vs Actual
    # -----------------------------
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual — {target_columns[target_name]}")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "k--")
    plt.show()

    # -----------------------------
    # VISUAL 2: Residuals
    # -----------------------------
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals — {target_columns[target_name]}")
    plt.xlabel("Residual")
    plt.show()

    # -----------------------------
    # VISUAL 3: Feature Importance
    # -----------------------------
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=X.columns)
    imp_top = imp_series.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8,6))
    sns.barplot(x=imp_top.values, y=imp_top.index)
    plt.title(f"Top Feature Importances — {target_columns[target_name]}")
    plt.xlabel("Importance")
    plt.show()

    return model


# ---------------------------
# TRAIN MODELS
# ---------------------------
model_tokens = train_and_plot("tokens")
model_power = train_and_plot("power")
model_eff = train_and_plot("eff")
