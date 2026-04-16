#!/usr/bin/env python3
"""
Train a power + performance predictor on moe_summary_tp4.xlsx

Targets:
  - avg_Tokens_per_s
  - total_gpu_power
  - power_efficiency

Inputs (features):
  - model (categorical)
  - cap_w
  - batch
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def main():
    # ---- 1. Load data ----
    data_path = Path("moe_summary_tp4.xlsx")  # adjust path if needed
    df = pd.read_excel(data_path)

    # Sanity check
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))

    # ---- 2. Select features & targets ----
    # You can expand features later (e.g., tp, etc.); tp is constant=4 in this file.
    feature_cols = ["model", "cap_w", "batch"]
    target_cols = ["avg_Tokens_per_s", "total_gpu_power", "power_efficiency"]

    X = df[feature_cols]
    y = df[target_cols]

    # ---- 3. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ["model"]
    numeric_features = ["cap_w", "batch"]

    # ---- 4. Preprocessing: one-hot encode model, pass through numeric ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # ---- 5. Base regressor ----
    base_regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    regressor = MultiOutputRegressor(base_regressor)

    # ---- 6. Full pipeline ----
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )

    # ---- 7. Train ----
    model.fit(X_train, y_train)

    # ---- 8. Evaluate on held-out test set ----
    y_pred = model.predict(X_test)

    # Per-target metrics
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    print("\n=== Test performance ===")
    for i, target in enumerate(target_cols):
        print(f"Target: {target}")
        print(f"  R² : {r2[i]:.4f}")
        print(f"  MAE: {mae[i]:.6f}")
        print()

    # ---- 9. Save model to disk ----
    out_path = Path("power_perf_model.joblib")
    joblib.dump(model, out_path)
    print(f"Saved trained model to {out_path.resolve()}")

    # ---- 10. Example: predicting for a new config ----
    example = pd.DataFrame(
        [
            {
                "model": "deepseek",  # must match strings in your data
                "cap_w": 130,
                "batch": 32,
            }
        ]
    )

    pred = model.predict(example)[0]
    print("Example config:", example.to_dict(orient="records")[0])
    print("Predicted avg_Tokens_per_s   :", pred[0])
    print("Predicted total_gpu_power    :", pred[1])
    print("Predicted power_efficiency   :", pred[2])


if __name__ == "__main__":
    main()
