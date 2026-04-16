import argparse
import sys
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error


def load_data(csv_path: str,
              target_col: str = "avg_Tokens_per_s",
              feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the CSV and return (X_encoded, y, original_df).

    feature_cols should include "model" plus numeric knobs like "tp", "cap_w", "batch".
    """
    if feature_cols is None:
        feature_cols = ["model", "tp", "cap_w", "batch"]

    df = pd.read_csv(csv_path)

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # Drop rows with missing values in those columns
    use_cols = feature_cols + [target_col]
    df_clean = df.dropna(subset=use_cols).copy()

    X_raw = df_clean[feature_cols].copy()
    y = df_clean[target_col].astype(float)

    # One-hot encode the model column
    X_enc = pd.get_dummies(X_raw, columns=["model"])

    return X_enc, y, df_clean


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the given features/target.
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

    print("=== Model evaluation ===")
    print(f"Samples (train+test): {len(X)}")
    print(f"R^2 on test set: {r2:.4f}")
    if mape == mape:  # not NaN
        print(f"MAPE on test set: {mape:.4f}")
    print("========================")

    return model


def recommend_config(
    df: pd.DataFrame,
    rf_model: RandomForestRegressor,
    feature_cols_encoded: List[str],
    model_name: str,
    power_cap: float,
    feature_cols_raw: List[str] = None,
) -> pd.Series:
    """
    Given a trained RF and the original df, recommend best config
    for a given model and power_cap (interpreted as max allowed cap_w).

    Returns a row with the chosen configuration and predicted tokens/s.
    """
    if feature_cols_raw is None:
        feature_cols_raw = ["model", "tp", "cap_w", "batch"]

    # Candidate configurations: same model, cap_w <= power_cap
    cand = (
        df[(df["model"] == model_name) & (df["cap_w"] <= power_cap)]
        [feature_cols_raw]
        .drop_duplicates()
    )

    if cand.empty:
        raise ValueError(
            f"No candidate configs for model='{model_name}' with cap_w <= {power_cap}. "
            f"Check available 'cap_w' values for this model."
        )

    # One-hot encode 'model' as in training
    cand_enc = pd.get_dummies(cand, columns=["model"])

    # Align columns with the training feature template
    missing_cols = set(feature_cols_encoded) - set(cand_enc.columns)
    for c in missing_cols:
        cand_enc[c] = 0

    extra_cols = set(cand_enc.columns) - set(feature_cols_encoded)
    if extra_cols:
        cand_enc = cand_enc.drop(columns=list(extra_cols))

    cand_enc = cand_enc[feature_cols_encoded]

    # Predict tokens/s
    cand["pred_tokens_per_s"] = rf_model.predict(cand_enc)

    # Pick the row with max predicted tokens/s
    best = cand.sort_values("pred_tokens_per_s", ascending=False).iloc[0]
    return best


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a model to predict tokens/s from (model, tp, cap_w, batch) and "
            "recommend the best configuration under a given power cap."
        )
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
        help="Name of the target column for performance (default: avg_Tokens_per_s).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the LLM/model to recommend a config for (e.g., deepseek).",
    )
    parser.add_argument(
        "--power-cap",
        type=float,
        default=None,
        help="Maximum allowed cap_w (Watts). Only configs with cap_w <= this are considered.",
    )
    args = parser.parse_args()

    feature_cols_raw = ["model", "tp", "cap_w", "batch"]

    print(f"Loading data from {args.csv_path} ...")
    X_enc, y, df_clean = load_data(
        csv_path=args.csv_path,
        target_col=args.target_col,
        feature_cols=feature_cols_raw,
    )

    print("Training model ...")
    rf = train_model(X_enc, y)

    # If user requested a recommendation, compute and print it
    if args.model_name is not None and args.power_cap is not None:
        print()
        print(f"Recommending config for model='{args.model_name}', power_cap={args.power_cap} W")
        try:
            best = recommend_config(
                df=df_clean,
                rf_model=rf,
                feature_cols_encoded=list(X_enc.columns),
                model_name=args.model_name,
                power_cap=args.power_cap,
                feature_cols_raw=feature_cols_raw,
            )
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        print("\n=== Recommended configuration ===")
        print(f"model        : {best['model']}")
        print(f"tp           : {best['tp']}")
        print(f"cap_w        : {best['cap_w']} W")
        print(f"batch        : {best['batch']}")
        print(f"pred_tokens/s: {best['pred_tokens_per_s']:.3f}")
        print("===============================")
    else:
        print()
        print("No recommendation requested (missing --model-name or --power-cap).")
        print("You can run, for example:")
        print(
            f"  python {sys.argv[0]} {args.csv_path} "
            "--model-name deepseek --power-cap 300"
        )


if __name__ == '__main__':
    main()
