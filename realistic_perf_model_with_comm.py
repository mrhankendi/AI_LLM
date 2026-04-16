#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

########################################################
# Helper: aggregate PCIe load across all GPUs
########################################################
def compute_comm_features(df):
    rx_cols = [c for c in df.columns if "PCIe_RX" in c]
    tx_cols = [c for c in df.columns if "PCIe_TX" in c]

    df["comm_rx"] = df[rx_cols].sum(axis=1)
    df["comm_tx"] = df[tx_cols].sum(axis=1)
    df["comm_bw"] = df["comm_rx"] + df["comm_tx"]
    df["inv_comm_load"] = 1.0 / (df["comm_bw"] + 1e-3)

    return df


########################################################
# Train compute-bound and comm-bound models per benchmark
########################################################
def train_per_model_surrogates(df, target_col):
    models = sorted(df["model"].unique())
    surrogates = {}

    print("\n=== Training Compute & Communication Surrogates (per model) ===")

    for m in models:
        sub = df[df["model"] == m].copy()
        if len(sub) < 20:
            print(f"[WARN] model {m} has too few samples for reliable surrogate")
            continue

        # -------------------------
        # Surrogate A: compute-bound
        # -------------------------
        Xc = sub[["batch", "tp", "cap_w"]]
        yc = sub[target_col]

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.25, random_state=42
        )

        reg_compute = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        reg_compute.fit(Xc_train, yc_train)
        yc_pred = reg_compute.predict(Xc_test)
        r2c = r2_score(yc_test, yc_pred)

        # -------------------------
        # Surrogate B: communication-bound
        # -------------------------
        Xcomm = sub[["inv_comm_load", "tp", "batch"]]
        ycomm = sub[target_col]

        Xcomm_train, Xcomm_test, ycomm_train, ycomm_test = train_test_split(
            Xcomm, ycomm, test_size=0.25, random_state=42
        )

        reg_comm = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        reg_comm.fit(Xcomm_train, ycomm_train)
        ycomm_pred = reg_comm.predict(Xcomm_test)
        r2d = r2_score(ycomm_test, ycomm_pred)

        print(f"{m:12s} | R2_compute={r2c:.3f} | R2_comm={r2d:.3f}")

        surrogates[m] = dict(
            compute=reg_compute,
            comm=reg_comm,
            y_min=sub[target_col].min(),
            y_max=sub[target_col].max(),
        )

    print("===============================================================\n")
    return surrogates


########################################################
# Predict throughput using bottleneck rule
########################################################
def predict_tp(surr, Xc, Xcomm):
    yA = surr["compute"].predict(Xc)
    yB = surr["comm"].predict(Xcomm)
    y = np.minimum(yA, yB)
    y = np.clip(y, surr["y_min"], surr["y_max"])
    return y


########################################################
# Recommend best config per model under a power cap
########################################################
def recommend(df, model_name, surr, power_cap, target_col):
    sub = df[(df["model"] == model_name) & (df["cap_w"] <= power_cap)]
    if sub.empty:
        raise ValueError(f"No configs for {model_name} under cap {power_cap}")

    Xc = sub[["batch", "tp", "cap_w"]]
    Xcomm = sub[["inv_comm_load", "tp", "batch"]]

    preds = predict_tp(surr, Xc, Xcomm)
    sub = sub.copy()
    sub["pred"] = preds

    best = sub.sort_values("pred", ascending=False).iloc[0]
    return best


########################################################
# Plot scaling curve per model
########################################################
def plot_model_curve(df, model_name, surr, outdir):
    sub = df[df["model"] == model_name]
    caps = np.linspace(sub["cap_w"].min(), sub["cap_w"].max(), 10)

    perf = []
    for cap in caps:
        try:
            best = recommend(df, model_name, surr, cap, target_col="avg_Tokens_per_s")
            perf.append(best["pred"])
        except:
            perf.append(np.nan)

    plt.plot(caps, perf, marker="o")
    plt.title(f"Throughput vs Cap — {model_name}")
    plt.xlabel("Power Cap (W)")
    plt.ylabel("Predicted Tokens/s")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{outdir}/curve_{model_name}.png")
    plt.close()


########################################################
# Main
########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--target-col", default="avg_Tokens_per_s")
    parser.add_argument("--out", default="comm_model_plots")
    parser.add_argument("--cap", type=float, default=300)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df = compute_comm_features(df)

    surrogates = train_per_model_surrogates(df, args.target_col)

    print(f"=== Example configs at cap={args.cap} W ===")
    for m in surrogates:
        try:
            best = recommend(df, m, surrogates[m], args.cap, args.target_col)
            print(f"{m:12s} → tp={best['tp']}, batch={best['batch']}, cap={best['cap_w']} → {best['pred']:.1f} tok/s")
        except Exception as e:
            print(f"{m:12s} → {e}")

    print("\n=== Plotting curves ===")
    for m in surrogates:
        plot_model_curve(df, m, surrogates[m], args.out)
    print("Done!")


if __name__ == "__main__":
    main()
