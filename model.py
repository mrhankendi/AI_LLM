#!/usr/bin/env python3
# MoE-only RandomForest (80/20 split by model) for efficiency & performance
# - Loads PROFILE.zip (same folder)
# - Reconstructs dataset from *summary CSVs* (fast). If you want PCIe comm too,
#   set USE_COMM=True and it will also parse gpu_timeseries CSVs to add comm_MBps.
# - Produces plots and a predictions CSV in ./moe_rf_outputs

import os, re, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- Config ----------------
ZIP_PATH   = Path("PROFILE.zip")
OUT_DIR    = Path("moe_rf_outputs")
USE_COMM   = False   # flip to True if you want to include cumulative PCIe MB/s (slower)

OUT_DIR.mkdir(exist_ok=True)

# ------------- Unzip once ---------------
extract_dir = Path("PROFILE_extracted")
if not extract_dir.exists():
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(extract_dir)

# ------------- Helpers ------------------
def parse_meta_from_name(name: str):
    name_low = name.lower()
    if   "gpt2"     in name_low: model="gpt2"
    elif "deepseek" in name_low: model="deepseek"
    elif "olmoe"    in name_low: model="olmoe"
    elif "qwen"     in name_low: model="qwen"
    elif "mixtral"  in name_low: model="mixtral"
    else: model="unknown"
    tp  = int(re.search(r"_tp(\d+)_",  name).group(1))
    cap = int(re.search(r"_cap(\d+)_", name).group(1))
    bs  = int(re.search(r"_bs(\d+)_",  name).group(1))
    return model, tp, cap, bs

# ------------- Build comm_MBps (optional) -------------
comm_df = None
if USE_COMM:
    ts_files = [
        Path(root)/f
        for root, dirs, files in os.walk(extract_dir)
        for f in files
        if "gpu_timeseries" in f.lower()
    ]
    rows = []
    for p in ts_files:
        try:
            df = pd.read_csv(p)
            model, tp, cap, bs = parse_meta_from_name(p.name)
            if model == "unknown":
                continue
            tx = df.filter(regex="PCIe_TX_MBps").sum(axis=1).mean()
            rx = df.filter(regex="PCIe_RX_MBps").sum(axis=1).mean()
            comm = float(tx + rx)
            rows.append({
                "model": model,
                "Tensor Parallel Size": tp,
                "Power Cap (W)": cap,
                "Batch Size": bs,
                "comm_MBps": comm
            })
        except Exception:
            pass
    comm_df = pd.DataFrame(rows)

# ------------- Load *summary* CSVs -------------
summary_files = [p for p in Path(extract_dir).rglob("*summary*csv")]
perf_rows = []
for p in summary_files:
    try:
        df = pd.read_csv(p)
        name = p.name.lower()
        if   "gpt2"     in name: model="gpt2"
        elif "deepseek" in name: model="deepseek"
        elif "olmoe"    in name: model="olmoe"
        elif "qwen"     in name: model="qwen"
        elif "mixtral"  in name: model="mixtral"
        else: continue
        df["model"] = model
        perf_rows.append(df)
    except Exception:
        pass

if not perf_rows:
    raise RuntimeError("No *summary* CSVs found inside PROFILE.zip extraction.")

perf_df = pd.concat(perf_rows, ignore_index=True)

# ------------- Merge comm if requested -------------
if USE_COMM and comm_df is not None and len(comm_df) > 0:
    merged = pd.merge(
        perf_df,
        comm_df,
        on=["model", "Tensor Parallel Size", "Power Cap (W)", "Batch Size"],
        how="left"
    )
else:
    merged = perf_df.copy()
    merged["comm_MBps"] = np.nan  # placeholder

# ------------- Filter to MoEs only -------------
moe_df = merged[merged["model"].isin(["deepseek", "olmoe", "qwen", "mixtral"])].copy()

# ------------- Features & targets -------------
base_feats = [
    "Tensor Parallel Size",
    "Power Cap (W)",
    "Batch Size",
    "Avg GPU Util (%)",
    "Avg GPU Total Power (W)",
    "Avg System Power (W)",
    "Throttle Fraction (Any)",
]
# Include comm if available and requested
if USE_COMM:
    base_feats = base_feats + ["comm_MBps"]

target_eff  = "Tokens/s/W (System)"
target_perf = "Tokens/s"

X = pd.concat([moe_df[base_feats], pd.get_dummies(moe_df["model"], prefix="model")], axis=1)
y_eff  = moe_df[target_eff].values
y_perf = moe_df[target_perf].values
labels = moe_df["model"]

# ------------- 80/20 split (stratified by model) -------------
X_tr, X_te, y_eff_tr, y_eff_te, y_perf_tr, y_perf_te, lab_tr, lab_te = train_test_split(
    X, y_eff, y_perf, labels, test_size=0.2, random_state=42, stratify=labels
)

# ------------- Train Random Forests -------------
rf_eff  = RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1)
rf_perf = RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1)

rf_eff.fit(X_tr, y_eff_tr)
rf_perf.fit(X_tr, y_perf_tr)

eff_pred  = rf_eff.predict(X_te)
perf_pred = rf_perf.predict(X_te)

def metrics(y_true, y_hat):
    return r2_score(y_true, y_hat), mean_squared_error(y_true, y_hat, squared=False)

r2_eff,  rmse_eff  = metrics(y_eff_te,  eff_pred)
r2_perf, rmse_perf = metrics(y_perf_te, perf_pred)

print("=== Overall (MoE, 80/20 stratified) ===")
print(f"Efficiency  (Tokens/s/W) -> R2: {r2_eff:.3f}, RMSE: {rmse_eff:.3f}")
print(f"Performance (Tokens/s)   -> R2: {r2_perf:.3f}, RMSE: {rmse_perf:.3f}")

# Per-model test metrics
rows = []
for m in sorted(lab_te.unique()):
    idx = (lab_te == m).values
    if idx.sum() < 2:  # avoid degenerate R2
        continue
    r2_e, rmse_e = metrics(y_eff_te[idx], eff_pred[idx])
    r2_p, rmse_p = metrics(y_perf_te[idx], perf_pred[idx])
    rows.append({"Model": m, "R2_eff": r2_e, "RMSE_eff": rmse_e,
                 "R2_perf": r2_p, "RMSE_perf": rmse_p})
per_model_df = pd.DataFrame(rows).sort_values("Model")
print("\n=== Per-model test metrics ===")
print(per_model_df.to_string(index=False))

# ------------- Plots (matplotlib only) -------------
def savefig(name):
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out

# 1) Predicted vs Actual — Efficiency
plt.figure()
plt.scatter(y_eff_te, eff_pred)
lims = [min(y_eff_te.min(), eff_pred.min()), max(y_eff_te.max(), eff_pred.max())]
plt.plot(lims, lims, '--')
plt.xlabel("Actual Tokens/s/W (System)")
plt.ylabel("Predicted Tokens/s/W (System)")
plt.title(f"MoE RF — Efficiency (80/20)\nR²={r2_eff:.3f}, RMSE={rmse_eff:.3f}")
savefig("moe_rf_efficiency_pred_vs_actual.png")

# 2) Predicted vs Actual — Performance
plt.figure()
plt.scatter(y_perf_te, perf_pred)
lims = [min(y_perf_te.min(), perf_pred.min()), max(y_perf_te.max(), perf_pred.max())]
plt.plot(lims, lims, '--')
plt.xlabel("Actual Tokens/s")
plt.ylabel("Predicted Tokens/s")
plt.title(f"MoE RF — Performance (80/20)\nR²={r2_perf:.3f}, RMSE={rmse_perf:.3f}")
savefig("moe_rf_performance_pred_vs_actual.png")

# 3) Feature importances — Efficiency (top 10)
imp_eff = pd.Series(rf_eff.feature_importances_, index=X_tr.columns).sort_values(ascending=False).head(10)
plt.figure()
plt.bar(range(len(imp_eff)), imp_eff.values)
plt.xticks(range(len(imp_eff)), imp_eff.index, rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Top Feature Importances — Efficiency model")
savefig("moe_rf_feature_importances_efficiency.png")

# 4) Feature importances — Performance (top 10)
imp_perf = pd.Series(rf_perf.feature_importances_, index=X_tr.columns).sort_values(ascending=False).head(10)
plt.figure()
plt.bar(range(len(imp_perf)), imp_perf.values)
plt.xticks(range(len(imp_perf)), imp_perf.index, rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Top Feature Importances — Performance model")
savefig("moe_rf_feature_importances_performance.png")

# ------------- Save predictions -------------
pred_df = pd.DataFrame({
    "model": lab_te.values,
    "TP": X_te["Tensor Parallel Size"].values,
    "cap_W": X_te["Power Cap (W)"].values,
    "batch": X_te["Batch Size"].values,
    "eff_true": y_eff_te,
    "eff_pred": eff_pred,
    "perf_true": y_perf_te,
    "perf_pred": perf_pred,
})
pred_csv = OUT_DIR / "moe_rf_80_20_predictions.csv"
pred_df.to_csv(pred_csv, index=False)

# ------------- Print artifact paths -------------
print("\nArtifacts written to:", OUT_DIR.resolve())
for f in sorted(OUT_DIR.iterdir()):
    print(" -", f.name)
