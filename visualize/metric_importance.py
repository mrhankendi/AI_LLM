import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# 1. Load & basic metrics
# ------------------------
df = pd.read_excel("output/moe_gpu_profile.xlsx")

# Sum GPU powers to get total power
gpu_power_cols = [c for c in df.columns
                  if c.startswith("avg_GPU") and c.endswith("_Power_W")]
df["total_power_W"] = df[gpu_power_cols].sum(axis=1)

# Raw efficiency
df["eff_tokens_per_watt"] = df["avg_Tokens_per_s"] / df["total_power_W"]

# Per-model max normalization (throughput first, then efficiency)
df["model_max_tokens"] = df.groupby("model")["avg_Tokens_per_s"].transform("max")
df["T_norm"] = df["avg_Tokens_per_s"] / df["model_max_tokens"]
df["eff_norm"] = df["T_norm"] / df["total_power_W"]

# ----------------------------------------
# 2. Define per-model baseline & ROI metrics
# ----------------------------------------

def pick_baseline(sub: pd.DataFrame) -> pd.Series:
    """
    Pick baseline row per model.
    Here: tp=2, batch=1, cap_w=200 if it exists;
    otherwise first row as fallback.
    """
    candidates = sub[
        (sub["tp"] == 2) &
        (sub["batch"] == 1) &
        (sub["cap_w"] == 200)
    ]
    if len(candidates) == 0:
        return sub.iloc[0]
    else:
        return candidates.iloc[0]

# Compute baseline metrics per model
baseline_rows = []
for model, g in df.groupby("model"):
    baseline_rows.append(pick_baseline(g))

baseline_df = pd.DataFrame(baseline_rows)
baseline_df = baseline_df[["model",
                           "avg_Tokens_per_s",
                           "total_power_W",
                           "eff_tokens_per_watt",
                           "eff_norm"]].rename(
    columns={
        "avg_Tokens_per_s": "T_base",
        "total_power_W": "P_base",
        "eff_tokens_per_watt": "eff_base",
        "eff_norm": "eff_norm_base",
    }
)

# Attach baseline values back to full df
df = df.merge(baseline_df, on="model", how="left")

# Percentage gains/overheads vs baseline
df["thr_gain_pct"] = 100.0 * (df["avg_Tokens_per_s"] - df["T_base"]) / df["T_base"]
df["pwr_overhead_pct"] = 100.0 * (df["total_power_W"] - df["P_base"]) / df["P_base"]
df["eff_loss_pct"] = 100.0 * (df["eff_tokens_per_watt"] - df["eff_base"]) / df["eff_base"]

# ROI metrics
df["roi_thr_vs_pwr"] = np.where(
    df["pwr_overhead_pct"] > 0,
    df["thr_gain_pct"] / df["pwr_overhead_pct"],
    np.nan,
)

df["roi_thr_vs_effloss"] = np.where(
    df["eff_loss_pct"] < 0,
    df["thr_gain_pct"] / np.abs(df["eff_loss_pct"]),
    np.nan,
)

# ---------------------------------------
# 3. Tier labels (optional, like your text)
# ---------------------------------------
# Simple example based on roi_thr_vs_pwr and efficiency loss
def classify_row(row):
    if row["tp"] == 2 and row["eff_loss_pct"] <= 10:
        return "TP2-Optimal"
    elif row["roi_thr_vs_pwr"] >= 1.0 and row["thr_gain_pct"] > 0:
        return "TP4-Viable"
    else:
        return "Inefficient"

df["tier"] = df.apply(classify_row, axis=1)

# ---------------------------------------
# 4. Plot: Efficiency loss vs Throughput ROI
# ---------------------------------------

plt.style.use("default")
plt.rcParams.update({
    "axes.facecolor": "#111111",
    "figure.facecolor": "#111111",
    "savefig.facecolor": "#111111",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

colors = {
    "TP2-Optimal": "#29B6F6",   # cyan
    "TP4-Viable":  "#FFB300",   # amber
    "Inefficient": "#EF5350",   # red
}

fig, ax = plt.subplots(figsize=(8, 6))

for tier, g in df.groupby("tier"):
    ax.scatter(
        g["eff_loss_pct"],
        g["roi_thr_vs_pwr"],
        label=tier,
        s=35,
        alpha=0.9,
        color=colors.get(tier, "white"),
        edgecolor="none",
    )

# Label each point with model name (optional, can be cluttered)
for _, row in df.iterrows():
    ax.text(
        row["eff_loss_pct"],
        row["roi_thr_vs_pwr"],
        row["model"],
        fontsize=7,
        color="white",
        alpha=0.7,
        ha="center",
        va="bottom",
    )

ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
ax.axhline(1, color="#444444", linestyle="--", linewidth=1)

ax.set_xlabel("Efficiency Loss vs Baseline (%)")
ax.set_ylabel("Throughput ROI (throughput_gain% / power_overhead%)")
ax.set_title("Model-Specific Scaling Viability: Efficiency Loss vs Throughput ROI")

ax.grid(axis="y", linestyle="--", alpha=0.3, color="#444444")
leg = ax.legend(frameon=True, facecolor="#222222", edgecolor="#555555")
for text in leg.get_texts():
    text.set_color("white")

fig.tight_layout()
fig.savefig("moe_efficiency_loss_vs_roi.png", dpi=300, bbox_inches="tight")

# -------------------------------------------------
# 5. Plot: Feature importance for eff_norm (metric importance)
# -------------------------------------------------

# Choose features
feature_cols = [
    "tp", "batch", "cap_w",
    # add whatever behavior metrics you want here:
    "avg_GPU0_Util_%", "avg_GPU0_MemUtil_%",
    "avg_GPU0_PCIe_RX_MBps", "avg_GPU0_PCIe_TX_MBps",
]
# Drop rows with missing cols if any
df_feat = df.dropna(subset=feature_cols + ["eff_norm"]).copy()

X = df_feat[feature_cols]
y = df_feat["eff_norm"]

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    max_depth=None,
)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(importances.index, importances.values)
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Metric / Knob Importance for Normalized Efficiency")
ax.grid(axis="x", linestyle="--", alpha=0.3, color="#444444")

fig.tight_layout()
fig.savefig("moe_feature_importance_eff_norm.png", dpi=300, bbox_inches="tight")
