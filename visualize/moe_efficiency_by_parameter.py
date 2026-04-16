#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("output/dense_gpu_profile.xlsx")
OUT_PNG  = Path("dense_power_efficiency_normalized.png")

SCALE = 1e3   # <-- key fix: rescale for readability

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_PATH)

# -----------------------------
# Compute normalized efficiency
# -----------------------------
gpu_power_cols = [
    c for c in df.columns
    if c.startswith("avg_GPU") and c.endswith("_Power_W")
]
df["total_power_W"] = df[gpu_power_cols].sum(axis=1)

df["model_max_tokens"] = df.groupby("model")["avg_Tokens_per_s"].transform("max")
df["norm_tokens"] = df["avg_Tokens_per_s"] / df["model_max_tokens"]
df["norm_tokens_per_watt"] = df["norm_tokens"] / df["total_power_W"]

# Apply scaling *after* normalization
df["norm_eff_scaled"] = df["norm_tokens_per_watt"] * SCALE

# -----------------------------
# Aggregate by config parameters
# -----------------------------
tp_eff = df.groupby("tp")["norm_eff_scaled"].mean().reset_index()
batch_eff = df.groupby("batch")["norm_eff_scaled"].mean().reset_index()
cap_eff = df.groupby("cap_w")["norm_eff_scaled"].mean().reset_index()

tp_df = tp_eff.assign(
    label=lambda x: x["tp"].map(lambda v: f"TP={v}"),
    group="Tensor Parallelism",
)

batch_df = batch_eff.assign(
    label=lambda x: x["batch"].map(lambda v: f"Batch-{v}"),
    group="Batch Size",
)

cap_df = cap_eff.assign(
    label=lambda x: x["cap_w"].map(lambda v: f"Cap-{v}W"),
    group="Power Cap",
)

plot_df = pd.concat(
    [
        batch_df[["label", "norm_eff_scaled", "group"]],
        cap_df[["label", "norm_eff_scaled", "group"]],
        tp_df[["label", "norm_eff_scaled", "group"]],
    ],
    ignore_index=True,
)

# -----------------------------
# Plot (same dark style)
# -----------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

colors = {
    "Batch Size": "#2E86DE",
    "Power Cap": "#E91E63",
    "Tensor Parallelism": "#F39C12",
}

x = range(len(plot_df))
bars = ax.bar(
    x,
    plot_df["norm_eff_scaled"],
    color=[colors[g] for g in plot_df["group"]],
)

# Value annotations (now readable)
ymax = plot_df["norm_eff_scaled"].max()
for bar, val in zip(bars, plot_df["norm_eff_scaled"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.03 * ymax,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        color="white",
        fontsize=9,
    )

ax.set_xticks(list(x))
ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", color="white")
ax.set_ylabel(
    "Normalized Power Efficiency × 10³\n((tokens/s ÷ model max) / watt)",
    color="white",
)
ax.set_xlabel("Configuration Parameter", color="white")
ax.set_title(
    "MoE Power Efficiency by Configuration Parameters",
    color="white",
    fontsize=14,
)

ax.tick_params(axis="y", colors="white")

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
legend = ax.legend(handles, colors.keys(),
                   facecolor="black", edgecolor="white")
for t in legend.get_texts():
    t.set_color("white")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

print(f"Saved {OUT_PNG}")
