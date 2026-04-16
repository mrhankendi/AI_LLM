import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel("output/moe_gpu_profile.xlsx")

# Compute total GPU power
power_cols = [c for c in df.columns
              if c.startswith("avg_GPU") and c.endswith("_Power_W")]
df["total_power_W"] = df[power_cols].sum(axis=1)

# Tokens per watt efficiency
df["tokens_per_watt"] = df["avg_Tokens_per_s"] / df["total_power_W"]

# Keep only batch values of interest
batches = [1, 4, 8, 16, 32, 64]
df = df[df["batch"].isin(batches)]

# Aggregate: mean per model per batch
agg = (
    df.groupby(["model", "batch"], as_index=False)
      .agg(tokens_per_watt=("tokens_per_watt", "mean"))
)

# -----------------------------
# Plot
# -----------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(9, 6))

fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

for model, g in agg.groupby("model"):
    g = g.sort_values("batch")
    ax.plot(
        g["batch"],
        g["tokens_per_watt"],
        marker="o",
        linewidth=2,
        label=model
    )

ax.set_xscale("log", base=2)
ax.set_xticks(batches)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel("Batch Size", color="white")
ax.set_ylabel("Tokens per Watt (Efficiency)", color="white")
ax.set_title("Batch Size Efficiency Progression Across Models",
             color="white", pad=12)

ax.tick_params(colors="white")

ax.grid(axis="y", linestyle="--", alpha=0.3, color="#444444")

legend = ax.legend(
    frameon=True,
    facecolor="#222222",
    edgecolor="#444444",
    fontsize=9
)
for text in legend.get_texts():
    text.set_color("white")

fig.tight_layout()
fig.savefig("batch_efficiency_progression.png", dpi=300, bbox_inches="tight")
plt.show()
