import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. Load data & basic metrics
# -------------------------------------------------
FILE = "output/moe_gpu_profile.xlsx"   # adjust path if needed

df = pd.read_excel(FILE)

# total GPU power across all GPUs
power_cols = [c for c in df.columns
              if c.startswith("avg_GPU") and c.endswith("_Power_W")]
df["total_power_W"] = df[power_cols].sum(axis=1)

# raw efficiency
df["tokens_per_watt"] = df["avg_Tokens_per_s"] / df["total_power_W"]

# (optional) per-model max-normalized throughput
df["model_max_tokens"] = df.groupby("model")["avg_Tokens_per_s"].transform("max")
df["norm_tokens"] = df["avg_Tokens_per_s"] / df["model_max_tokens"]
df["norm_tokens_per_watt"] = df["norm_tokens"] / df["total_power_W"]

print("Models x TP present:")
print(df.groupby(["model", "tp"]).size(), "\n")

# -------------------------------------------------
# 2. Generic ROI computation
# -------------------------------------------------
def compute_roi(
    data: pd.DataFrame,
    param: str,
    base_val,
    tgt_val,
    use_normalized: bool = False,
):
    """
    Compare base_val -> tgt_val for given param ('tp', 'batch', 'cap_w', ...).

    Returns a dataframe with per-model:
      - throughput_gain_pct
      - efficiency_loss_pct
      - power_overhead_pct
      - roi_throughput_per_power_overhead
    """

    eff_col = "norm_tokens_per_watt" if use_normalized else "tokens_per_watt"

    # aggregate per model & param
    agg = (
        data.groupby(["model", param])
        .agg(
            tokens_per_s=("avg_Tokens_per_s", "mean"),
            eff=(eff_col, "mean"),
            power_W=("total_power_W", "mean"),
        )
        .reset_index()
    )

    base = agg[agg[param] == base_val].copy()
    tgt = agg[agg[param] == tgt_val].copy()

    base = base.rename(
        columns={
            "tokens_per_s": "tokens_base",
            "eff": "eff_base",
            "power_W": "power_base",
        }
    )
    tgt = tgt.rename(
        columns={
            "tokens_per_s": "tokens_tgt",
            "eff": "eff_tgt",
            "power_W": "power_tgt",
        }
    )

    merged = pd.merge(base, tgt, on="model", how="inner", suffixes=("_b", "_t"))

    # if models are missing either config, they drop out HERE
    # (this is why you currently see only 3 models)
    if merged.empty:
        print("No overlapping configs found for", param, base_val, "->", tgt_val)
        return merged

    # percentage metrics
    merged["throughput_gain_pct"] = (
        (merged["tokens_tgt"] - merged["tokens_base"]) / merged["tokens_base"] * 100
    )
    merged["efficiency_loss_pct"] = (
        (merged["eff_base"] - merged["eff_tgt"]) / merged["eff_base"] * 100
    )
    merged["power_overhead_pct"] = (
        (merged["power_tgt"] - merged["power_base"]) / merged["power_base"] * 100
    )

    # ROI: throughput gain per 1% power overhead
    merged["roi_throughput_per_power_overhead"] = (
        merged["throughput_gain_pct"] / merged["power_overhead_pct"]
    )

    return merged


# -------------------------------------------------
# 3. Classification into tiers (similar to your screenshot)
# -------------------------------------------------
def classify_models(
    roi_df: pd.DataFrame,
    eff_loss_threshold: float = 10.0,  # e.g., <10% efficiency loss is "ok"
    roi_threshold: float = 1.0,        # e.g., ≥1% throughput gain per 1% power
):
    """
    Simple rule-based tiering:
      - Tier A: efficiency loss <= threshold AND ROI >= roi_threshold
      - Tier B: efficiency loss <= threshold BUT ROI < roi_threshold
      - Tier C: efficiency loss > threshold (too inefficient)
    """

    def _tier(row):
        if row["efficiency_loss_pct"] <= eff_loss_threshold:
            if row["roi_throughput_per_power_overhead"] >= roi_threshold:
                return "Tier A (high ROI)"
            else:
                return "Tier B (low ROI)"
        else:
            return "Tier C (too much eff. loss)"

    roi_df["tier"] = roi_df.apply(_tier, axis=1)
    return roi_df


# -------------------------------------------------
# 4. Plot: Efficiency Loss vs Throughput ROI
# -------------------------------------------------
def plot_efficiency_vs_roi(
    roi_df: pd.DataFrame,
    title: str,
    filename_png: str,
):
    # dark theme
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    # colors per tier
    color_map = {
        "Tier A (high ROI)": "#4FC3F7",   # cyan-ish
        "Tier B (low ROI)": "#FFB74D",    # orange
        "Tier C (too much eff. loss)": "#EF5350",  # red
    }

    for tier, group in roi_df.groupby("tier"):
        ax.scatter(
            group["efficiency_loss_pct"],
            group["roi_throughput_per_power_overhead"],
            label=tier,
            s=60,
            edgecolor="white",
            linewidth=0.6,
            color=color_map.get(tier, "white"),
        )
        # annotate model names
        for _, row in group.iterrows():
            ax.text(
                row["efficiency_loss_pct"] + 0.3,
                row["roi_throughput_per_power_overhead"] + 0.3,
                row["model"],
                fontsize=8,
                color="white",
            )

    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.axhline(0, color="#444444", linewidth=0.8)

    ax.set_xlabel("Efficiency Loss (%)  (lower is better)", color="white")
    ax.set_ylabel("Throughput ROI (% gain per 1% power overhead)", color="white")
    ax.set_title(title, color="white", pad=12)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")

    legend = ax.legend(frameon=True, facecolor="#222222", edgecolor="#555555")
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(filename_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------
# 5. EXAMPLES
# -------------------------------------------------

# Example 1: TP ROI (TP=2 -> TP=4), using *normalized* efficiency
roi_tp = compute_roi(df, param="tp", base_val=2, tgt_val=4, use_normalized=True)

if not roi_tp.empty:
    roi_tp = classify_models(roi_tp, eff_loss_threshold=10.0, roi_threshold=1.0)
    print("TP ROI table:")
    print(roi_tp[[
        "model",
        "throughput_gain_pct",
        "efficiency_loss_pct",
        "power_overhead_pct",
        "roi_throughput_per_power_overhead",
        "tier",
    ]].round(2))

    plot_efficiency_vs_roi(
        roi_tp,
        title="Model-Specific Scaling Viability: TP2 → TP4",
        filename_png="roi_tp2_to_tp4.png",
    )

# Example 2: Batch ROI (e.g., batch 8 -> 32)
#   Change these values to whatever you want to compare.
roi_batch = compute_roi(df, param="batch", base_val=1, tgt_val=64, use_normalized=True)

if not roi_batch.empty:
    roi_batch = classify_models(roi_batch, eff_loss_threshold=10.0, roi_threshold=1.0)
    plot_efficiency_vs_roi(
        roi_batch,
        title="Scaling Viability: Batch 8 → 32",
        filename_png="roi_batch1_to_64.png",
    )

# Example 3: Power-cap ROI (e.g., 200W -> 300W)
roi_cap = compute_roi(df, param="cap_w", base_val=100, tgt_val=400, use_normalized=True)

if not roi_cap.empty:
    roi_cap = classify_models(roi_cap, eff_loss_threshold=10.0, roi_threshold=1.0)
    plot_efficiency_vs_roi(
        roi_cap,
        title="Scaling Viability: Cap 200W → 300W",
        filename_png="roi_cap100_to_400.png",
    )
