#!/usr/bin/env python3
"""
MoE configuration analysis & policy figures.

- Loads moe_gpu_profile.xlsx
- Computes normalized efficiency
- Batch and power-cap 'cliff ratios'
- Efficiency regret for suboptimal batches
- Pareto frontier (batch, cap, efficiency)
- Batch-policy and power-cap-policy figures with confidence bands
- A small runtime decision helper for batch/cap/TP

Figures (all PNG) are written to ./figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

###############################################################################
# CONFIG
###############################################################################

DATA_PATH = Path("output/moe_gpu_profile.xlsx")
OUT_DIR = Path("figures")

# Batch / cap regions for cliff-ratio calculations
BATCH_EARLY_MAX = 4      # "cliff" region upper batch
BATCH_LATE_MIN = 32      # late region lower batch
BATCH_LATE_MAX = 64      # late region upper batch (usually saturation)

CAP_EARLY_MAX = 150      # "cliff" region upper cap (W)
CAP_LATE_MIN = 250       # late region lower cap (W)
CAP_LATE_MAX = 400       # late region upper cap (W)

###############################################################################
# LOADING & BASE METRICS
###############################################################################


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Total GPU power and raw efficiency
    power_cols = [c for c in df.columns
                  if c.startswith("avg_GPU") and c.endswith("_Power_W")]
    df["total_power_W"] = df[power_cols].sum(axis=1)
    df["tokens_per_watt"] = df["avg_Tokens_per_s"] / df["total_power_W"]

    # Per-model-max throughput normalization
    df["model_max_tokens"] = df.groupby("model")["avg_Tokens_per_s"].transform("max")
    df["norm_tokens"] = df["avg_Tokens_per_s"] / df["model_max_tokens"]
    df["norm_tokens_per_watt"] = df["norm_tokens"] / df["total_power_W"]

    return df


###############################################################################
# CLIFF RATIOS
###############################################################################


def _safe_pick(series: pd.Series, target_value):
    """Pick the value at target_value if present, else closest by index."""
    series = series.dropna()
    if target_value in series.index:
        return series.loc[target_value]
    # pick closest x in index numerically
    idx = min(series.index, key=lambda x: abs(x - target_value))
    return series.loc[idx]


def compute_cliff_ratio(
    df: pd.DataFrame,
    param: str,
    early_max,
    late_min,
    late_max,
    metric: str = "norm_tokens_per_watt",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Formalization:

      For each model m:

        Let f_m(p) be mean efficiency over all other dims at parameter value p.

        Let p0  = min(p) over that model
        Let pe  = early_max (or closest available)
        Let ps  = late_max (or closest available, interpreted as saturation point)

        baseline   = f_m(p0)
        early_eff  = f_m(pe)
        sat_eff    = f_m(ps)

        total_gain = sat_eff - baseline
        cliff_gain = early_eff - baseline
        tail_gain  = sat_eff - early_eff

        cliff_share = cliff_gain / total_gain
        tail_share  = tail_gain / total_gain
        cliff_ratio = cliff_gain / tail_gain

      We then summarize mean ± std of these per-model ratios.
    """
    rows = []

    for model, g in df.groupby("model"):
        # Average over all other dimensions for stability
        eff_by_param = (
            g.groupby(param)[metric]
             .mean()
             .sort_index()
        )

        if eff_by_param.empty:
            continue

        p0 = eff_by_param.index.min()
        baseline = eff_by_param.loc[p0]

        early_eff = _safe_pick(eff_by_param, early_max)
        late_min_eff = _safe_pick(eff_by_param, late_min)
        sat_eff = _safe_pick(eff_by_param, late_max)

        total_gain = sat_eff - baseline
        cliff_gain = early_eff - baseline
        tail_gain = sat_eff - early_eff

        # Guard against numerical issues
        if total_gain <= 0 or tail_gain <= 0:
            continue

        cliff_share = cliff_gain / total_gain
        tail_share = tail_gain / total_gain
        cliff_ratio = cliff_gain / tail_gain

        rows.append(
            dict(
                model=model,
                p0=p0,
                baseline=baseline,
                early_eff=early_eff,
                late_min_eff=late_min_eff,
                sat_eff=sat_eff,
                total_gain=total_gain,
                cliff_gain=cliff_gain,
                tail_gain=tail_gain,
                cliff_share=cliff_share,
                tail_share=tail_share,
                cliff_ratio=cliff_ratio,
            )
        )

    res = pd.DataFrame(rows)

    summary = {}
    if not res.empty:
        for k in ["cliff_share", "tail_share", "cliff_ratio"]:
            summary[k + "_mean"] = res[k].mean()
            summary[k + "_std"] = res[k].std()

    return res, summary


###############################################################################
# EFFICIENCY REGRET
###############################################################################


def compute_batch_regret(
    df: pd.DataFrame,
    metric: str = "norm_tokens_per_watt",
) -> pd.DataFrame:
    """
    For each model, find its best batch (over all caps/TPs).
    Regret for batch b is:

      regret(m, b) = (eff_opt(m) - eff(m, b)) / eff_opt(m)

    We average regret across models for each batch.
    """
    eff = (
        df.groupby(["model", "batch"])[metric]
          .mean()
          .reset_index()
    )

    eff["model_best"] = eff.groupby("model")[metric].transform("max")
    eff["regret"] = (eff["model_best"] - eff[metric]) / eff["model_best"]

    regret_summary = (
        eff.groupby("batch")["regret"]
           .agg(["mean", "std", "count"])
           .reset_index()
    )
    return regret_summary


###############################################################################
# PARETO FRONTIER (batch, cap, efficiency)
###############################################################################


def pareto_frontier(
    df: pd.DataFrame,
    tp: int,
    metric: str = "norm_tokens_per_watt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pareto frontier for (batch, cap_w, efficiency) at fixed TP.

    Objectives:
      - maximize efficiency
      - minimize batch
      - minimize cap_w

    We aggregate efficiency over models for each (batch, cap_w).
    """
    agg = (
        df[df["tp"] == tp]
        .groupby(["batch", "cap_w"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "eff"})
    )

    if agg.empty:
        return agg, agg

    dominated = np.zeros(len(agg), dtype=bool)

    for i, row_i in agg.iterrows():
        if dominated[i]:
            continue
        for j, row_j in agg.iterrows():
            if i == j:
                continue
            # j dominates i if:
            #   eff_j >= eff_i AND batch_j <= batch_i AND cap_j <= cap_i
            #   with at least one strict
            cond1 = row_j["eff"] >= row_i["eff"]
            cond2 = row_j["batch"] <= row_i["batch"]
            cond3 = row_j["cap_w"] <= row_i["cap_w"]
            strict = (
                (row_j["eff"] > row_i["eff"])
                or (row_j["batch"] < row_i["batch"])
                or (row_j["cap_w"] < row_i["cap_w"])
            )
            if cond1 and cond2 and cond3 and strict:
                dominated[i] = True
                break

    frontier = agg[~dominated].copy()
    frontier = frontier.sort_values(["cap_w", "batch", "eff"], ascending=[True, True, False])
    return agg, frontier


###############################################################################
# DECISION POLICY (RUNTIME-TREE STYLE)
###############################################################################


def build_global_batch_default(regret_summary: pd.DataFrame, threshold: float = 0.05) -> int:
    """
    Pick a global default batch that keeps mean regret below a threshold.
    If several batches satisfy, choose the smallest (latency-friendlier).
    """
    ok = regret_summary[regret_summary["mean"] <= threshold]
    if ok.empty:
        # fall back to batch with minimal mean regret
        return int(regret_summary.loc[regret_summary["mean"].idxmin(), "batch"])
    return int(ok.sort_values("batch")["batch"].iloc[0])


def recommend_config(
    df: pd.DataFrame,
    latency_sensitivity: str,
    power_budget_W: int,
    throughput_target: str,
    metric: str = "norm_tokens_per_watt",
) -> Dict[str, int]:
    """
    A simple runtime decision tree using the analyses above.

    latency_sensitivity: 'low', 'medium', 'high'
    power_budget_W: approximate per-GPU cap budget
    throughput_target: 'low', 'medium', 'high'
    """
    # 1. Batch policy via regret
    regret_summary = compute_batch_regret(df, metric=metric)
    default_batch = build_global_batch_default(regret_summary, threshold=0.05)

    # For 'high' latency sensitivity, move towards smaller batch until regret explodes
    if latency_sensitivity == "high":
        candidate = regret_summary.sort_values("batch")  # small to large
        # pick smallest batch with regret <= 2 * default batch regret
        base_regret = regret_summary.loc[regret_summary["batch"] == default_batch, "mean"].iloc[0]
        cand = candidate[candidate["mean"] <= 2 * base_regret]
        if not cand.empty:
            chosen_batch = int(cand["batch"].iloc[0])
        else:
            chosen_batch = default_batch
    elif latency_sensitivity == "low":
        # prefer default or even larger batches if regret continues to decline
        candidate = regret_summary[regret_summary["batch"] >= default_batch]
        chosen_batch = int(
            candidate.loc[candidate["mean"].idxmin(), "batch"]
        )
    else:
        # 'medium'
        chosen_batch = default_batch

    # 2. Power-cap choice: pick Pareto front under the given budget
    tps = sorted(df["tp"].unique())
    # crude throughput preference: use higher TP if throughput_target is 'high'
    if throughput_target == "high" and 4 in tps:
        chosen_tp = 4
    else:
        chosen_tp = min(tps)

    agg, frontier = pareto_frontier(df, chosen_tp, metric=metric)
    within_budget = frontier[frontier["cap_w"] <= power_budget_W]
    if within_budget.empty:
        # pick smallest cap on frontier
        chosen_cap = int(frontier["cap_w"].min())
    else:
        # among frontier points inside budget, maximize efficiency
        row = within_budget.loc[within_budget["eff"].idxmax()]
        chosen_cap = int(row["cap_w"])

    return dict(tp=chosen_tp, batch=chosen_batch, cap_w=chosen_cap)


###############################################################################
# PLOTTING HELPERS (DARK POLICY STYLE)
###############################################################################


def set_dark_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.facecolor": "#050608",
            "figure.facecolor": "#050608",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "legend.edgecolor": "white",
            "grid.color": "#333333",
        }
    )


def plot_batch_policy(df: pd.DataFrame, out_dir: Path, metric: str = "norm_tokens_per_watt"):
    """
    One-page batch policy figure:

      - Top: efficiency vs batch with 95% CI band (normalized efficiency).
      - Bottom: efficiency regret vs batch.
      - Title block includes cliff-ratio summary.
    """
    set_dark_style()

    # Aggregate efficiency by (model, batch) first, then across models
    eff_mb = (
        df.groupby(["model", "batch"])[metric]
          .mean()
          .reset_index()
    )
    g = eff_mb.groupby("batch")[metric]
    batch_stats = pd.DataFrame(
        {
            "batch": g.mean().index.values,
            "mean": g.mean().values,
            "std": g.std().values,
            "n": g.count().values,
        }
    )
    batch_stats["se"] = batch_stats["std"] / np.sqrt(batch_stats["n"])
    batch_stats["ci_low"] = batch_stats["mean"] - 1.96 * batch_stats["se"]
    batch_stats["ci_high"] = batch_stats["mean"] + 1.96 * batch_stats["se"]

    # Regret summary
    regret_summary = compute_batch_regret(df, metric=metric)

    # Cliff ratio for batches
    batch_cliff_df, batch_cliff_summary = compute_cliff_ratio(
        df, param="batch",
        early_max=BATCH_EARLY_MAX,
        late_min=BATCH_LATE_MIN,
        late_max=BATCH_LATE_MAX,
        metric=metric,
    )

    cliff_mean = batch_cliff_summary.get("cliff_share_mean", np.nan) * 100
    cliff_std = batch_cliff_summary.get("cliff_share_std", np.nan) * 100
    tail_mean = batch_cliff_summary.get("tail_share_mean", np.nan) * 100
    cliff_ratio_mean = batch_cliff_summary.get("cliff_ratio_mean", np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: efficiency vs batch with band ---
    ax = axes[0]
    x = batch_stats["batch"].values
    ax.plot(x, batch_stats["mean"], marker="o", label="Mean normalized efficiency")
    ax.fill_between(x, batch_stats["ci_low"], batch_stats["ci_high"], alpha=0.3)

    ax.set_ylabel("Normalized Efficiency\n((tokens/s ÷ model max) / W)")
    ax.grid(alpha=0.4)

    subtitle = (
        f"Early cliff (≤ batch {BATCH_EARLY_MAX}) captures "
        f"{cliff_mean:.1f} ± {cliff_std:.1f}% of total gain; "
        f"late region ({BATCH_LATE_MIN}-{BATCH_LATE_MAX}) ~{tail_mean:.1f}%.\n"
        f"Cliff-to-progression ratio ≈ {cliff_ratio_mean:.2f}×"
        if not np.isnan(cliff_mean)
        else "Cliff statistics unavailable (check batch coverage)."
    )
    ax.set_title("Batch Size Efficiency Progression (MoEs)", pad=15)
    ax.text(
        0.01,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    # --- Bottom: efficiency regret vs batch ---
    ax = axes[1]
    ax.errorbar(
        regret_summary["batch"],
        regret_summary["mean"] * 100,
        yerr=regret_summary["std"] * 100,
        fmt="-o",
        capsize=4,
    )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Efficiency Regret vs Per-Model Optimum (%)")
    ax.grid(alpha=0.4)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_policy_figure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved batch policy figure to {out_path}")


def plot_power_cap_policy_and_pareto(df: pd.DataFrame, out_dir: Path, metric: str = "norm_tokens_per_watt"):
    """
    Power-cap cliff analysis and Pareto frontier visualization.
    Creates one PNG per TP value.
    """
    set_dark_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cliff ratios over cap_w
    _, cap_cliff_summary = compute_cliff_ratio(
        df, param="cap_w",
        early_max=CAP_EARLY_MAX,
        late_min=CAP_LATE_MIN,
        late_max=CAP_LATE_MAX,
        metric=metric,
    )

    cliff_mean = cap_cliff_summary.get("cliff_share_mean", np.nan) * 100
    cliff_std = cap_cliff_summary.get("cliff_share_std", np.nan) * 100
    tail_mean = cap_cliff_summary.get("tail_share_mean", np.nan) * 100
    cliff_ratio_mean = cap_cliff_summary.get("cliff_ratio_mean", np.nan)

    for tp in sorted(df["tp"].unique()):
        agg, frontier = pareto_frontier(df, tp, metric=metric)
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        sc = ax.scatter(
            agg["cap_w"],
            agg["batch"],
            c=agg["eff"],
            cmap="viridis",
            s=60,
            alpha=0.8,
            label="Configs",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Normalized Efficiency ((tokens/s ÷ model max)/W)")

        # Frontier overlay
        ax.plot(
            frontier["cap_w"],
            frontier["batch"],
            "-o",
            color="orange",
            linewidth=2.5,
            markersize=6,
            label="Pareto Frontier",
        )

        ax.set_xlabel("Power Cap (W)")
        ax.set_ylabel("Batch Size")
        ax.grid(alpha=0.4)
        ax.set_title(
            f"Batch–Power Cap Pareto Frontier (TP = {tp})\n"
            f"Early cap cliff (≤{CAP_EARLY_MAX}W) captures {cliff_mean:.1f} ± {cliff_std:.1f}% of gain; "
            f"late region ({CAP_LATE_MIN}-{CAP_LATE_MAX}W) ~{tail_mean:.1f}% — ratio ≈ {cliff_ratio_mean:.2f}×"
            if not np.isnan(cliff_mean)
            else f"Batch–Power Cap Pareto Frontier (TP = {tp})"
        )
        ax.legend(loc="upper left")

        out_path = out_dir / f"cap_pareto_tp{tp}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved power-cap Pareto figure to {out_path}")


###############################################################################
# MAIN
###############################################################################


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare(DATA_PATH)

    print("[INFO] Models in dataset:", sorted(df["model"].unique()))
    print("[INFO] TPs in dataset:", sorted(df["tp"].unique()))
    print("[INFO] Batches:", sorted(df["batch"].unique()))
    print("[INFO] Caps:", sorted(df["cap_w"].unique()))

    # Figures
    plot_batch_policy(df, OUT_DIR, metric="norm_tokens_per_watt")
    plot_power_cap_policy_and_pareto(df, OUT_DIR, metric="norm_tokens_per_watt")

    # Example runtime decision recommendation
    example_policy = recommend_config(
        df,
        latency_sensitivity="medium",
        power_budget_W=250,
        throughput_target="high",
        metric="norm_tokens_per_watt",
    )
    print("[INFO] Example runtime recommendation:", example_policy)


if __name__ == "__main__":
    main()
