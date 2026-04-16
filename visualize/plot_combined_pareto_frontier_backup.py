import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def find_pareto_frontier(df, x_col, y_col, maximize_both=True):
    """
    Find the Pareto frontier points.
    
    Args:
        df: DataFrame with the data
        x_col: Column name for x-axis (e.g., 'Tokens/s')
        y_col: Column name for y-axis (e.g., 'Tokens/s/W (System)')
        maximize_both: If True, both objectives are maximized
    
    Returns:
        DataFrame with only Pareto-optimal points
    """
    # Remove NaN values
    df_clean = df[[x_col, y_col]].dropna()
    
    # Sort by x value
    df_sorted = df_clean.sort_values(by=x_col)
    
    pareto_points = []
    max_y = -np.inf if maximize_both else np.inf
    
    for idx, row in df_sorted.iterrows():
        y_val = row[y_col]
        if maximize_both:
            if y_val >= max_y:
                pareto_points.append(idx)
                max_y = y_val
        else:
            if y_val <= max_y:
                pareto_points.append(idx)
                max_y = y_val
    
    return df.loc[pareto_points]


def process_model(csv_file):
    """Process a single model's CSV file and generate all plots"""
    print(f"\n{'='*90}")
    print(f"Processing: {csv_file}")
    print(f"{'='*90}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract model name from filename
    model_name = Path(csv_file).stem.replace('_hellaswag_tp_powercap_summary_fixed_duration', '')

    # Define metrics for Pareto analysis
    throughput_col = 'Tokens/s'
    efficiency_col = 'Tokens/s/W (System)'  # System-level efficiency
    gpu_efficiency_col = 'Tokens/s/W (GPU)'  # GPU-level efficiency
    avg_gpu_power_col = 'Avg GPU Total Power (W)'
    avg_system_power_col = 'Avg System Power (W)'

    # Create figure with subplots - 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Color mapping by Tensor Parallel Size (for scatter points)
    tp_sizes = sorted(df['Tensor Parallel Size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tp_sizes)))
    tp_color_map = dict(zip(tp_sizes, colors))

    # Color for combined Pareto frontier
    pareto_color = 'red'
    pareto_linewidth = 3.0

    # Filter data for TP4, BS64 power sweep analysis
    df_tp4_bs64 = df[(df['Tensor Parallel Size'] == 4) & (df['Batch Size'] == 64)]

    # Plot 1: System Efficiency vs Throughput
    ax1 = axes[0, 0]
    for tp_size in tp_sizes:
        df_tp = df[df['Tensor Parallel Size'] == tp_size]
        ax1.scatter(df_tp[throughput_col], df_tp[efficiency_col], 
                   color=tp_color_map[tp_size], alpha=0.4, s=100,
                   label=f'TP={tp_size}')

    # Find and plot combined Pareto frontier
    pareto_df = find_pareto_frontier(df, throughput_col, efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)

    ax1.plot(pareto_df_sorted[throughput_col], pareto_df_sorted[efficiency_col],
            color=pareto_color, linewidth=pareto_linewidth, marker='o', 
            markersize=10, markeredgecolor='black', markeredgewidth=2,
            label='Combined Pareto Frontier', zorder=10)

    # Annotate Pareto points with TP, batch size, and power cap
    for idx, row in pareto_df_sorted.iterrows():
        ax1.annotate(f"TP={int(row['Tensor Parallel Size'])}\nBS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                    (row[throughput_col], row[efficiency_col]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'))

    ax1.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
    ax1.set_title('Combined Pareto Frontier: System Efficiency vs Throughput', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: GPU Efficiency vs Throughput
    ax2 = axes[0, 1]
    for tp_size in tp_sizes:
        df_tp = df[df['Tensor Parallel Size'] == tp_size]
        ax2.scatter(df_tp[throughput_col], df_tp[gpu_efficiency_col], 
                   color=tp_color_map[tp_size], alpha=0.4, s=100,
                   label=f'TP={tp_size}')

# Find and plot combined Pareto frontier
pareto_df = find_pareto_frontier(df, throughput_col, gpu_efficiency_col, maximize_both=True)
pareto_df_sorted = pareto_df.sort_values(by=throughput_col)

ax2.plot(pareto_df_sorted[throughput_col], pareto_df_sorted[gpu_efficiency_col],
        color=pareto_color, linewidth=pareto_linewidth, marker='o', 
        markersize=10, markeredgecolor='black', markeredgewidth=2,
        label='Combined Pareto Frontier', zorder=10)

# Annotate Pareto points
for idx, row in pareto_df_sorted.iterrows():
    ax2.annotate(f"TP={int(row['Tensor Parallel Size'])}\nBS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[throughput_col], row[gpu_efficiency_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'))

ax2.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('GPU Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
ax2.set_title('Combined Pareto Frontier: GPU Efficiency vs Throughput', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Throughput vs GPU Power (minimize power, maximize throughput)
ax3 = axes[1, 0]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    ax3.scatter(df_tp[avg_gpu_power_col], df_tp[throughput_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')

# Find combined Pareto frontier for throughput vs power
pareto_points = []
df_sorted = df.sort_values(by=avg_gpu_power_col)
max_throughput = -np.inf

for idx, row in df_sorted.iterrows():
    if row[throughput_col] >= max_throughput:
        pareto_points.append(idx)
        max_throughput = row[throughput_col]

pareto_df = df.loc[pareto_points].sort_values(by=avg_gpu_power_col)

ax3.plot(pareto_df[avg_gpu_power_col], pareto_df[throughput_col],
        color=pareto_color, linewidth=pareto_linewidth, marker='o', 
        markersize=10, markeredgecolor='black', markeredgewidth=2,
        label='Combined Pareto Frontier', zorder=10)

# Annotate Pareto points
for idx, row in pareto_df.iterrows():
    ax3.annotate(f"TP={int(row['Tensor Parallel Size'])}\nBS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[avg_gpu_power_col], row[throughput_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'))

ax3.set_xlabel('Avg GPU Total Power (W)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax3.set_title('Combined Pareto Frontier: Throughput vs GPU Power', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Throughput vs System Power (minimize power, maximize throughput)
ax4 = axes[1, 1]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    ax4.scatter(df_tp[avg_system_power_col], df_tp[throughput_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')

# Find combined Pareto frontier
pareto_points = []
df_sorted = df.sort_values(by=avg_system_power_col)
max_throughput = -np.inf

for idx, row in df_sorted.iterrows():
    if row[throughput_col] >= max_throughput:
        pareto_points.append(idx)
        max_throughput = row[throughput_col]

pareto_df = df.loc[pareto_points].sort_values(by=avg_system_power_col)

ax4.plot(pareto_df[avg_system_power_col], pareto_df[throughput_col],
        color=pareto_color, linewidth=pareto_linewidth, marker='o', 
        markersize=10, markeredgecolor='black', markeredgewidth=2,
        label='Combined Pareto Frontier', zorder=10)

# Annotate Pareto points
for idx, row in pareto_df.iterrows():
    ax4.annotate(f"TP={int(row['Tensor Parallel Size'])}\nBS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[avg_system_power_col], row[throughput_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'))

ax4.set_xlabel('Avg System Power (W)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax4.set_title('Combined Pareto Frontier: Throughput vs System Power', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f'{model_name} - Combined Pareto Frontiers (All TP Sizes)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the figure
output_file = f'combined_pareto_frontier_{model_name}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved combined Pareto frontier plot to: {output_file}")

plt.show()

# ============================================================================
# NEW FIGURE: TP4, BS64 Power Cap Sweep
# ============================================================================
if len(df_tp4_bs64) > 0:
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color gradient for power caps
    power_caps = sorted(df_tp4_bs64['Power Cap (W)'].unique())
    power_colors = plt.cm.viridis(np.linspace(0, 1, len(power_caps)))
    power_color_map = dict(zip(power_caps, power_colors))
    
    # Plot 1: System Efficiency vs Throughput (TP4, BS64)
    ax1 = axes2[0, 0]
    for pc in power_caps:
        df_pc = df_tp4_bs64[df_tp4_bs64['Power Cap (W)'] == pc]
        ax1.scatter(df_pc[throughput_col], df_pc[efficiency_col],
                   color=power_color_map[pc], s=200, marker='o',
                   edgecolors='black', linewidths=2,
                   label=f'{int(pc)}W')
        # Annotate each point
        for idx, row in df_pc.iterrows():
            ax1.annotate(f"{int(row['Power Cap (W)'])}W",
                        (row[throughput_col], row[efficiency_col]),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=9, ha='center', fontweight='bold')
    
    # Connect points to show the sweep
    df_tp4_bs64_sorted = df_tp4_bs64.sort_values(by='Power Cap (W)')
    ax1.plot(df_tp4_bs64_sorted[throughput_col], df_tp4_bs64_sorted[efficiency_col],
            color='black', linewidth=2, linestyle='--', alpha=0.5, zorder=1)
    
    ax1.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
    ax1.set_title('TP4, BS64: System Efficiency vs Throughput\n(Power Cap Sweep)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, title='Power Cap', ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: GPU Efficiency vs Throughput (TP4, BS64)
    ax2 = axes2[0, 1]
    for pc in power_caps:
        df_pc = df_tp4_bs64[df_tp4_bs64['Power Cap (W)'] == pc]
        ax2.scatter(df_pc[throughput_col], df_pc[gpu_efficiency_col],
                   color=power_color_map[pc], s=200, marker='o',
                   edgecolors='black', linewidths=2,
                   label=f'{int(pc)}W')
        for idx, row in df_pc.iterrows():
            ax2.annotate(f"{int(row['Power Cap (W)'])}W",
                        (row[throughput_col], row[gpu_efficiency_col]),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=9, ha='center', fontweight='bold')
    
    ax2.plot(df_tp4_bs64_sorted[throughput_col], df_tp4_bs64_sorted[gpu_efficiency_col],
            color='black', linewidth=2, linestyle='--', alpha=0.5, zorder=1)
    
    ax2.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GPU Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
    ax2.set_title('TP4, BS64: GPU Efficiency vs Throughput\n(Power Cap Sweep)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, title='Power Cap', ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Throughput vs GPU Power (TP4, BS64)
    ax3 = axes2[1, 0]
    for pc in power_caps:
        df_pc = df_tp4_bs64[df_tp4_bs64['Power Cap (W)'] == pc]
        ax3.scatter(df_pc[avg_gpu_power_col], df_pc[throughput_col],
                   color=power_color_map[pc], s=200, marker='o',
                   edgecolors='black', linewidths=2,
                   label=f'{int(pc)}W')
        for idx, row in df_pc.iterrows():
            ax3.annotate(f"{int(row['Power Cap (W)'])}W",
                        (row[avg_gpu_power_col], row[throughput_col]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=9, va='center', fontweight='bold')
    
    ax3.plot(df_tp4_bs64_sorted[avg_gpu_power_col], df_tp4_bs64_sorted[throughput_col],
            color='black', linewidth=2, linestyle='--', alpha=0.5, zorder=1)
    
    ax3.set_xlabel('Avg GPU Total Power (W)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax3.set_title('TP4, BS64: Throughput vs GPU Power\n(Power Cap Sweep)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, title='Power Cap', ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Throughput vs System Power (TP4, BS64)
    ax4 = axes2[1, 1]
    for pc in power_caps:
        df_pc = df_tp4_bs64[df_tp4_bs64['Power Cap (W)'] == pc]
        ax4.scatter(df_pc[avg_system_power_col], df_pc[throughput_col],
                   color=power_color_map[pc], s=200, marker='o',
                   edgecolors='black', linewidths=2,
                   label=f'{int(pc)}W')
        for idx, row in df_pc.iterrows():
            ax4.annotate(f"{int(row['Power Cap (W)'])}W",
                        (row[avg_system_power_col], row[throughput_col]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=9, va='center', fontweight='bold')
    
    ax4.plot(df_tp4_bs64_sorted[avg_system_power_col], df_tp4_bs64_sorted[throughput_col],
            color='black', linewidth=2, linestyle='--', alpha=0.5, zorder=1)
    
    ax4.set_xlabel('Avg System Power (W)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax4.set_title('TP4, BS64: Throughput vs System Power\n(Power Cap Sweep)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, title='Power Cap', ncol=2)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'{model_name} - TP4, BS64 Power Cap Sweep Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save the figure
    output_file2 = f'power_sweep_tp4_bs64_{model_name}.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Saved TP4 BS64 power sweep plot to: {output_file2}")
    
    plt.show()
    
    # Print power cap sweep data
    print("\n" + "="*90)
    print("TP4, BS64 POWER CAP SWEEP RESULTS")
    print("="*90)
    df_tp4_bs64_sorted = df_tp4_bs64.sort_values(by='Power Cap (W)')
    for idx, row in df_tp4_bs64_sorted.iterrows():
        print(f"Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"Throughput={row[throughput_col]:7.2f} tokens/s, "
              f"Sys Eff={row[efficiency_col]:.4f} tokens/s/W, "
              f"GPU Eff={row[gpu_efficiency_col]:.4f} tokens/s/W, "
              f"GPU Power={row[avg_gpu_power_col]:.2f}W, "
              f"Sys Power={row[avg_system_power_col]:.2f}W")
else:
    print("\nWarning: No data found for TP4, BS64 configuration")

# ============================================================================
# NEW FIGURE: Comparison of Pareto Frontiers - Power Cap Only vs All Sweeps
# ============================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))

# Filter data for TP/BS sweep at constant power cap (400W)
df_pc400 = df[df['Power Cap (W)'] == 400]

# Plot 1: System Efficiency vs Throughput - Comparison
ax1 = axes3[0, 0]

# Pareto with only power cap sweep (TP4, BS64, varying power caps)
if len(df_tp4_bs64) > 0:
    pareto_power_only = find_pareto_frontier(df_tp4_bs64, throughput_col, efficiency_col, maximize_both=True)
    pareto_power_only_sorted = pareto_power_only.sort_values(by=throughput_col)
    
    ax1.plot(pareto_power_only_sorted[throughput_col], pareto_power_only_sorted[efficiency_col],
            color='blue', linewidth=3, marker='s', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: Power Cap Sweep Only (TP4, BS64)', zorder=10)
    
    # Annotate power-only Pareto points
    for idx, row in pareto_power_only_sorted.iterrows():
        ax1.annotate(f"PC={int(row['Power Cap (W)'])}W",
                    (row[throughput_col], row[efficiency_col]),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=8, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Pareto with TP/BS sweep at constant power cap (400W)
if len(df_pc400) > 0:
    pareto_tp_bs_only = find_pareto_frontier(df_pc400, throughput_col, efficiency_col, maximize_both=True)
    pareto_tp_bs_only_sorted = pareto_tp_bs_only.sort_values(by=throughput_col)
    
    ax1.plot(pareto_tp_bs_only_sorted[throughput_col], pareto_tp_bs_only_sorted[efficiency_col],
            color='green', linewidth=3, marker='^', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: TP/BS Sweep Only (PC=400W)', zorder=10)
    
    # Annotate TP/BS-only Pareto points
    for idx, row in pareto_tp_bs_only_sorted.iterrows():
        ax1.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}",
                    (row[throughput_col], row[efficiency_col]),
                    xytext=(-15, 8), textcoords='offset points',
                    fontsize=8, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Pareto with all configurations (all TP, BS, power caps)
pareto_all = find_pareto_frontier(df, throughput_col, efficiency_col, maximize_both=True)
pareto_all_sorted = pareto_all.sort_values(by=throughput_col)

ax1.plot(pareto_all_sorted[throughput_col], pareto_all_sorted[efficiency_col],
        color='red', linewidth=3, marker='o', markersize=10,
        markeredgecolor='black', markeredgewidth=2,
        label='Pareto: All Sweeps (TP, BS, Power Cap)', zorder=9)

# Annotate all-sweep Pareto points
for idx, row in pareto_all_sorted.iterrows():
    ax1.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[throughput_col], row[efficiency_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=7, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

# Plot all data points in background
ax1.scatter(df[throughput_col], df[efficiency_col], 
           color='gray', alpha=0.2, s=50, label='All Configurations')

ax1.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('System Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
ax1.set_title('System Efficiency vs Throughput\nPareto Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: GPU Efficiency vs Throughput - Comparison
ax2 = axes3[0, 1]

if len(df_tp4_bs64) > 0:
    pareto_power_only = find_pareto_frontier(df_tp4_bs64, throughput_col, gpu_efficiency_col, maximize_both=True)
    pareto_power_only_sorted = pareto_power_only.sort_values(by=throughput_col)
    
    ax2.plot(pareto_power_only_sorted[throughput_col], pareto_power_only_sorted[gpu_efficiency_col],
            color='blue', linewidth=3, marker='s', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: Power Cap Sweep Only (TP4, BS64)', zorder=10)
    
    for idx, row in pareto_power_only_sorted.iterrows():
        ax2.annotate(f"PC={int(row['Power Cap (W)'])}W",
                    (row[throughput_col], row[gpu_efficiency_col]),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=8, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

if len(df_pc400) > 0:
    pareto_tp_bs_only = find_pareto_frontier(df_pc400, throughput_col, gpu_efficiency_col, maximize_both=True)
    pareto_tp_bs_only_sorted = pareto_tp_bs_only.sort_values(by=throughput_col)
    
    ax2.plot(pareto_tp_bs_only_sorted[throughput_col], pareto_tp_bs_only_sorted[gpu_efficiency_col],
            color='green', linewidth=3, marker='^', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: TP/BS Sweep Only (PC=400W)', zorder=10)
    
    for idx, row in pareto_tp_bs_only_sorted.iterrows():
        ax2.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}",
                    (row[throughput_col], row[gpu_efficiency_col]),
                    xytext=(-15, 8), textcoords='offset points',
                    fontsize=8, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

pareto_all = find_pareto_frontier(df, throughput_col, gpu_efficiency_col, maximize_both=True)
pareto_all_sorted = pareto_all.sort_values(by=throughput_col)

ax2.plot(pareto_all_sorted[throughput_col], pareto_all_sorted[gpu_efficiency_col],
        color='red', linewidth=3, marker='o', markersize=10,
        markeredgecolor='black', markeredgewidth=2,
        label='Pareto: All Sweeps (TP, BS, Power Cap)', zorder=9)

for idx, row in pareto_all_sorted.iterrows():
    ax2.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[throughput_col], row[gpu_efficiency_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=7, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

ax2.scatter(df[throughput_col], df[gpu_efficiency_col], 
           color='gray', alpha=0.2, s=50, label='All Configurations')

ax2.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('GPU Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Efficiency vs Throughput\nPareto Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Throughput vs GPU Power - Comparison
ax3 = axes3[1, 0]

if len(df_tp4_bs64) > 0:
    # For power vs throughput, find Pareto frontier
    pareto_points = []
    df_tp4_bs64_sorted = df_tp4_bs64.sort_values(by=avg_gpu_power_col)
    max_throughput = -np.inf
    for idx, row in df_tp4_bs64_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_power_only = df_tp4_bs64.loc[pareto_points].sort_values(by=avg_gpu_power_col)
    
    ax3.plot(pareto_power_only[avg_gpu_power_col], pareto_power_only[throughput_col],
            color='blue', linewidth=3, marker='s', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: Power Cap Sweep Only (TP4, BS64)', zorder=10)
    
    for idx, row in pareto_power_only.iterrows():
        ax3.annotate(f"PC={int(row['Power Cap (W)'])}W",
                    (row[avg_gpu_power_col], row[throughput_col]),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=8, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

if len(df_pc400) > 0:
    pareto_points = []
    df_pc400_sorted = df_pc400.sort_values(by=avg_gpu_power_col)
    max_throughput = -np.inf
    for idx, row in df_pc400_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_tp_bs_only = df_pc400.loc[pareto_points].sort_values(by=avg_gpu_power_col)
    
    ax3.plot(pareto_tp_bs_only[avg_gpu_power_col], pareto_tp_bs_only[throughput_col],
            color='green', linewidth=3, marker='^', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: TP/BS Sweep Only (PC=400W)', zorder=10)
    
    for idx, row in pareto_tp_bs_only.iterrows():
        ax3.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}",
                    (row[avg_gpu_power_col], row[throughput_col]),
                    xytext=(-15, 8), textcoords='offset points',
                    fontsize=8, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

pareto_points = []
df_sorted = df.sort_values(by=avg_gpu_power_col)
max_throughput = -np.inf
for idx, row in df_sorted.iterrows():
    if row[throughput_col] >= max_throughput:
        pareto_points.append(idx)
        max_throughput = row[throughput_col]
pareto_all = df.loc[pareto_points].sort_values(by=avg_gpu_power_col)

ax3.plot(pareto_all[avg_gpu_power_col], pareto_all[throughput_col],
        color='red', linewidth=3, marker='o', markersize=10,
        markeredgecolor='black', markeredgewidth=2,
        label='Pareto: All Sweeps (TP, BS, Power Cap)', zorder=9)

for idx, row in pareto_all.iterrows():
    ax3.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[avg_gpu_power_col], row[throughput_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=7, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

ax3.scatter(df[avg_gpu_power_col], df[throughput_col], 
           color='gray', alpha=0.2, s=50, label='All Configurations')

ax3.set_xlabel('Avg GPU Total Power (W)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax3.set_title('Throughput vs GPU Power\nPareto Comparison', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Throughput vs System Power - Comparison
ax4 = axes3[1, 1]

if len(df_tp4_bs64) > 0:
    pareto_points = []
    df_tp4_bs64_sorted = df_tp4_bs64.sort_values(by=avg_system_power_col)
    max_throughput = -np.inf
    for idx, row in df_tp4_bs64_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_power_only = df_tp4_bs64.loc[pareto_points].sort_values(by=avg_system_power_col)
    
    ax4.plot(pareto_power_only[avg_system_power_col], pareto_power_only[throughput_col],
            color='blue', linewidth=3, marker='s', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: Power Cap Sweep Only (TP4, BS64)', zorder=10)
    
    for idx, row in pareto_power_only.iterrows():
        ax4.annotate(f"PC={int(row['Power Cap (W)'])}W",
                    (row[avg_system_power_col], row[throughput_col]),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=8, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

if len(df_pc400) > 0:
    pareto_points = []
    df_pc400_sorted = df_pc400.sort_values(by=avg_system_power_col)
    max_throughput = -np.inf
    for idx, row in df_pc400_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_tp_bs_only = df_pc400.loc[pareto_points].sort_values(by=avg_system_power_col)
    
    ax4.plot(pareto_tp_bs_only[avg_system_power_col], pareto_tp_bs_only[throughput_col],
            color='green', linewidth=3, marker='^', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='Pareto: TP/BS Sweep Only (PC=400W)', zorder=10)
    
    for idx, row in pareto_tp_bs_only.iterrows():
        ax4.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}",
                    (row[avg_system_power_col], row[throughput_col]),
                    xytext=(-15, 8), textcoords='offset points',
                    fontsize=8, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

pareto_points = []
df_sorted = df.sort_values(by=avg_system_power_col)
max_throughput = -np.inf
for idx, row in df_sorted.iterrows():
    if row[throughput_col] >= max_throughput:
        pareto_points.append(idx)
        max_throughput = row[throughput_col]
pareto_all = df.loc[pareto_points].sort_values(by=avg_system_power_col)

ax4.plot(pareto_all[avg_system_power_col], pareto_all[throughput_col],
        color='red', linewidth=3, marker='o', markersize=10,
        markeredgecolor='black', markeredgewidth=2,
        label='Pareto: All Sweeps (TP, BS, Power Cap)', zorder=9)

for idx, row in pareto_all.iterrows():
    ax4.annotate(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                (row[avg_system_power_col], row[throughput_col]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=7, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

ax4.scatter(df[avg_system_power_col], df[throughput_col], 
           color='gray', alpha=0.2, s=50, label='All Configurations')

ax4.set_xlabel('Avg System Power (W)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax4.set_title('Throughput vs System Power\nPareto Comparison', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f'{model_name} - Pareto Frontier Comparison: Single-Dimension vs All Sweeps', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the figure
output_file3 = f'pareto_comparison_{model_name}.png'
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"Saved Pareto comparison plot to: {output_file3}")

plt.show()

# Original summary output continues below

# Original summary output - now moved inside process_model function
    # Print Pareto-optimal configurations for each metric
    print("\n" + "="*90)
    print("COMBINED PARETO-OPTIMAL CONFIGURATIONS (System Efficiency vs Throughput)")
    print("="*90)
    pareto_df = find_pareto_frontier(df, throughput_col, efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)
    for idx, row in pareto_df_sorted.iterrows():
        print(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size']):2d}, "
              f"Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"{row[throughput_col]:7.2f} tokens/s, {row[efficiency_col]:.4f} tokens/s/W")

    print("\n" + "="*90)
    print("COMBINED PARETO-OPTIMAL CONFIGURATIONS (GPU Efficiency vs Throughput)")
    print("="*90)
    pareto_df = find_pareto_frontier(df, throughput_col, gpu_efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)
    for idx, row in pareto_df_sorted.iterrows():
        print(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size']):2d}, "
              f"Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"{row[throughput_col]:7.2f} tokens/s, {row[gpu_efficiency_col]:.4f} tokens/s/W")

    print("\n" + "="*90)
    print("COMBINED PARETO-OPTIMAL CONFIGURATIONS (Throughput vs GPU Power)")
    print("="*90)
    pareto_points = []
    df_sorted = df.sort_values(by=avg_gpu_power_col)
    max_throughput = -np.inf
    for idx, row in df_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_df = df.loc[pareto_points].sort_values(by=avg_gpu_power_col)
    for idx, row in pareto_df.iterrows():
        print(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size']):2d}, "
              f"Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"{row[throughput_col]:7.2f} tokens/s, {row[avg_gpu_power_col]:.2f}W GPU power")

    print("\n" + "="*90)
    print("COMBINED PARETO-OPTIMAL CONFIGURATIONS (Throughput vs System Power)")
    print("="*90)
    pareto_points = []
    df_sorted = df.sort_values(by=avg_system_power_col)
    max_throughput = -np.inf
    for idx, row in df_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    pareto_df = df.loc[pareto_points].sort_values(by=avg_system_power_col)
    for idx, row in pareto_df.iterrows():
        print(f"TP={int(row['Tensor Parallel Size'])}, BS={int(row['Batch Size']):2d}, "
              f"Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"{row[throughput_col]:7.2f} tokens/s, {row[avg_system_power_col]:.2f}W system power")
