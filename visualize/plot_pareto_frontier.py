import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

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


# Read the CSV file
csv_file = r"output\mistralai_Mistral-7B-Instruct-v0.3_hellaswag_tp_powercap_summary_fixed_duration.csv"
df = pd.read_csv(csv_file)

# Extract model name from filename
model_name = Path(csv_file).stem.replace('_hellaswag_tp_powercap_summary_fixed_duration', '')

# Define metrics for Pareto analysis
throughput_col = 'Tokens/s'
efficiency_col = 'Tokens/s/W (System)'  # System-level efficiency
gpu_efficiency_col = 'Tokens/s/W (GPU)'  # GPU-level efficiency
avg_gpu_power_col = 'Avg GPU Total Power (W)'
avg_system_power_col = 'Avg System Power (W)'

# Create figure with subplots - now 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Color mapping by Tensor Parallel Size
tp_sizes = sorted(df['Tensor Parallel Size'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(tp_sizes)))
tp_color_map = dict(zip(tp_sizes, colors))

# Plot 1: System Efficiency vs Throughput
ax1 = axes[0, 0]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    
    # Plot all points
    ax1.scatter(df_tp[throughput_col], df_tp[efficiency_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')
    
    # Find and plot Pareto frontier for this TP size
    pareto_df = find_pareto_frontier(df_tp, throughput_col, efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)
    
    ax1.plot(pareto_df_sorted[throughput_col], pareto_df_sorted[efficiency_col],
            color=tp_color_map[tp_size], linewidth=2.5, marker='o', 
            markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Annotate Pareto points with batch size and power cap
    for idx, row in pareto_df.iterrows():
        ax1.annotate(f"BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                    (row[throughput_col], row[efficiency_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=tp_color_map[tp_size], alpha=0.2))

ax1.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('System Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
ax1.set_title('Pareto Frontier: System Efficiency vs Throughput', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: GPU Efficiency vs Throughput
ax2 = axes[0, 1]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    
    # Plot all points
    ax2.scatter(df_tp[throughput_col], df_tp[gpu_efficiency_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')
    
    # Find and plot Pareto frontier for this TP size
    pareto_df = find_pareto_frontier(df_tp, throughput_col, gpu_efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)
    
    ax2.plot(pareto_df_sorted[throughput_col], pareto_df_sorted[gpu_efficiency_col],
            color=tp_color_map[tp_size], linewidth=2.5, marker='o', 
            markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Annotate Pareto points
    for idx, row in pareto_df.iterrows():
        ax2.annotate(f"BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                    (row[throughput_col], row[gpu_efficiency_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=tp_color_map[tp_size], alpha=0.2))

ax2.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('GPU Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
ax2.set_title('Pareto Frontier: GPU Efficiency vs Throughput', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Throughput vs GPU Power (minimize power, maximize throughput)
ax3 = axes[1, 0]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    
    # Plot all points
    ax3.scatter(df_tp[avg_gpu_power_col], df_tp[throughput_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')
    
    # For this plot, we want to maximize throughput while minimizing power
    # So we find Pareto frontier where moving right (more power) must increase throughput
    pareto_points = []
    df_tp_sorted = df_tp.sort_values(by=avg_gpu_power_col)
    max_throughput = -np.inf
    
    for idx, row in df_tp_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    
    pareto_df = df_tp.loc[pareto_points].sort_values(by=avg_gpu_power_col)
    
    ax3.plot(pareto_df[avg_gpu_power_col], pareto_df[throughput_col],
            color=tp_color_map[tp_size], linewidth=2.5, marker='o', 
            markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Annotate Pareto points
    for idx, row in pareto_df.iterrows():
        ax3.annotate(f"BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                    (row[avg_gpu_power_col], row[throughput_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=tp_color_map[tp_size], alpha=0.2))

ax3.set_xlabel('Avg GPU Total Power (W)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax3.set_title('Pareto Frontier: Throughput vs GPU Power', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# Plot 4: Throughput vs System Power (minimize power, maximize throughput)
ax4 = axes[1, 1]
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    
    # Plot all points
    ax4.scatter(df_tp[avg_system_power_col], df_tp[throughput_col], 
               color=tp_color_map[tp_size], alpha=0.4, s=100,
               label=f'TP={tp_size}')
    
    # Find Pareto frontier
    pareto_points = []
    df_tp_sorted = df_tp.sort_values(by=avg_system_power_col)
    max_throughput = -np.inf
    
    for idx, row in df_tp_sorted.iterrows():
        if row[throughput_col] >= max_throughput:
            pareto_points.append(idx)
            max_throughput = row[throughput_col]
    
    pareto_df = df_tp.loc[pareto_points].sort_values(by=avg_system_power_col)
    
    ax4.plot(pareto_df[avg_system_power_col], pareto_df[throughput_col],
            color=tp_color_map[tp_size], linewidth=2.5, marker='o', 
            markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Annotate Pareto points
    for idx, row in pareto_df.iterrows():
        ax4.annotate(f"BS={int(row['Batch Size'])}\nPC={int(row['Power Cap (W)'])}W",
                    (row[avg_system_power_col], row[throughput_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=tp_color_map[tp_size], alpha=0.2))

ax4.set_xlabel('Avg System Power (W)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
ax4.set_title('Pareto Frontier: Throughput vs System Power', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.suptitle(f'{model_name} - Pareto Frontiers', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the figure
output_file = f'pareto_frontier_{model_name}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved Pareto frontier plot to: {output_file}")

plt.show()

# Print Pareto-optimal configurations
print("\n" + "="*80)
print("PARETO-OPTIMAL CONFIGURATIONS (System Efficiency)")
print("="*80)
for tp_size in tp_sizes:
    df_tp = df[df['Tensor Parallel Size'] == tp_size]
    pareto_df = find_pareto_frontier(df_tp, throughput_col, efficiency_col, maximize_both=True)
    pareto_df_sorted = pareto_df.sort_values(by=throughput_col)
    
    print(f"\nTensor Parallel Size = {tp_size}:")
    print("-" * 80)
    for idx, row in pareto_df_sorted.iterrows():
        print(f"  BS={int(row['Batch Size']):2d}, Power Cap={int(row['Power Cap (W)']):3d}W: "
              f"{row[throughput_col]:7.2f} tokens/s, {row[efficiency_col]:.4f} tokens/s/W")
