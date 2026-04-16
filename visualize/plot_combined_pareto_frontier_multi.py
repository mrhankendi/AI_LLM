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


def plot_pareto_comparison(df, model_name):
    """Create Pareto comparison plot (Power Cap Only vs TP/BS Only vs All Sweeps)"""
    
    # Define metrics
    throughput_col = 'Tokens/s'
    efficiency_col = 'Tokens/s/W (System)'
    gpu_efficiency_col = 'Tokens/s/W (GPU)'
    avg_gpu_power_col = 'Avg GPU Total Power (W)'
    avg_system_power_col = 'Avg System Power (W)'
    
    # Filter data
    df_tp4_bs64 = df[(df['Tensor Parallel Size'] == 4) & (df['Batch Size'] == 64)]
    df_pc400 = df[df['Power Cap (W)'] == 400]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: System Efficiency vs Throughput
    ax1 = axes[0, 0]
    
    if len(df_tp4_bs64) > 0:
        pareto_power = find_pareto_frontier(df_tp4_bs64, throughput_col, efficiency_col, maximize_both=True)
        pareto_power_sorted = pareto_power.sort_values(by=throughput_col)
        ax1.plot(pareto_power_sorted[throughput_col], pareto_power_sorted[efficiency_col],
                color='blue', linewidth=3, marker='s', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='Power Cap Sweep (TP4, BS64)', zorder=10)
    
    if len(df_pc400) > 0:
        pareto_tpbs = find_pareto_frontier(df_pc400, throughput_col, efficiency_col, maximize_both=True)
        pareto_tpbs_sorted = pareto_tpbs.sort_values(by=throughput_col)
        ax1.plot(pareto_tpbs_sorted[throughput_col], pareto_tpbs_sorted[efficiency_col],
                color='green', linewidth=3, marker='^', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='TP/BS Sweep (PC=400W)', zorder=10)
    
    pareto_all = find_pareto_frontier(df, throughput_col, efficiency_col, maximize_both=True)
    pareto_all_sorted = pareto_all.sort_values(by=throughput_col)
    ax1.plot(pareto_all_sorted[throughput_col], pareto_all_sorted[efficiency_col],
            color='red', linewidth=3, marker='o', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='All Sweeps', zorder=9)
    
    ax1.scatter(df[throughput_col], df[efficiency_col], 
               color='gray', alpha=0.2, s=50, label='All Configs')
    ax1.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
    ax1.set_title('System Efficiency vs Throughput', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: GPU Efficiency vs Throughput
    ax2 = axes[0, 1]
    
    if len(df_tp4_bs64) > 0:
        pareto_power = find_pareto_frontier(df_tp4_bs64, throughput_col, gpu_efficiency_col, maximize_both=True)
        pareto_power_sorted = pareto_power.sort_values(by=throughput_col)
        ax2.plot(pareto_power_sorted[throughput_col], pareto_power_sorted[gpu_efficiency_col],
                color='blue', linewidth=3, marker='s', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='Power Cap Sweep (TP4, BS64)', zorder=10)
    
    if len(df_pc400) > 0:
        pareto_tpbs = find_pareto_frontier(df_pc400, throughput_col, gpu_efficiency_col, maximize_both=True)
        pareto_tpbs_sorted = pareto_tpbs.sort_values(by=throughput_col)
        ax2.plot(pareto_tpbs_sorted[throughput_col], pareto_tpbs_sorted[gpu_efficiency_col],
                color='green', linewidth=3, marker='^', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='TP/BS Sweep (PC=400W)', zorder=10)
    
    pareto_all = find_pareto_frontier(df, throughput_col, gpu_efficiency_col, maximize_both=True)
    pareto_all_sorted = pareto_all.sort_values(by=throughput_col)
    ax2.plot(pareto_all_sorted[throughput_col], pareto_all_sorted[gpu_efficiency_col],
            color='red', linewidth=3, marker='o', markersize=10,
            markeredgecolor='black', markeredgewidth=2,
            label='All Sweeps', zorder=9)
    
    ax2.scatter(df[throughput_col], df[gpu_efficiency_col], 
               color='gray', alpha=0.2, s=50, label='All Configs')
    ax2.set_xlabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GPU Efficiency (Tokens/s/W)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Efficiency vs Throughput', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Throughput vs GPU Power
    ax3 = axes[1, 0]
    
    if len(df_tp4_bs64) > 0:
        pareto_points = []
        df_sorted = df_tp4_bs64.sort_values(by=avg_gpu_power_col)
        max_throughput = -np.inf
        for idx, row in df_sorted.iterrows():
            if row[throughput_col] >= max_throughput:
                pareto_points.append(idx)
                max_throughput = row[throughput_col]
        pareto_power = df_tp4_bs64.loc[pareto_points].sort_values(by=avg_gpu_power_col)
        ax3.plot(pareto_power[avg_gpu_power_col], pareto_power[throughput_col],
                color='blue', linewidth=3, marker='s', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='Power Cap Sweep (TP4, BS64)', zorder=10)
    
    if len(df_pc400) > 0:
        pareto_points = []
        df_sorted = df_pc400.sort_values(by=avg_gpu_power_col)
        max_throughput = -np.inf
        for idx, row in df_sorted.iterrows():
            if row[throughput_col] >= max_throughput:
                pareto_points.append(idx)
                max_throughput = row[throughput_col]
        pareto_tpbs = df_pc400.loc[pareto_points].sort_values(by=avg_gpu_power_col)
        ax3.plot(pareto_tpbs[avg_gpu_power_col], pareto_tpbs[throughput_col],
                color='green', linewidth=3, marker='^', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='TP/BS Sweep (PC=400W)', zorder=10)
    
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
            label='All Sweeps', zorder=9)
    
    ax3.scatter(df[avg_gpu_power_col], df[throughput_col], 
               color='gray', alpha=0.2, s=50, label='All Configs')
    ax3.set_xlabel('Avg GPU Total Power (W)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput vs GPU Power', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Throughput vs System Power
    ax4 = axes[1, 1]
    
    if len(df_tp4_bs64) > 0:
        pareto_points = []
        df_sorted = df_tp4_bs64.sort_values(by=avg_system_power_col)
        max_throughput = -np.inf
        for idx, row in df_sorted.iterrows():
            if row[throughput_col] >= max_throughput:
                pareto_points.append(idx)
                max_throughput = row[throughput_col]
        pareto_power = df_tp4_bs64.loc[pareto_points].sort_values(by=avg_system_power_col)
        ax4.plot(pareto_power[avg_system_power_col], pareto_power[throughput_col],
                color='blue', linewidth=3, marker='s', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='Power Cap Sweep (TP4, BS64)', zorder=10)
    
    if len(df_pc400) > 0:
        pareto_points = []
        df_sorted = df_pc400.sort_values(by=avg_system_power_col)
        max_throughput = -np.inf
        for idx, row in df_sorted.iterrows():
            if row[throughput_col] >= max_throughput:
                pareto_points.append(idx)
                max_throughput = row[throughput_col]
        pareto_tpbs = df_pc400.loc[pareto_points].sort_values(by=avg_system_power_col)
        ax4.plot(pareto_tpbs[avg_system_power_col], pareto_tpbs[throughput_col],
                color='green', linewidth=3, marker='^', markersize=10,
                markeredgecolor='black', markeredgewidth=2,
                label='TP/BS Sweep (PC=400W)', zorder=10)
    
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
            label='All Sweeps', zorder=9)
    
    ax4.scatter(df[avg_system_power_col], df[throughput_col], 
               color='gray', alpha=0.2, s=50, label='All Configs')
    ax4.set_xlabel('Avg System Power (W)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax4.set_title('Throughput vs System Power', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'{model_name} - Pareto Frontier Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = f'pareto_comparison_{model_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# Main execution: Find and process all summary CSV files
if __name__ == "__main__":
    # Find all summary CSV files matching the pattern
    csv_pattern = r"output\*_hellaswag_tp_powercap_summary_fixed_duration.csv"
    csv_files = glob.glob(csv_pattern)
    
    if len(csv_files) == 0:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        print("Please check the output directory and file naming pattern.")
    else:
        print(f"Found {len(csv_files)} CSV file(s) to process:")
        for f in csv_files:
            print(f"  - {f}")
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                print(f"\n{'='*90}")
                print(f"Processing: {csv_file}")
                print(f"{'='*90}")
                
                df = pd.read_csv(csv_file)
                model_name = Path(csv_file).stem.replace('_hellaswag_tp_powercap_summary_fixed_duration', '')
                
                # Generate the comparison plot
                plot_pareto_comparison(df, model_name)
                
            except Exception as e:
                print(f"\nError processing {csv_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*90}")
        print(f"Completed processing all {len(csv_files)} file(s)")
        print(f"{'='*90}")
