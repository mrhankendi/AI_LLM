import matplotlib.pyplot as plt
import pandas as pd
import os

############################################
# Plots
############################################
def plot_metric(metric_col, ylabel, title, filename, df):
    """
    Plot a metric from the DataFrame grouped by Tensor Parallel Size and Power Cap.
    """
    plt.figure()
    for tp in sorted(df['Tensor Parallel Size'].unique()):
        for cap in sorted(df['Power Cap (W)'].unique()):
            sub_df = df[(df['Tensor Parallel Size'] == tp) & (df['Power Cap (W)'] == cap)]
            bs_list = sub_df['Batch Size']
            values = sub_df[metric_col]
            label = f"TP{tp} | {cap}W"
            plt.plot(bs_list, values, marker='o', label=label)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)



# Input CSV path
csv_path = 'input/allenai_OLMoE-1B-7B-0924_hellaswag_tp_powercap_summary.csv'
df = pd.read_csv(csv_path)

# Extract model name from CSV filename
csv_filename = os.path.basename(csv_path)
model_name = csv_filename.split('_hellaswag')[0]




# Output folder
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

plot_metric("Accuracy (%)", "Accuracy (%)", "Accuracy vs Batch Size", os.path.join(output_dir, f"{model_name}_accuracy_vs_batch_size.png"), df)
plot_metric("Tokens/s", "Tokens/s", "Throughput vs Batch Size", os.path.join(output_dir, f"{model_name}_throughput_vs_batch_size.png"), df)
plot_metric("Avg Total Power (W)", "Avg Total Power (W)", "Power vs Batch Size", os.path.join(output_dir, f"{model_name}_power_vs_batch_size.png"), df)
plot_metric("Avg Total Util (%)", "Avg GPU Util (%)", "GPU Utilization vs Batch Size", os.path.join(output_dir, f"{model_name}_gpu_util_vs_batch_size.png"), df)
plot_metric("Tokens/s/W", "Tokens/s/W", "Energy Efficiency vs Batch Size", os.path.join(output_dir, f"{model_name}_efficiency_vs_batch_size.png"), df)

print("\n=== Plotting Complete ===")
