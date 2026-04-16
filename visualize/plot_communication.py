import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Specify the CSV file to plot
filename = r"output\meta-llama_Llama-2-7b-hf_tp2_cap100_bs1_gpu_timeseries.csv"

# Read the CSV file
df = pd.read_csv(filename)

# Create time axis (assuming sampling every second or use row index)
if 'Timestamp' in df.columns:
    time = pd.to_datetime(df['Timestamp'])
    time = (time - time.iloc[0]).dt.total_seconds()
else:
    time = np.arange(len(df))

# Identify all PCIe columns
pcie_tx_cols = [col for col in df.columns if 'PCIe_TX_MBps' in col]
pcie_rx_cols = [col for col in df.columns if 'PCIe_RX_MBps' in col]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot PCIe TX (Transmit) for all GPUs
for col in pcie_tx_cols:
    ax1.plot(time, df[col], label=col, linewidth=1.5)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('PCIe TX (MB/s)')
ax1.set_title('PCIe Transmit Bandwidth Over Time')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot PCIe RX (Receive) for all GPUs
for col in pcie_rx_cols:
    ax2.plot(time, df[col], label=col, linewidth=1.5)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('PCIe RX (MB/s)')
ax2.set_title('PCIe Receive Bandwidth Over Time')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pcie_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plotted {len(pcie_tx_cols)} TX and {len(pcie_rx_cols)} RX PCIe columns")
print(f"Data points: {len(df)}")
print(f"Time range: {time.iloc[0]:.1f}s - {time.iloc[-1]:.1f}s")
