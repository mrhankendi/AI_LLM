#!/usr/bin/env python3
import time
import threading
import subprocess
import csv
from statistics import mean
from datasets import load_dataset
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
import gc
import torch
import argparse
import os

import pynvml

############################################
# NVML: GPU Power + Utilization logger
############################################
def init_nvml():
    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(f"[WARN] Failed to initialize NVML: {e}")

def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

def log_gpu_power_util(gpu_log, stop_flag, total_tokens_ref, interval=0.5):
    """
    Logs per-GPU power (W), utilization (%), total tokens, and tokens/s.
    Uses NVML for power/util (faster & more accurate than nvidia-smi).
    """
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as e:
        print(f"[WARN] NVML error, GPU logging disabled: {e}")
        return

    last_tokens = 0
    last_time = time.time()

    while not stop_flag["stop"]:
        try:
            now = time.time()
            powers = []
            utils = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # power in mW -> W
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                u = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                powers.append(p)
                utils.append(float(u))

            current_tokens = total_tokens_ref["count"]
            delta_tokens = current_tokens - last_tokens
            delta_time = now - last_time
            tps = delta_tokens / delta_time if delta_time > 0 else 0.0

            gpu_log.append((now, powers, utils, current_tokens, tps))

            last_tokens = current_tokens
            last_time = now

            time.sleep(interval)
        except Exception as e:
            print(f"[WARN] GPU logging thread exiting due to error: {e}")
            break

############################################
# IPMI logger (slower)
############################################
IPMI_SENSORS = [
    "Sys Power",
    "CPU Power",
    "Mem Power",
    "GPU Board Power",
    "Riser 1 Power",
]

def parse_ipmi_sensor_output(stdout):
    """
    Expected format from:
    ipmitool -I open sensor reading "Sys Power" "CPU Power" ...
    e.g.
    Sys Power        | 340
    CPU Power        | 240
    ...
    Returns dict: { "Sys Power": 340.0, ... }
    """
    values = {}
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            name = parts[0]
            val_str = parts[1]
            try:
                val = float(val_str)
            except ValueError:
                val = None
            values[name] = val
    return values

def log_ipmi(ipmi_log, stop_flag, interval=3.0):
    """
    Logs IPMI readings for selected sensors.
    IPMI is slow (~1.4s per call), so polling interval is coarse.
    """
    cmd = [
        "ipmitool", "-I", "open", "sensor", "reading",
        *[f"{s}" for s in IPMI_SENSORS]
    ]

    while not stop_flag["stop"]:
        try:
            now = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print("[WARN] IPMI command failed.")
            values = parse_ipmi_sensor_output(result.stdout)
            # Order: Sys, CPU, Mem, GPU Board, Riser 1
            entry = (
                now,
                values.get("Sys Power"),
                values.get("CPU Power"),
                values.get("Mem Power"),
                values.get("GPU Board Power"),
                values.get("Riser 1 Power"),
            )
            ipmi_log.append(entry)
        except Exception as e:
            print(f"[WARN] IPMI logging thread exiting due to error: {e}")
            break

        time.sleep(interval)

############################################
# Set GPU power cap (nvidia-smi)
############################################
def set_power_cap(watts):
    if watts > 0:
        print(f"[INFO] Setting power cap to {watts} W for all GPUs...")
        subprocess.run(f"sudo nvidia-smi -pl {watts}", shell=True, check=False)
    else:
        print("[INFO] Restoring default GPU power limits...")
        subprocess.run("sudo nvidia-smi -rgc", shell=True, check=False)

############################################
# Plot functions
############################################
def plot_tokens_timeseries(gpu_log, model_id, tp, cap, bs):
    if not gpu_log:
        return
    times = [ts - gpu_log[0][0] for ts, *_ in gpu_log]
    tps_vals = [tps for *_, tps in gpu_log]
    plt.figure()
    plt.plot(times, tps_vals, label="Tokens/sec")
    plt.xlabel("Time (s)")
    plt.ylabel("Tokens/sec")
    plt.title(f"Throughput Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_tokens_over_time.png")
    plt.close()

def plot_power_timeseries(gpu_log, model_id, tp, cap, bs):
    if not gpu_log:
        return
    times = [ts - gpu_log[0][0] for ts, *_ in gpu_log]
    gpu_count = len(gpu_log[0][1])
    plt.figure()
    for i in range(gpu_count):
        powers = [p[i] for _, p, *_ in gpu_log]
        plt.plot(times, powers, label=f"GPU {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"Per-GPU Power Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_power_all_gpus.png")
    plt.close()

def plot_gpu_vs_system_power(gpu_log, ipmi_log, model_id, tp, cap, bs):
    if not gpu_log or not ipmi_log:
        return
    t0 = gpu_log[0][0]
    gpu_times = [ts - t0 for ts, *_ in gpu_log]
    gpu_total_power = [sum(powers) for _, powers, *_ in gpu_log]

    ipmi_times = [ts - t0 for ts, *_ in ipmi_log]
    sys_powers = [entry[1] for entry in ipmi_log]  # Sys Power
    gpu_board_powers = [entry[4] for entry in ipmi_log]  # GPU Board Power

    # Overlay GPU total power & System power
    plt.figure()
    plt.plot(gpu_times, gpu_total_power, label="Sum GPU NVML Power")
    plt.plot(ipmi_times, sys_powers, label="IPMI Sys Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"GPU vs System Power Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_gpu_vs_system_power.png")
    plt.close()

    # Full IPMI breakdown
    plt.figure()
    plt.plot(ipmi_times, sys_powers, label="Sys Power")
    plt.plot(ipmi_times, [e[2] for e in ipmi_log], label="CPU Power")
    plt.plot(ipmi_times, [e[3] for e in ipmi_log], label="Mem Power")
    plt.plot(ipmi_times, gpu_board_powers, label="GPU Board Power")
    plt.plot(ipmi_times, [e[5] for e in ipmi_log], label="Riser 1 Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"IPMI Power Breakdown Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_ipmi_breakdown.png")
    plt.close()

############################################
# Pareto frontier (Power vs Throughput)
############################################
def compute_pareto(points):
    """
    points: list of (power, throughput)
    Minimizing power, maximizing throughput.
    Returns sorted frontier.
    """
    points_sorted = sorted(points, key=lambda x: x[0])  # sort by power ascending
    frontier = []
    best_throughput = -float("inf")
    for p, t in points_sorted:
        if t > best_throughput:
            frontier.append((p, t))
            best_throughput = t
    return frontier

def plot_pareto_frontier(summary_results, model_id):
    # summary_results row layout (indices):
    # 0: TP, 1: Cap, 2: BS, 3: Accuracy, 4: Tokens/s
    # 5: Avg Total Power, 10: Tokens/s/W
    if not summary_results:
        return

    powers = [r[5] for r in summary_results]
    throughputs = [r[4] for r in summary_results]
    pts = list(zip(powers, throughputs))
    frontier = compute_pareto(pts)

    plt.figure()
    plt.scatter(powers, throughputs, alpha=0.6, label="All configs")
    if frontier:
        f_p, f_t = zip(*frontier)
        plt.plot(f_p, f_t, marker="o", label="Pareto frontier")
    plt.xlabel("Avg Total Power (W)")
    plt.ylabel("Tokens/s")
    plt.title(f"Power vs Throughput Pareto Frontier ({model_id})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_id.replace('/', '_')}_pareto_power_vs_throughput.png")
    plt.close()

############################################
# Arguments
############################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    help="Hugging Face model name or local path")
parser.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 4, 8, 16],
                    help="Batch sizes to test")
parser.add_argument("--max-len", type=int, default=1024,
                    help="Max model length (must not exceed model's positional limit)")
parser.add_argument("--gpu-util", type=float, default=0.9,
                    help="GPU memory utilization fraction")
parser.add_argument("--tensor-parallel-sizes", type=int, nargs="+",
                    default=[4],
                    help="List of tensor parallel sizes to sweep")
parser.add_argument("--subset", type=str, default="validation",
                    help='HellaSwag split/subset, e.g. "validation[:100]"')
parser.add_argument("--power-caps", type=int, nargs="+",
                    default=[0],
                    help="List of GPU power caps in watts. 0 means default/unlimited.")
args = parser.parse_args()

############################################
# Load dataset (HellaSwag)
############################################
dataset = load_dataset("hellaswag", split=args.subset)
prompts = []
answers = []
for item in dataset:
    ctx = item["ctx_a"] + item["ctx_b"]
    choices = item["endings"]
    gold = item["label"]
    prompt = (
        f"{ctx}\nOptions:\n"
        + "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        + "\nAnswer:"
    )
    prompts.append(prompt)
    answers.append(int(gold))

############################################
# Run benchmarks
############################################
summary_results = []

init_nvml()
try:
    for tp in args.tensor_parallel_sizes:
        for cap in args.power_caps:
            set_power_cap(cap)
            time.sleep(1)  # give driver a moment to apply

            for bs in args.batch_sizes:
                print(f"\n=== TP {tp} | Power cap {cap if cap > 0 else 'Default'} W | Batch size {bs} ===")

                llm = LLM(
                    model=args.model,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=args.gpu_util,
                    max_model_len=args.max_len,
                    trust_remote_code=True,
                )
                sampling_params = SamplingParams(temperature=0.0, max_tokens=5)

                gpu_log = []
                ipmi_log = []
                total_tokens_ref = {"count": 0}
                stop_flag = {"stop": False}

                # Start logging threads
                gpu_thread = threading.Thread(
                    target=log_gpu_power_util,
                    args=(gpu_log, stop_flag, total_tokens_ref, 0.5),
                    daemon=True,
                )
                ipmi_thread = threading.Thread(
                    target=log_ipmi,
                    args=(ipmi_log, stop_flag, 1.5),
                    daemon=True,
                )
                gpu_thread.start()
                ipmi_thread.start()

                start_time = time.time()
                correct = 0

                for i in range(0, len(prompts), bs):
                    batch_prompts = prompts[i:i + bs]
                    batch_answers = answers[i:i + bs]
                    outputs = llm.generate(batch_prompts, sampling_params)

                    for ref_answer, output in zip(batch_answers, outputs):
                        text = output.outputs[0].text.strip()
                        total_tokens_ref["count"] += len(output.outputs[0].token_ids)
                        try:
                            pred = int([c for c in text if c.isdigit()][0])
                        except Exception:
                            pred = -1
                        if pred == ref_answer:
                            correct += 1

                elapsed_time = time.time() - start_time
                stop_flag["stop"] = True
                gpu_thread.join()
                ipmi_thread.join()

                # Aggregate GPU stats
                accuracy = correct / len(prompts) * 100.0
                tokens_per_sec = total_tokens_ref["count"] / elapsed_time if elapsed_time > 0 else 0.0

                if gpu_log:
                    per_gpu_power = [
                        mean([p[i] for _, p, _, _, _ in gpu_log])
                        for i in range(len(gpu_log[0][1]))
                    ]
                    per_gpu_util = [
                        mean([u[i] for _, _, u, _, _ in gpu_log])
                        for i in range(len(gpu_log[0][2]))
                    ]
                    avg_total_power = mean([sum(p) for _, p, _, _, _ in gpu_log])
                    avg_total_util = mean([mean(u) for _, _, u, _, _ in gpu_log])
                    total_energy_Wh = avg_total_power * (elapsed_time / 3600.0)
                else:
                    per_gpu_power = []
                    per_gpu_util = []
                    avg_total_power = 0.0
                    avg_total_util = 0.0
                    total_energy_Wh = 0.0

                tokens_per_watt = tokens_per_sec / avg_total_power if avg_total_power > 0 else 0.0

                # Aggregate IPMI stats
                if ipmi_log:
                    sys_vals = [e[1] for e in ipmi_log if e[1] is not None]
                    cpu_vals = [e[2] for e in ipmi_log if e[2] is not None]
                    mem_vals = [e[3] for e in ipmi_log if e[3] is not None]
                    gpu_board_vals = [e[4] for e in ipmi_log if e[4] is not None]
                    riser_vals = [e[5] for e in ipmi_log if e[5] is not None]

                    avg_sys_power = mean(sys_vals) if sys_vals else 0.0
                    avg_cpu_power = mean(cpu_vals) if cpu_vals else 0.0
                    avg_mem_power = mean(mem_vals) if mem_vals else 0.0
                    avg_gpu_board_power = mean(gpu_board_vals) if gpu_board_vals else 0.0
                    avg_riser_power = mean(riser_vals) if riser_vals else 0.0
                else:
                    avg_sys_power = avg_cpu_power = avg_mem_power = 0.0
                    avg_gpu_board_power = avg_riser_power = 0.0

                summary_results.append([
                    tp,
                    cap if cap > 0 else "Default",
                    bs,
                    accuracy,
                    tokens_per_sec,
                    avg_total_power,
                    per_gpu_power,
                    avg_total_util,
                    per_gpu_util,
                    total_energy_Wh,
                    tokens_per_watt,
                    avg_sys_power,
                    avg_cpu_power,
                    avg_mem_power,
                    avg_gpu_board_power,
                    avg_riser_power,
                ])

                # Build time-aligned timeseries CSV combining GPU + IPMI
                timeseries_file = f"{args.model.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_timeseries.csv"
                with open(timeseries_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    gpu_count = len(per_gpu_power)
                    header = (
                        ["Timestamp"]
                        + [f"GPU{i}_Power_W" for i in range(gpu_count)]
                        + [f"GPU{i}_Util_%" for i in range(gpu_count)]
                        + ["Total_Tokens", "Tokens_per_s",
                           "IPMI_Sys_Power_W", "IPMI_CPU_Power_W",
                           "IPMI_Mem_Power_W", "IPMI_GPU_Board_Power_W",
                           "IPMI_Riser1_Power_W"]
                    )
                    writer.writerow(header)

                    # Align IPMI samples to nearest earlier timestamp for each GPU sample
                    ipmi_idx = -1
                    last_ipmi_vals = (None, None, None, None, None)
                    for ts, powers, utils, total_toks, tps in gpu_log:
                        # advance ipmi_idx while next ipmi sample is earlier than ts
                        while (ipmi_idx + 1) < len(ipmi_log) and ipmi_log[ipmi_idx + 1][0] <= ts:
                            ipmi_idx += 1
                            last_ipmi_vals = ipmi_log[ipmi_idx][1:]  # drop timestamp
                        sys_p, cpu_p, mem_p, gpu_board_p, riser_p = last_ipmi_vals
                        writer.writerow([
                            ts,
                            *powers,
                            *utils,
                            total_toks,
                            tps,
                            sys_p,
                            cpu_p,
                            mem_p,
                            gpu_board_p,
                            riser_p,
                        ])

                # Plots (keep existing + new ones)
                plot_tokens_timeseries(gpu_log, args.model, tp, cap, bs)
                plot_power_timeseries(gpu_log, args.model, tp, cap, bs)
                plot_gpu_vs_system_power(gpu_log, ipmi_log, args.model, tp, cap, bs)

                # Cleanup
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2)

    # Restore defaults
    set_power_cap(0)

finally:
    shutdown_nvml()

############################################
# Save summary CSV (append IPMI breakdown)
############################################
summary_file = f"{args.model.replace('/', '_')}_hellaswag_tp_powercap_summary.csv"
with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Tensor Parallel Size",
        "Power Cap (W)",
        "Batch Size",
        "Accuracy (%)",
        "Tokens/s",
        "Avg Total GPU Power (W)",
        "Per-GPU Power (W)",
        "Avg GPU Util (%)",
        "Per-GPU Util (%)",
        "Total Energy (Wh)",
        "Tokens/s/W",
        "Avg Sys Power (W)",
        "Avg CPU Power (W)",
        "Avg Mem Power (W)",
        "Avg GPU Board Power (W)",
        "Avg Riser 1 Power (W)",
    ])
    for row in summary_results:
        writer.writerow(row)

############################################
# Keep your original plot_metric + plots
############################################
def plot_metric(metric_idx, ylabel, title, filename):
    plt.figure()
    for tp in sorted(set(r[0] for r in summary_results)):
        # sort caps: Default first, then numeric
        caps = sorted(
            set(r[1] for r in summary_results if r[0] == tp),
            key=lambda x: (x != "Default", float(x) if x != "Default" else 0.0),
        )
        for cap in caps:
            bs_list = [r[2] for r in summary_results if r[0] == tp and r[1] == cap]
            values = [r[metric_idx] for r in summary_results if r[0] == tp and r[1] == cap]
            label = f"TP{tp} | {cap}W" if cap != "Default" else f"TP{tp} | Default"
            plt.plot(bs_list, values, marker='o', label=label)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({args.model})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Indices unchanged:
# 3: Accuracy, 4: Tokens/s, 5: Avg Total GPU Power, 7: Avg GPU Util, 10: Tokens/s/W
plot_metric(3, "Accuracy (%)", "Accuracy vs Batch Size",
            f"{args.model.replace('/', '_')}_accuracy_vs_batch_size.png")
plot_metric(4, "Tokens/s", "Throughput vs Batch Size",
            f"{args.model.replace('/', '_')}_throughput_vs_batch_size.png")
plot_metric(5, "Avg Total Power (W)", "Power vs Batch Size",
            f"{args.model.replace('/', '_')}_power_vs_batch_size.png")
plot_metric(7, "Avg GPU Util (%)", "GPU Utilization vs Batch Size",
            f"{args.model.replace('/', '_')}_gpu_util_vs_batch_size.png")
plot_metric(10, "Tokens/s/W", "Energy Efficiency vs Batch Size",
            f"{args.model.replace('/', '_')}_efficiency_vs_batch_size.png")

# Pareto frontier (Power vs Throughput)
plot_pareto_frontier(summary_results, args.model)

print("\n=== Benchmark Complete ===")
print(f"Summary saved to {summary_file}")

