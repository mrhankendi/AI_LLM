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

############################################
# Power + Utilization + Tokens/sec logging
############################################
def log_power_util(gpu_log, stop_flag, total_tokens_ref, interval=0.5):
    last_tokens = 0
    last_time = time.time()
    while not stop_flag["stop"]:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            now = time.time()
            entries = [line.split(",") for line in result.stdout.strip().split("\n") if line]
            powers = [float(e[0]) for e in entries]
            utils = [float(e[1]) for e in entries]
            current_tokens = total_tokens_ref["count"]
            delta_tokens = current_tokens - last_tokens
            delta_time = now - last_time
            tps = delta_tokens / delta_time if delta_time > 0 else 0
            gpu_log.append((now, powers, utils, current_tokens, tps))
            last_tokens = current_tokens
            last_time = now
            time.sleep(interval)
        except Exception:
            break

############################################
# Set GPU power cap
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
    times = [ts - gpu_log[0][0] for ts, *_ in gpu_log]
    tps_vals = [tps for *_, tps in gpu_log]
    plt.figure()
    plt.plot(times, tps_vals, label="Tokens/sec")
    plt.xlabel("Time (s)")
    plt.ylabel("Tokens/sec")
    plt.title(f"Throughput Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_tokens_over_time.png")
    plt.close()

def plot_power_timeseries(gpu_log, model_id, tp, cap, bs):
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
    plt.savefig(f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_power_all_gpus.png")
    plt.close()

############################################
# Arguments
############################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    help="Hugging Face model name or local path")
parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                    help="Batch sizes to test")
parser.add_argument("--max-len", type=int, default=1024,
                    help="Max model length")
parser.add_argument("--gpu-util", type=float, default=0.9,
                    help="GPU memory utilization fraction")
parser.add_argument("--tensor-parallel-sizes", type=int, nargs="+", default=[4],
                    help="List of tensor parallel sizes to sweep")
parser.add_argument("--subset", type=str, default="validation",
                    help="HellaSwag split/subset (e.g., validation[:100])")
parser.add_argument("--power-caps", type=int, nargs="+", default=[0],
                    help="List of GPU power caps in watts. 0 means default/unlimited.")
args = parser.parse_args()

############################################
# Load dataset
############################################
dataset = load_dataset("hellaswag", split=args.subset)
prompts = []
answers = []
for item in dataset:
    ctx = item["ctx_a"] + item["ctx_b"]
    choices = item["endings"]
    gold = item["label"]
    prompt = f"{ctx}\nOptions:\n" + "\n".join([f"{i}: {c}" for i, c in enumerate(choices)]) + "\nAnswer:"
    prompts.append(prompt)
    answers.append(int(gold))

############################################
# Run benchmarks
############################################
summary_results = []

for tp in args.tensor_parallel_sizes:
    for cap in args.power_caps:
        set_power_cap(cap)
        time.sleep(1)  # give NVML a moment to apply

        for bs in args.batch_sizes:
            print(f"\n=== TP {tp} | Power cap {cap if cap > 0 else 'Default'} W | Batch size {bs} ===")
            llm = LLM(
                model=args.model,
                tensor_parallel_size=tp,
                gpu_memory_utilization=args.gpu_util,
                max_model_len=args.max_len,
                trust_remote_code=True
            )
            sampling_params = SamplingParams(temperature=0.0, max_tokens=5)

            gpu_log = []
            total_tokens_ref = {"count": 0}
            stop_flag = {"stop": False}
            log_thread = threading.Thread(target=log_power_util,
                                          args=(gpu_log, stop_flag, total_tokens_ref),
                                          daemon=True)
            log_thread.start()

            start_time = time.time()
            correct = 0

            for i in range(0, len(prompts), bs):
                batch_prompts = prompts[i:i+bs]
                batch_answers = answers[i:i+bs]
                outputs = llm.generate(batch_prompts, sampling_params)

                for ref_answer, output in zip(batch_answers, outputs):
                    text = output.outputs[0].text.strip()
                    total_tokens_ref["count"] += len(output.outputs[0].token_ids)
                    try:
                        pred = int([c for c in text if c.isdigit()][0])
                    except:
                        pred = -1
                    if pred == ref_answer:
                        correct += 1

            elapsed_time = time.time() - start_time
            stop_flag["stop"] = True
            log_thread.join()

            accuracy = correct / len(prompts) * 100
            tokens_per_sec = total_tokens_ref["count"] / elapsed_time

            if gpu_log:
                per_gpu_power = [mean([p[i] for _, p, _, _, _ in gpu_log]) for i in range(len(gpu_log[0][1]))]
                per_gpu_util = [mean([u[i] for _, _, u, _, _ in gpu_log]) for i in range(len(gpu_log[0][2]))]
                avg_total_power = mean([sum(p) for _, p, _, _, _ in gpu_log])
                avg_total_util = mean([mean(u) for _, _, u, _, _ in gpu_log])
                total_energy_Wh = avg_total_power * (elapsed_time / 3600)
            else:
                per_gpu_power = []
                per_gpu_util = []
                avg_total_power = 0
                avg_total_util = 0
                total_energy_Wh = 0

            tokens_per_watt = tokens_per_sec / avg_total_power if avg_total_power > 0 else 0

            summary_results.append([
                tp, cap if cap > 0 else "Default", bs, accuracy, tokens_per_sec, avg_total_power, per_gpu_power,
                avg_total_util, per_gpu_util, total_energy_Wh, tokens_per_watt
            ])

            # Save time series CSV
            with open(f"{args.model.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_timeseries.csv", "w", newline="") as f:
                writer = csv.writer(f)
                header = ["Timestamp"] + \
                         [f"GPU{i}_Power_W" for i in range(len(per_gpu_power))] + \
                         [f"GPU{i}_Util_%" for i in range(len(per_gpu_util))] + \
                         ["Total_Tokens", "Tokens_per_s"]
                writer.writerow(header)
                for ts, powers, utils, total_toks, tps in gpu_log:
                    writer.writerow([ts] + powers + utils + [total_toks, tps])

            # Plots
            plot_tokens_timeseries(gpu_log, args.model, tp, cap, bs)
            plot_power_timeseries(gpu_log, args.model, tp, cap, bs)

            # Cleanup
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

# Restore defaults
set_power_cap(0)

############################################
# Save summary CSV
############################################
summary_file = f"{args.model.replace('/', '_')}_hellaswag_tp_powercap_summary.csv"
with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Tensor Parallel Size", "Power Cap (W)", "Batch Size", "Accuracy (%)", "Tokens/s",
        "Avg Total Power (W)", "Per-GPU Power (W)", "Avg Total Util (%)", "Per-GPU Util (%)",
        "Total Energy (Wh)", "Tokens/s/W"
    ])
    for row in summary_results:
        writer.writerow(row)

############################################
# Plots
############################################
def plot_metric(metric_idx, ylabel, title, filename):
    plt.figure()
    for tp in sorted(set(r[0] for r in summary_results)):
        for cap in sorted(set(r[1] for r in summary_results), key=lambda x: (x != "Default", x)):
            bs_list = [r[2] for r in summary_results if r[0] == tp and r[1] == cap]
            values = [r[metric_idx] for r in summary_results if r[0] == tp and r[1] == cap]
            label = f"TP{tp} | {cap}W" if cap != "Default" else f"TP{tp} | Default"
            plt.plot(bs_list, values, marker='o', label=label)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({args.model})")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)

plot_metric(3, "Accuracy (%)", "Accuracy vs Batch Size", f"{args.model.replace('/', '_')}_accuracy_vs_batch_size.png")
plot_metric(4, "Tokens/s", "Throughput vs Batch Size", f"{args.model.replace('/', '_')}_throughput_vs_batch_size.png")
plot_metric(5, "Avg Total Power (W)", "Power vs Batch Size", f"{args.model.replace('/', '_')}_power_vs_batch_size.png")
plot_metric(7, "Avg GPU Util (%)", "GPU Utilization vs Batch Size", f"{args.model.replace('/', '_')}_gpu_util_vs_batch_size.png")
plot_metric(10, "Tokens/s/W", "Energy Efficiency vs Batch Size", f"{args.model.replace('/', '_')}_efficiency_vs_batch_size.png")

print("\n=== Benchmark Complete ===")
print(f"Summary saved to {summary_file}")

print("\n=== Benchmark Complete ===")
print(f"Summary saved to {summary_file}")

