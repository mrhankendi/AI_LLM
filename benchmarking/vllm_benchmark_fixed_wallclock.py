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
import math

############################################
# NVML (GPU power + utilization)
############################################
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[WARN] pynvml not available, GPU power/util logging will be disabled.")

nvml_handles = []
nvml_device_count = 0


def init_nvml():
    global nvml_handles, nvml_device_count
    if not NVML_AVAILABLE:
        return
    pynvml.nvmlInit()
    nvml_device_count = pynvml.nvmlDeviceGetCount()
    nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(nvml_device_count)]
    print(f"[INFO] NVML initialized with {nvml_device_count} GPUs.")


def shutdown_nvml():
    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()
        print("[INFO] NVML shut down.")


############################################
# Power + Utilization + Tokens/sec logging
############################################
def log_gpu_power_util(gpu_log, stop_flag, total_tokens_ref,
                       interval=0.5):
    """High-frequency GPU logging via NVML."""
    if not NVML_AVAILABLE or nvml_device_count == 0:
        print("[WARN] NVML not available, skipping GPU power/util logging.")
        return

    last_tokens = 0
    last_time = time.time()

    while not stop_flag["stop"]:
        now = time.time()
        try:
            powers = []
            utils = []
            for h in nvml_handles:
                try:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
                    p_w = p_mw / 1000.0
                except pynvml.NVMLError:
                    p_w = float("nan")

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                except pynvml.NVMLError:
                    util = float("nan")

                powers.append(p_w)
                utils.append(util)

            current_tokens = total_tokens_ref["count"]
            delta_tokens = current_tokens - last_tokens
            delta_time = now - last_time
            tps = delta_tokens / delta_time if delta_time > 0 else 0.0

            gpu_log.append((now, powers, utils, current_tokens, tps))

            last_tokens = current_tokens
            last_time = now
            time.sleep(interval)
        except Exception as e:
            print(f"[WARN] GPU logging thread error: {e}")
            break


def parse_ipmi_sensor_output(output: str):
    """
    Parse `ipmitool -I open sensor reading "Sys Power" "CPU Power" ...`
    Returns (sys, cpu, mem, gpu_board, riser1) in Watts, or None if parse fails.
    """
    # Initialize as NaN so we can see missing values explicitly
    values = {
        "Sys Power": math.nan,
        "CPU Power": math.nan,
        "Mem Power": math.nan,
        "GPU Board Power": math.nan,
        "Riser 1 Power": math.nan,
    }

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Typical line format:
        # Sys Power        | 340
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        name, val_str = parts[0], parts[1]
        if name in values:
            try:
                # Handle cases like "340", or "340 Watts"
                val = val_str.split()[0]
                values[name] = float(val)
            except ValueError:
                pass

    return (
        values["Sys Power"],
        values["CPU Power"],
        values["Mem Power"],
        values["GPU Board Power"],
        values["Riser 1 Power"],
    )


def log_ipmi(ipmi_log, stop_flag, interval=1.5):
    """
    Lower-frequency IPMI logging. Uses `sensor reading` for the 5 powers.

    NOTE: IPMI is ~1.4-1.5s per call on your system, so keep interval >= ~1.5s
    to avoid hammering the BMC.
    """
    cmd = [
        "ipmitool", "-I", "open", "sensor", "reading",
        "Sys Power", "CPU Power", "Mem Power", "GPU Board Power", "Riser 1 Power",
    ]

    while not stop_flag["stop"]:
        now = time.time()
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            sys_p, cpu_p, mem_p, gpu_board_p, riser1_p = parse_ipmi_sensor_output(
                result.stdout
            )
            ipmi_log.append(
                (now, sys_p, cpu_p, mem_p, gpu_board_p, riser1_p)
            )
            time.sleep(interval)
        except Exception as e:
            print(f"[WARN] IPMI logging thread error: {e}")
            break


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
# Plot functions (time series)
############################################
def plot_tokens_timeseries(gpu_log, model_id, tp, cap, bs, out_dir):
    if not gpu_log:
        return
    t0 = gpu_log[0][0]
    times = [ts - t0 for ts, *_ in gpu_log]
    tps_vals = [tps for *_, tps in gpu_log]

    plt.figure()
    plt.plot(times, tps_vals, label="Tokens/sec")
    plt.xlabel("Time (s)")
    plt.ylabel("Tokens/sec")
    plt.title(f"Throughput Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    fname = os.path.join(
        out_dir,
        f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_tokens_over_time.png",
    )
    plt.savefig(fname)
    plt.close()


def plot_power_timeseries(gpu_log, model_id, tp, cap, bs, out_dir):
    if not gpu_log:
        return
    t0 = gpu_log[0][0]
    gpu_count = len(gpu_log[0][1])

    plt.figure()
    for i in range(gpu_count):
        powers = [p[i] for _, p, *_ in gpu_log]
        times = [ts - t0 for ts, *_ in gpu_log]
        plt.plot(times, powers, label=f"GPU {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"Per-GPU Power Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}")
    plt.grid(True)
    plt.legend()
    fname = os.path.join(
        out_dir,
        f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_power_all_gpus.png",
    )
    plt.savefig(fname)
    plt.close()


def plot_gpu_vs_system_power_overlay(gpu_log, ipmi_log, model_id, tp, cap, bs, out_dir):
    if not gpu_log or not ipmi_log:
        return

    t0 = min(gpu_log[0][0], ipmi_log[0][0])

    gpu_times = [ts - t0 for ts, *_ in gpu_log]
    gpu_total_power = [sum(p) for _, p, *_ in gpu_log]

    ipmi_times = [ts - t0 for ts, *_ in ipmi_log]
    sys_powers = [sys for _, sys, *_ in ipmi_log]

    plt.figure()
    plt.plot(gpu_times, gpu_total_power, label="Total GPU Power (NVML)")
    plt.plot(ipmi_times, sys_powers, label="System Power (IPMI)", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(
        f"GPU vs System Power Over Time\n{model_id} TP={tp} Cap={cap}W BS={bs}"
    )
    plt.grid(True)
    plt.legend()
    fname = os.path.join(
        out_dir,
        f"{model_id.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_gpu_vs_system_power.png",
    )
    plt.savefig(fname)
    plt.close()


############################################
# Summary plots helpers
############################################
def plot_metric(summary_results, metric_idx, ylabel, title, filename, model_name):
    plt.figure()
    for tp in sorted(set(r[0] for r in summary_results)):
        caps = sorted(
            set(r[1] for r in summary_results if r[0] == tp),
            key=lambda x: (x != "Default", x),
        )
        for cap in caps:
            bs_list = [
                r[2]
                for r in summary_results
                if r[0] == tp and r[1] == cap
            ]
            values = [
                r[metric_idx]
                for r in summary_results
                if r[0] == tp and r[1] == cap
            ]
            label = f"TP{tp} | {cap}W" if cap != "Default" else f"TP{tp} | Default"
            plt.plot(bs_list, values, marker="o", label=label)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({model_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def compute_pareto_frontier(points):
    """
    points: list of (power, throughput)
    Returns Pareto frontier points sorted by increasing power, where throughput
    is monotonically non-increasing (upper-left frontier).
    """
    points = sorted(points, key=lambda x: x[0])  # sort by power ascending
    frontier = []
    best_tps = -1.0
    for p, t in points:
        if t > best_tps:
            frontier.append((p, t))
            best_tps = t
    return frontier


def plot_pareto(summary_results, out_file, use_system_power=False):
    # power_idx: 5 for avg GPU total power, 11 for avg system power
    power_idx = 11 if use_system_power else 5
    tokens_idx = 4

    points = []
    for r in summary_results:
        power = r[power_idx]
        tps = r[tokens_idx]
        if power > 0 and tps > 0:
            points.append((power, tps))

    if not points:
        return

    frontier = compute_pareto_frontier(points)

    plt.figure()
    ps = [p for p, _ in points]
    ts = [t for _, t in points]
    plt.scatter(ps, ts, alpha=0.5, label="All configs")

    fp = [p for p, _ in frontier]
    ft = [t for _, t in frontier]
    plt.plot(fp, ft, marker="o", label="Pareto frontier")

    xlabel = "Avg System Power (W)" if use_system_power else "Avg GPU Power (W)"
    plt.xlabel(xlabel)
    plt.ylabel("Tokens/s")
    title = "Pareto Frontier (System Power vs Throughput)" if use_system_power \
        else "Pareto Frontier (GPU Power vs Throughput)"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_file)
    plt.close()


############################################
# Main
############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or local path",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Max model length (must not exceed model's max_position_embeddings).",
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction for vLLM",
    )
    parser.add_argument(
        "--tensor-parallel-sizes",
        type=int,
        nargs="+",
        default=[4],
        help="List of tensor parallel sizes to sweep",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="validation",
        help='HellaSwag split/subset, e.g. "validation", "validation[:100]"',
    )
    parser.add_argument(
        "--power-caps",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU power caps in watts. 0 means default/unlimited.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=0.0,
        help=(
            "Max duration per (TP, power cap, BS) run in seconds. "
            "0 means run until dataset is exhausted."
        ),
    )
    parser.add_argument(
        "--gpu-log-interval",
        type=float,
        default=0.5,
        help="GPU (NVML) logging interval in seconds.",
    )
    parser.add_argument(
        "--ipmi-log-interval",
        type=float,
        default=1.5,
        help="IPMI logging interval in seconds.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for CSVs and plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ############################################
    # Load dataset (HellaSwag)
    ############################################
    print(f"[INFO] Loading HellaSwag split: {args.subset}")
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

    print(f"[INFO] Loaded {len(prompts)} HellaSwag prompts.")

    ############################################
    # Init NVML
    ############################################
    init_nvml()

    summary_results = []

    ############################################
    # Run benchmarks
    ############################################
    for tp in args.tensor_parallel_sizes:
        for cap in args.power_caps:
            set_power_cap(cap)
            time.sleep(1)  # give driver a moment

            for bs in args.batch_sizes:
                cap_label = cap if cap > 0 else "Default"
                print(
                    f"\n=== TP {tp} | Power cap {cap_label} W | "
                    f"Batch size {bs} | Time budget {args.time_budget}s ==="
                )

                llm = LLM(
                    model=args.model,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=args.gpu_util,
                    max_model_len=args.max_len,
                    trust_remote_code=True,
                )
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=5,
                )

                gpu_log = []
                ipmi_log = []

                total_tokens_ref = {"count": 0}
                stop_flag = {"stop": False}

                # Start logging threads
                gpu_thread = threading.Thread(
                    target=log_gpu_power_util,
                    args=(gpu_log, stop_flag, total_tokens_ref,
                          args.gpu_log_interval),
                    daemon=True,
                )
                ipmi_thread = threading.Thread(
                    target=log_ipmi,
                    args=(ipmi_log, stop_flag, args.ipmi_log_interval),
                    daemon=True,
                )

                gpu_thread.start()
                ipmi_thread.start()

                start_time = time.time()
                correct = 0
                total_samples = 0

                idx = 0
                n = len(prompts)

                # Fixed-duration loop: stop when either dataset is exhausted
                # OR time budget is reached (if > 0).
                while idx < n:
                    if args.time_budget > 0.0:
                        if time.time() - start_time >= args.time_budget:
                            break

                    batch_prompts = prompts[idx: idx + bs]
                    batch_answers = answers[idx: idx + bs]
                    if not batch_prompts:
                        break

                    outputs = llm.generate(batch_prompts, sampling_params)

                    for ref_answer, output in zip(batch_answers, outputs):
                        text = output.outputs[0].text.strip()
                        total_tokens_ref["count"] += len(
                            output.outputs[0].token_ids
                        )
                        try:
                            pred = int([c for c in text if c.isdigit()][0])
                        except Exception:
                            pred = -1
                        if pred == ref_answer:
                            correct += 1
                        total_samples += 1

                    idx += bs

                elapsed_time = time.time() - start_time

                # Stop logging
                stop_flag["stop"] = True
                gpu_thread.join()
                ipmi_thread.join()

                accuracy = (
                    (correct / total_samples) * 100.0
                    if total_samples > 0
                    else 0.0
                )
                tokens_per_sec = (
                    total_tokens_ref["count"] / elapsed_time
                    if elapsed_time > 0
                    else 0.0
                )

                # Aggregate GPU stats
                if gpu_log:
                    gpu_count = len(gpu_log[0][1])
                    per_gpu_power = [
                        mean([p[i] for _, p, _, _, _ in gpu_log])
                        for i in range(gpu_count)
                    ]
                    per_gpu_util = [
                        mean([u[i] for _, _, u, _, _ in gpu_log])
                        for i in range(gpu_count)
                    ]
                    avg_total_gpu_power = mean(
                        [sum(p) for _, p, _, _, _ in gpu_log]
                    )
                    avg_total_gpu_util = mean(
                        [mean(u) for _, _, u, _, _ in gpu_log]
                    )
                    total_gpu_energy_Wh = avg_total_gpu_power * (
                        elapsed_time / 3600.0
                    )
                else:
                    per_gpu_power = []
                    per_gpu_util = []
                    avg_total_gpu_power = 0.0
                    avg_total_gpu_util = 0.0
                    total_gpu_energy_Wh = 0.0

                tokens_per_watt_gpu = (
                    tokens_per_sec / avg_total_gpu_power
                    if avg_total_gpu_power > 0
                    else 0.0
                )

                # Aggregate IPMI stats
                if ipmi_log:
                    avg_sys_power = mean([sys for _, sys, *_ in ipmi_log])
                    avg_cpu_power = mean([cpu for _, _, cpu, *_ in ipmi_log])
                    avg_mem_power = mean([mem for _, _, _, mem, *_ in ipmi_log])
                    avg_gpu_board_power = mean(
                        [g for _, _, _, _, g, _ in ipmi_log]
                    )
                    avg_riser1_power = mean(
                        [r for _, _, _, _, _, r in ipmi_log]
                    )
                    total_sys_energy_Wh = avg_sys_power * (
                        elapsed_time / 3600.0
                    )
                else:
                    avg_sys_power = 0.0
                    avg_cpu_power = 0.0
                    avg_mem_power = 0.0
                    avg_gpu_board_power = 0.0
                    avg_riser1_power = 0.0
                    total_sys_energy_Wh = 0.0

                tokens_per_watt_sys = (
                    tokens_per_sec / avg_sys_power
                    if avg_sys_power > 0
                    else 0.0
                )

                # Save summary row
                summary_results.append(
                    [
                        tp,                   # 0
                        cap_label,            # 1
                        bs,                   # 2
                        accuracy,             # 3
                        tokens_per_sec,       # 4
                        avg_total_gpu_power,  # 5
                        per_gpu_power,        # 6
                        avg_total_gpu_util,   # 7
                        per_gpu_util,         # 8
                        total_gpu_energy_Wh,  # 9
                        tokens_per_watt_gpu,  # 10
                        avg_sys_power,        # 11
                        avg_cpu_power,        # 12
                        avg_mem_power,        # 13
                        avg_gpu_board_power,  # 14
                        avg_riser1_power,     # 15
                        total_sys_energy_Wh,  # 16
                        tokens_per_watt_sys,  # 17
                        elapsed_time,         # 18
                    ]
                )

                # Save GPU time-series CSV
                gpu_csv = os.path.join(
                    args.out_dir,
                    f"{args.model.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_gpu_timeseries.csv",
                )
                if gpu_log:
                    gpu_count = len(gpu_log[0][1])
                    with open(gpu_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        header = (
                            ["Timestamp"]
                            + [f"GPU{i}_Power_W" for i in range(gpu_count)]
                            + [f"GPU{i}_Util_%" for i in range(gpu_count)]
                            + ["Total_Tokens", "Tokens_per_s"]
                        )
                        writer.writerow(header)
                        for ts, powers, utils, total_toks, tps in gpu_log:
                            writer.writerow(
                                [ts] + powers + utils + [total_toks, tps]
                            )

                # Save IPMI time-series CSV
                ipmi_csv = os.path.join(
                    args.out_dir,
                    f"{args.model.replace('/', '_')}_tp{tp}_cap{cap}_bs{bs}_ipmi_timeseries.csv",
                )
                if ipmi_log:
                    with open(ipmi_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "Timestamp",
                                "Sys_Power_W",
                                "CPU_Power_W",
                                "Mem_Power_W",
                                "GPU_Board_Power_W",
                                "Riser1_Power_W",
                            ]
                        )
                        for ts, sys_p, cpu_p, mem_p, gpu_b_p, riser_p in ipmi_log:
                            writer.writerow(
                                [ts, sys_p, cpu_p, mem_p, gpu_b_p, riser_p]
                            )

                # Plots
                plot_tokens_timeseries(
                    gpu_log, args.model, tp, cap, bs, args.out_dir
                )
                plot_power_timeseries(
                    gpu_log, args.model, tp, cap, bs, args.out_dir
                )
                plot_gpu_vs_system_power_overlay(
                    gpu_log, ipmi_log, args.model, tp, cap, bs, args.out_dir
                )

                # Cleanup
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2)

    # Restore default power caps
    set_power_cap(0)

    ############################################
    # Save summary CSV
    ############################################
    summary_file = os.path.join(
        args.out_dir,
        f"{args.model.replace('/', '_')}_hellaswag_tp_powercap_summary_fixed_duration.csv",
    )
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Tensor Parallel Size",  # 0
                "Power Cap (W)",         # 1
                "Batch Size",            # 2
                "Accuracy (%)",          # 3
                "Tokens/s",              # 4
                "Avg GPU Total Power (W)",   # 5
                "Per-GPU Power (W)",         # 6
                "Avg GPU Util (%)",          # 7
                "Per-GPU Util (%)",          # 8
                "GPU Energy (Wh)",           # 9
                "Tokens/s/W (GPU)",          # 10
                "Avg System Power (W)",      # 11
                "Avg CPU Power (W)",         # 12
                "Avg Mem Power (W)",         # 13
                "Avg GPU Board Power (W)",   # 14
                "Avg Riser1 Power (W)",      # 15
                "System Energy (Wh)",        # 16
                "Tokens/s/W (System)",       # 17
                "Elapsed Time (s)",          # 18
            ]
        )
        for row in summary_results:
            writer.writerow(row)

    ############################################
    # Summary plots (keep your existing style)
    ############################################
    model_tag = args.model
    # Indices: see summary_results above
    plot_metric(
        summary_results,
        3,
        "Accuracy (%)",
        "Accuracy vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_accuracy_vs_batch_size.png",
        ),
        model_tag,
    )
    plot_metric(
        summary_results,
        4,
        "Tokens/s",
        "Throughput vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_throughput_vs_batch_size.png",
        ),
        model_tag,
    )
    plot_metric(
        summary_results,
        5,
        "Avg GPU Total Power (W)",
        "GPU Power vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_gpu_power_vs_batch_size.png",
        ),
        model_tag,
    )
    plot_metric(
        summary_results,
        7,
        "Avg GPU Util (%)",
        "GPU Utilization vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_gpu_util_vs_batch_size.png",
        ),
        model_tag,
    )
    plot_metric(
        summary_results,
        10,
        "Tokens/s/W (GPU)",
        "GPU Energy Efficiency vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_gpu_efficiency_vs_batch_size.png",
        ),
        model_tag,
    )
    # System-level extra if you want them:
    plot_metric(
        summary_results,
        11,
        "Avg System Power (W)",
        "System Power vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_system_power_vs_batch_size.png",
        ),
        model_tag,
    )
    plot_metric(
        summary_results,
        17,
        "Tokens/s/W (System)",
        "System Energy Efficiency vs Batch Size",
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_system_efficiency_vs_batch_size.png",
        ),
        model_tag,
    )

    # Pareto plots
    plot_pareto(
        summary_results,
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_pareto_gpu_power_vs_throughput.png",
        ),
        use_system_power=False,
    )
    plot_pareto(
        summary_results,
        os.path.join(
            args.out_dir,
            f"{args.model.replace('/', '_')}_pareto_system_power_vs_throughput.png",
        ),
        use_system_power=True,
    )

    shutdown_nvml()

    print("\n=== Benchmark Complete (Fixed Duration Mode) ===")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

