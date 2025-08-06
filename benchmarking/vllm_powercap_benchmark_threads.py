
import openai
import time
import statistics
import subprocess
from datasets import load_dataset
from threading import Thread
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os

# === CONFIG ===
DATASET_NAME = "hellaswag"
SPLIT = "validation"
NUM_QUESTIONS = 100
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROMPT_TEMPLATE = "Context: {ctx}\n\nOptions:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nQ: Which option best completes the sentence?\nA:"
VLLM_URL = "http://localhost:8000/v1"
GPU_INDICES = [0, 1, 2, 3]
POWER_CAPS = [250, 200, 175, 150]

def set_power_cap(watts):
    for idx in GPU_INDICES:
        try:
            subprocess.run(
                f"sudo nvidia-smi -i {idx} -pl {watts}",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to set power cap for GPU {idx}: {e.stderr.decode()}")

def log_power_all_gpus(power_log, gpu_indices, stop_flag):
    while not stop_flag[0]:
        try:
            timestamp = time.time()
            powers = []
            for idx in gpu_indices:
                result = subprocess.check_output(
                    f"nvidia-smi --id={idx} --query-gpu=power.draw --format=csv,noheader,nounits",
                    shell=True
                )
                powers.append(float(result.decode().strip()))
            power_log.append((timestamp, powers))
        except Exception:
            break
        time.sleep(0.5)

from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 32  # Parallelism for batching

def run_benchmark(client, dataset, num_questions, prompt_template):
    total_tokens = 0
    labels = ["A", "B", "C", "D"]

    def infer(i_sample):
        i, sample = i_sample
        result = {"latency": None, "correct": False, "tokens": 0}
        try:
            ctx = sample["ctx"]
            endings = sample["endings"]
            label = int(sample["label"])
            prompt = prompt_template.format(
                ctx=ctx.strip(),
                a=endings[0],
                b=endings[1],
                c=endings[2],
                d=endings[3]
            )
            expected = labels[label]
            if i == 0:
                print(f"\n[Prompt Preview]\n{prompt}\nExpected: {expected}\n")
            start = time.time()
            response = client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                max_tokens=5,
                temperature=0.0,
            )
            end = time.time()
            latency = end - start
            output = response.choices[0].text.strip().upper()
            prediction = output[0] if output else "?"
            is_correct = prediction == expected
            tokens = len(output.split())
            result["latency"] = latency
            result["correct"] = is_correct
            result["tokens"] = tokens
            print(f"[{i+1}] Latency: {latency:.2f}s | Correct: {is_correct} | Pred: {prediction} | Output: {output} | Tokens: {tokens}")
        except Exception as e:
            print(f"[{i+1}] Failed: {e}")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for result in executor.map(infer, list(enumerate(dataset[:num_questions]))):
            results.append(result)
    latencies = [r["latency"] for r in results if r["latency"] is not None]
    correct = sum(1 for r in results if r["correct"])
    total_tokens = sum(r["tokens"] for r in results)
    return latencies, correct, total_tokens

def compute_power_stats(power_log, gpu_indices):
    valid_powers = [p for t, p in power_log if isinstance(p, list)]
    if valid_powers:
        avg_power_per_gpu = [statistics.mean([gpu[i] for gpu in valid_powers]) for i in range(len(gpu_indices))]
        total_avg_power = sum(avg_power_per_gpu)
    else:
        avg_power_per_gpu = [0.0] * len(gpu_indices)
        total_avg_power = 0.0
    return avg_power_per_gpu, total_avg_power

def save_power_timeseries(power_log, gpu_indices, filename):
    with open(filename, "w") as pf:
        writer = csv.writer(pf)
        writer.writerow(["timestamp"] + [f"gpu_{i}" for i in gpu_indices])
        for ts, powers in power_log:
            if ts == "STOP":
                continue
            writer.writerow([ts] + powers)

def save_latency_timeseries(latencies, filename):
    with open(filename, "w") as lf:
        writer = csv.writer(lf)
        writer.writerow(["index", "latency"])
        for i, latency in enumerate(latencies):
            writer.writerow([i, latency])

def plot_power(filename, out_file):
    power_df = pd.read_csv(filename)
    plt.figure(figsize=(12, 6))
    for i in range(1, len(power_df.columns)):
        plt.plot(power_df["timestamp"], power_df.iloc[:, i], label=f"GPU{i-1}")
    plt.xlabel("Timestamp")
    plt.ylabel("Power (W)")
    plt.title("GPU Power Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_latency(filename, out_file):
    latency_df = pd.read_csv(filename)
    plt.figure(figsize=(12, 4))
    plt.plot(latency_df["index"], latency_df["latency"], marker='o', linestyle='-')
    plt.xlabel("Inference Index")
    plt.ylabel("Latency (s)")
    plt.title("Inference Latency Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def save_benchmark_summary(power_w, model_name, dataset_name, split, num_questions, correct, latencies, avg_power_per_gpu, total_avg_power, total_tokens, filename):
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            power_w,
            model_name,
            f"{dataset_name}/{split}",
            num_questions,
            f"{100*correct/num_questions:.2f}%",
            f"{statistics.mean(latencies):.2f}" if latencies else "NA",
            total_tokens,
            f"{total_tokens/statistics.mean(latencies):.2f}" if latencies and statistics.mean(latencies) > 0 else "NA",
            f"{total_avg_power/total_tokens:.4f}" if total_tokens > 0 else "NA",
            *[f"{p:.2f}" for p in avg_power_per_gpu],
            f"{total_avg_power:.2f}"
        ])

def main():
    client = openai.OpenAI(base_url=VLLM_URL, api_key="not-needed")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"Loaded {len(dataset)} samples from {DATASET_NAME}/{SPLIT}")

    for power_cap in POWER_CAPS:
        print(f"\n===== Running Benchmark with Power Cap: {power_cap}W =====")
        set_power_cap(power_cap)

        power_log = []
        stop_flag = [False]
        thread = Thread(target=log_power_all_gpus, args=(power_log, GPU_INDICES, stop_flag))
        thread.daemon = True
        thread.start()

        latencies, correct, total_tokens = run_benchmark(client, dataset, NUM_QUESTIONS, PROMPT_TEMPLATE)

        stop_flag[0] = True
        thread.join(timeout=2)
        power_log.append(("STOP", [0] * len(GPU_INDICES)))

        avg_power_per_gpu, total_avg_power = compute_power_stats(power_log, GPU_INDICES)

        tag = f"{power_cap}W"
        save_power_timeseries(power_log, GPU_INDICES, filename=f"power_{tag}.csv")
        save_latency_timeseries(latencies, filename=f"latency_{tag}.csv")
        plot_power(filename=f"power_{tag}.csv", out_file=f"power_plot_{tag}.png")
        plot_latency(filename=f"latency_{tag}.csv", out_file=f"latency_plot_{tag}.png")
        save_benchmark_summary(power_cap, MODEL_NAME, DATASET_NAME, SPLIT, NUM_QUESTIONS, correct, latencies, avg_power_per_gpu, total_avg_power, total_tokens, filename=f"summary_{tag}.csv")

    print("\n=== Benchmark Complete ===")
    set_power_cap(250)

if __name__ == "__main__":
    main()
