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

############################################
# Power + Utilization logging thread
############################################
def log_power_util(gpu_log, stop_flag, interval=0.5):
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
            entries = [line.split(",") for line in result.stdout.strip().split("\n") if line]
            powers = [float(e[0]) for e in entries]
            utils = [float(e[1]) for e in entries]
            gpu_log.append((time.time(), powers, utils))
            time.sleep(interval)
        except Exception:
            break

############################################
# Load HellaSwag dataset
############################################
dataset = load_dataset("hellaswag", split="validation")  # full validation set
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
# Configurations
############################################
model_name = "meta-llama/Llama-2-7b-hf"
tensor_parallel = 4
gpu_mem_util = 0.9
max_len = 1024
batch_sizes = [4, 8, 16]

summary_results = []

############################################
# Run benchmarks for each batch size
############################################
for bs in batch_sizes:
    print(f"\n=== Running batch size {bs} ===")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_len
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)

    gpu_log = []
    stop_flag = {"stop": False}
    log_thread = threading.Thread(target=log_power_util, args=(gpu_log, stop_flag), daemon=True)
    log_thread.start()

    start_time = time.time()
    correct = 0
    total_tokens = 0

    for i in range(0, len(prompts), bs):
        batch_prompts = prompts[i:i+bs]
        batch_answers = answers[i:i+bs]
        outputs = llm.generate(batch_prompts, sampling_params)

        for ref_answer, output in zip(batch_answers, outputs):
            text = output.outputs[0].text.strip()
            total_tokens += len(output.outputs[0].token_ids)
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
    tokens_per_sec = total_tokens / elapsed_time

    if gpu_log:
        per_gpu_power = [mean([p[i] for _, p, _ in gpu_log]) for i in range(len(gpu_log[0][1]))]
        per_gpu_util = [mean([u[i] for _, _, u in gpu_log]) for i in range(len(gpu_log[0][2]))]
        avg_total_power = mean([sum(p) for _, p, _ in gpu_log])
        avg_total_util = mean([mean(u) for _, _, u in gpu_log])
        total_energy_Wh = avg_total_power * (elapsed_time / 3600)
    else:
        per_gpu_power = []
        per_gpu_util = []
        avg_total_power = 0
        avg_total_util = 0
        total_energy_Wh = 0

    tokens_per_watt = tokens_per_sec / avg_total_power if avg_total_power > 0 else 0

    summary_results.append([
        bs, accuracy, tokens_per_sec, avg_total_power, per_gpu_power,
        avg_total_util, per_gpu_util, total_energy_Wh, tokens_per_watt
    ])

    # Save per-GPU log
    with open(f"hellaswag_gpu_log_bs{bs}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp"] +
                        [f"GPU{i}_Power_W" for i in range(len(per_gpu_power))] +
                        [f"GPU{i}_Util_%" for i in range(len(per_gpu_util))])
        for ts, powers, utils in gpu_log:
            writer.writerow([ts] + powers + utils)

    # ==== GPU MEMORY CLEANUP ====
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # short pause to let CUDA free memory

############################################
# Save summary CSV
############################################
with open("hellaswag_vllm_batch_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Batch Size", "Accuracy (%)", "Tokens/s", "Avg Total Power (W)",
        "Per-GPU Power (W)", "Avg Total Util (%)", "Per-GPU Util (%)",
        "Total Energy (Wh)", "Tokens/s/W"
    ])
    for row in summary_results:
        writer.writerow(row)

############################################
# Plots
############################################
batch_sizes_list = [row[0] for row in summary_results]
accuracy_list = [row[1] for row in summary_results]
tps_list = [row[2] for row in summary_results]
power_list = [row[3] for row in summary_results]
util_list = [row[5] for row in summary_results]
tpw_list = [row[8] for row in summary_results]

plt.figure()
plt.plot(batch_sizes_list, accuracy_list, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Batch Size")
plt.grid(True)
plt.savefig("accuracy_vs_batch_size.png")

plt.figure()
plt.plot(batch_sizes_list, tps_list, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Tokens/s")
plt.title("Throughput vs Batch Size")
plt.grid(True)
plt.savefig("throughput_vs_batch_size.png")

plt.figure()
plt.plot(batch_sizes_list, power_list, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Avg Total Power (W)")
plt.title("Power vs Batch Size")
plt.grid(True)
plt.savefig("power_vs_batch_size.png")

plt.figure()
plt.plot(batch_sizes_list, util_list, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Avg GPU Util (%)")
plt.title("GPU Utilization vs Batch Size")
plt.grid(True)
plt.savefig("gpu_util_vs_batch_size.png")

plt.figure()
plt.plot(batch_sizes_list, tpw_list, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Tokens/s/W")
plt.title("Energy Efficiency vs Batch Size")
plt.grid(True)
plt.savefig("efficiency_vs_batch_size.png")

print("\n=== Benchmark Complete ===")
print("Results saved to hellaswag_vllm_batch_summary.csv and hellaswag_gpu_log_bs*.csv")
print("Plots saved as PNG files")

