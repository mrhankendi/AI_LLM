#!/bin/bash
# run_moe_benchmarks.sh
# Loop over multiple models, TP sizes, and power caps for benchmarking

# === CONFIGURATION ===
MODELS=(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "Qwen/Qwen1.5-MoE-A2.7B"
    "allenai/OLMoE-1B-7B-0924"
    "meta-llama/Llama-2-7b-hf"
    "openai-community/gpt2"
)
BATCH_SIZES="1 4 8 16"
TP_SIZES="2 4"
POWER_CAPS="300 250 200 150"
GPU_UTIL=0.90
MAX_LEN=1024
SUBSET="validation"  # smaller subset for faster runs

# Path to your Python benchmark script
BENCH_SCRIPT="/home/cc/vLLM/moe_power_cap_hellaswag_2.py"
# Path to your GPU cleanup script
CLEAN_SCRIPT="/home/cc/vLLM/kill_GPU_jobs.sh"

echo "[INFO] Cleaning GPU VRAM..."
bash "$CLEAN_SCRIPT"

# === LOOP ===
for MODEL in "${MODELS[@]}"; do
    echo "============================"
    echo "[INFO] Starting benchmarks for model: $MODEL"
    echo "============================"

    python3 "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --batch-sizes $BATCH_SIZES \
        --tensor-parallel-sizes $TP_SIZES \
	--max-len $MAX_LEN \
        --power-caps $POWER_CAPS \
        --gpu-util $GPU_UTIL \
        --subset $SUBSET

    echo "[INFO] Cleaning up GPU memory..."
    bash "$CLEAN_SCRIPT"

    echo "[INFO] Completed benchmarks for $MODEL $BATCH_SIZES $TP_SIZES $POWER_CAPS"
    echo
done

echo "=== All benchmarks completed! ==="

bash "$CLEAN_SCRIPT"
