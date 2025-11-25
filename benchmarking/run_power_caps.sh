#!/bin/bash
# Loop over multiple models, TP sizes, and power caps for benchmarking

# === CONFIGURATION ===
MODELS=(
   "deepseek-ai/deepseek-moe-16b-base"
   #"mistralai/Mixtral-8x7B-Instruct-v0.1"
   # "Qwen/Qwen1.5-MoE-A2.7B"
   # "allenai/OLMoE-1B-7B-0924"
   # "meta-llama/Llama-2-7b-hf"
   # "openai-community/gpt2"
)
BATCH_SIZES="64"
TP_SIZES="2"
DP_SIZES="2"
POWER_CAPS="400" # 350 300 250 200 150 100"
GPU_UTIL=0.90
MAX_LEN=1024
ENABLE_EP=true
SUBSET="validation[:500]"  # smaller subset for faster runs

# Path to your Python benchmark script
BENCH_SCRIPT="vllm_benchmark_fixed_wallclock.py"

# Path to your GPU cleanup script
CLEAN_SCRIPT="../tools/kill_GPU_jobs.sh"

echo "[INFO] Cleaning GPU VRAM..."
bash "$CLEAN_SCRIPT"

# === LOOP ===
for MODEL in "${MODELS[@]}"; do
    echo "============================"
    echo "[INFO] Starting benchmarks for model: $MODEL"
    echo "============================"

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$DP_SIZE "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --batch-sizes $BATCH_SIZES \
        --tensor-parallel-sizes $TP_SIZES \
	    --max-len $MAX_LEN \
        --power-caps $POWER_CAPS \
        --gpu-util $GPU_UTIL \
        --subset $SUBSET \
	    --time-budget 300 \
        --enable-ep $ENABLE_EP

    echo "[INFO] Cleaning up GPU memory..."
    bash "$CLEAN_SCRIPT"

    echo "[INFO] Completed benchmarks for $MODEL $BATCH_SIZES $TP_SIZES $POWER_CAPS"
    echo
done

echo "=== All benchmarks completed! ==="

bash "$CLEAN_SCRIPT"
