#!/bin/bash
# run_moe_benchmarks.sh
# Loop over multiple models, TP sizes, and power caps for benchmarking
set -euo pipefail

############################################
# CONFIGURATION
############################################
# Tasks to run (toggle as needed)
TASKS=("hellaswag" "gsm8k")

# Models to sweep
MODELS=(
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "Qwen/Qwen1.5-MoE-A2.7B"
  "allenai/OLMoE-1B-7B-0924"
  "meta-llama/Llama-2-7b-hf"
  "openai-community/gpt2"
)

# Parallelism / batch / power caps
TP_SIZES="4"
BATCH_SIZES_HELLA="1 4 8 16 32 64"
BATCH_SIZES_GSM="1 2 4 8 16 32"      # GSM8K outputs are longer; keep batches modest
POWER_CAPS="400 350 300 250 200 150 100"

# Memory/utilization
GPU_UTIL=0.90

# Context limits (adjust if you OOM)
MAX_LEN_HELLA=1024
MAX_LEN_GSM=3072
MAX_NEW_TOKENS_GSM=512
FEWSHOT_GSM=0                 # set to 8 for 8-shot GSM8K

# Dataset subsets (fast sanity runs)
SUBSET_HELLA="validation[:200]"
SUBSET_GSM="test[:200]"

# Paths
BENCH_SCRIPT="/home/cc/vLLM/run_eval.py"     # <-- new Python script path
CLEAN_SCRIPT="/home/cc/vLLM/kill_GPU_jobs.sh"

# Optional: allocator settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

############################################
# PREP
############################################
echo "[INFO] Cleaning GPU VRAM..."
bash "$CLEAN_SCRIPT" || true

############################################
# MAIN LOOP
############################################
for MODEL in "${MODELS[@]}"; do
  echo "============================"
  echo "[INFO] Starting benchmarks for model: $MODEL"
  echo "============================"

  for TASK in "${TASKS[@]}"; do
    if [[ "$TASK" == "hellaswag" ]]; then
      BSS="$BATCH_SIZES_HELLA"
      SUBSET="$SUBSET_HELLA"
      MAX_LEN="$MAX_LEN_HELLA"
      EXTRA_ARGS=()   # none
    else
      BSS="$BATCH_SIZES_GSM"
      SUBSET="$SUBSET_GSM"
      MAX_LEN="$MAX_LEN_GSM"
      EXTRA_ARGS=(--fewshot "$FEWSHOT_GSM" --max-new-tokens "$MAX_NEW_TOKENS_GSM")
    fi

    echo "[INFO] Task=$TASK | Model=$MODEL"
    echo "[INFO] TP_SIZES=[$TP_SIZES] | POWER_CAPS=[$POWER_CAPS] | BATCH_SIZES=[$BSS]"
    echo "[INFO] SUBSET=$SUBSET | MAX_LEN=$MAX_LEN"

    python3 "$BENCH_SCRIPT" \
      --task "$TASK" \
      --model "$MODEL" \
      --tensor-parallel-sizes $TP_SIZES \
      --power-caps $POWER_CAPS \
      --batch-sizes $BSS \
      --gpu-util $GPU_UTIL \
      --max-len $MAX_LEN \
      --subset "$SUBSET" \
      "${EXTRA_ARGS[@]}"

    echo "[INFO] Cleaning up GPU memory..."
    bash "$CLEAN_SCRIPT" || true
    echo "[INFO] Completed $TASK for $MODEL"
    echo
  done

  echo "[INFO] Completed all tasks for $MODEL"
  echo
done

echo "=== All benchmarks completed! ==="
bash "$CLEAN_SCRIPT" || true
