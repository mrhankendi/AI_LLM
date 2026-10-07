#!/bin/bash
# Loop over multiple models, TP sizes, and power caps for benchmarking

# === ENVIRONMENT (SCC-501, L40S) ===
# Home quota is tiny, so keep every cache on node-local /scratch.
# See docs/scc_L40S_notes.md for why.
export HF_HOME=/scratch/$USER/hf_cache
export VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/$USER/vllm_config
export TORCH_HOME=/scratch/$USER/torch
export TRITON_CACHE_DIR=/scratch/$USER/triton_cache
export XDG_CACHE_HOME=/scratch/$USER/xdg_cache
export TMPDIR=/scratch/$USER/tmp
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$FLASHINFER_WORKSPACE_BASE"

# FlashInfer's sampler JIT-compiles with nvcc; torch here is built for CUDA 13.0.
# Activate beforehand: module load python3/3.13.8 cuda/13.2;
#                      source /scratch/$USER/vllm-env/bin/activate
export CUDA_HOME=/share/pkg.8/cuda/13.2/install

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
TP_SIZES="4"
DP_SIZES="1"
POWER_CAPS="400" # 350 300 250 200 150 100"
GPU_UTIL=0.90
MAX_LEN=1024
ENABLE_EP=true
SUBSET="validation"  # full HellaSwag validation split (10,042 examples)

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

    python3 "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --batch-sizes $BATCH_SIZES \
        --tensor-parallel-sizes $TP_SIZES \
        --data-parallel-sizes $DP_SIZES \
        --max-len $MAX_LEN \
        --power-caps $POWER_CAPS \
        --gpu-util $GPU_UTIL \
        --subset $SUBSET \
        --enable-ep $ENABLE_EP

    echo "[INFO] Cleaning up GPU memory..."
    bash "$CLEAN_SCRIPT"

    echo "[INFO] Completed benchmarks for $MODEL $BATCH_SIZES $TP_SIZES $POWER_CAPS"
    echo
done

echo "=== All benchmarks completed! ==="

bash "$CLEAN_SCRIPT"
