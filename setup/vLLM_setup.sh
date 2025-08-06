
mkdir ~/vLLM
python3 -m venv vllm-env
source vllm-env/bin/activate

python -m pip install --upgrade pip
pip install vllm matplotlib datasets



python moe_hellaswag_benchmark.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --batch-sizes 1 4 8 16 \
  --tensor-parallel 4

python moe_hellaswag_benchmark.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --batch-sizes 1 4 8 16 \
  --tensor-parallel 4


python moe_hellaswag_benchmark.py \
  --model deepseek-ai/DeepSeek-V3 \
  --batch-sizes 1 4 8 16 \
  --tensor-parallel 4

python moe_hellaswag_benchmark.py \
  --model 01-ai/Yi-MoE-9B \
  --batch-sizes 16 \
  --tensor-parallel 4

python3 moe_power_cap_hellaswag.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --batch-sizes 1 \
  --tensor-parallel-sizes 1 \
  --power-caps 0 \
  --gpu-util 0.9 \
  --max-len 1024 \
  --subset validation[:500]
