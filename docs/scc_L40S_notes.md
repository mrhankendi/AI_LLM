# SCC-501 (L40S) setup notes

Environment notes from getting `benchmarking/run_power_caps.sh` running on the
SCC-501 node (4x NVIDIA L40S). Versions at the time: Python 3.13.8,
vllm 0.31.0, torch 2.13.0 (CUDA 13.0), triton 3.7.1.

## 1. Disk quota problems (the big one)

**Symptoms**
- `OSError: [Errno 122] Disk quota exceeded` while downloading the model
  (HF `xet_get`), while `pip install`-ing, and in vLLM's usage-stats writer
  (`~/.config/vllm/usage_stats.json`).
- A forced `pip install --force-reinstall triton` rolled back on the same error.

**Cause**
Home (`/usr2/...`, GPFS) has a small per-user quota (~10 GB). The venv alone was
8.5 GB and `~/.cache` another 1.8 GB. A 16B model is ~33 GB, so nothing fit.
`/`, `/tmp` (2 GB), and `/home` (2 GB) are also too small. `df -h` showed
`/scratch` as local NVMe with ~800 GB free.

**What we did**
1. Freed home: `rm -rf ~/.cache/{huggingface,pip,vllm,torch,triton}`.
2. Rebuilt the venv on scratch (after `module load python3/3.13.8`), with pip's
   cache and temp dir also on scratch:
   ```bash
   module load python3/3.13.8
   export PIP_CACHE_DIR=/scratch/$USER/pip_cache TMPDIR=/scratch/$USER/tmp
   mkdir -p $PIP_CACHE_DIR $TMPDIR
   python3 -m venv /scratch/$USER/vllm-env
   source /scratch/$USER/vllm-env/bin/activate
   pip install --upgrade pip && pip install vllm matplotlib datasets
   ```
3. Redirected every cache to `/scratch/$USER/...` in `run_power_caps.sh`
   (`HF_HOME`, `VLLM_CACHE_ROOT`, `VLLM_CONFIG_ROOT`, `TORCH_HOME`,
   `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`, `FLASHINFER_WORKSPACE_BASE`)
   and disabled vLLM usage stats (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`).

**Caveats**
- `/scratch` is **node-local**: the venv and model cache exist only on
  `scc-501` and may not survive reboots/cleanups. If missing, repeat step 2 and
  re-download the model. For multi-node use, put these under `/projectnb/...`
  (check space with `pquota`).
- `HF_TOKEN`/`huggingface-cli login` is stored under `$HF_HOME`; log in after
  setting it (not needed for the non-gated deepseek-moe-16b).

## 2. Wrong Python in the venv
`pip install vllm` -> "No matching distribution found". The venv had been
created with the system Python 3.6 (pip 9.0.3/21.3.1) because the Python module
wasn't loaded. vLLM needs Python >= 3.9. Always `module load python3/3.13.8`
*before* `python3 -m venv`, and check `python --version` inside the venv.

## 3. Script fixes
- `load_dataset("hellaswag")` fails on recent `datasets`/`huggingface_hub`
  ("Repository id must be 'namespace/name'"). Now `Rowan/hellaswag`.
- `--max-len: command not found` in the wrapper came from broken `\` line
  continuations (stray whitespace/tab indentation; check for CRLF on the
  cluster copy with `cat -A`).
- Full-dataset run: `SUBSET="validation"` (10,042 prompts) and no
  `--time-budget` (default `0` = run until the dataset is exhausted).

## 4. vLLM / triton crash in warmup
`AttributeError: 'NoneType' object has no attribute 'start'` at
`triton/runtime/jit.py:470`, raised from `kimi_k3_triton_warmup`. vllm 0.31.0's
optional JIT warmups import Kimi-K3 / DeepSeek-V4 kernels that triton 3.7.1
can't parse, and the exception kills every TP worker even though we don't use
those models. Reinstalling did not help (fresh venv, same error).
Workaround: `setup/patch_vllm_warmup.sh` wraps those warmups in try/except
(backup kept as `kernel_warmup.py.bak`). Re-run it after reinstalling vllm.
Consider reporting upstream (vllm 0.31.0 / triton 3.7.1 / torch 2.13.0).

## 5. FlashInfer needs nvcc
`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist` during sampler warmup (FlashInfer JIT-compiles its top-k/top-p
sampler). Torch is built for CUDA 13.0, so:
```bash
module load cuda/13.2
export CUDA_HOME=/share/pkg.8/cuda/13.2/install
```
(Fallback: `export VLLM_USE_FLASHINFER_SAMPLER=0`; verify the name exists in
`vllm/envs.py` for your version.) Compiled kernels go to
`FLASHINFER_WORKSPACE_BASE` on scratch. `module load` may be unavailable in a
non-interactive script, so load modules in your shell before launching.

## 6. Harmless warnings
- `Cubin file saved by TritonBundler not found` — stale torch.compile cache;
  torch recompiles. Delete `$VLLM_CACHE_ROOT/torch_compile_cache` to silence.
- `Using default MoE config ... E=60,N=352,device_name=NVIDIA_L40S.json not
  found` — no tuned fused-MoE config for L40S; keep it constant across runs.

## 7. Open item: IPMI power reads NaN
`log_ipmi` leaves a sensor as NaN when `ipmitool` prints nothing parseable, and
stderr is discarded. Test by hand:
```bash
ipmitool -I open sensor reading "Sys Power" "CPU Power" "Mem Power" "GPU Board Power" "Riser 1 Power"
sudo ipmitool -I open sensor | grep -i -E "power|watt"
```
Likely causes: no permission on `/dev/ipmi0`, or different sensor names on
this node.

## Per-session checklist
```bash
ssh scc-501
module load python3/3.13.8 cuda/13.2
source /scratch/$USER/vllm-env/bin/activate
cd ~/AI_LLM/benchmarking
tmux new -s bench          # so ssh drops don't kill the run
bash run_power_caps.sh 2>&1 | tee run.log
```
Power caps must lie within `nvidia-smi -q -d POWER` Min/Max Power Limit.
