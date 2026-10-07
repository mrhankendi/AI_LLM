#!/bin/bash
# Local workaround for vllm 0.31.0 + triton 3.7.1 (see docs/scc_L40S_notes.md).
# Wraps the optional Kimi-K3 / watermark / Qwen4 JIT warmups in try/except so a
# triton JIT crash in kernels we never use doesn't kill the workers.
# Re-run after every (re)install of vllm. Backup is saved as kernel_warmup.py.bak.
set -e
F=$(python - <<'PY'
import importlib.util, os
spec = importlib.util.find_spec("vllm")
print(os.path.join(os.path.dirname(spec.origin), "model_executor/warmup/kernel_warmup.py"))
PY
)
[ -f "$F.bak" ] || cp "$F" "$F.bak"
python - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "local patch: skip unrelated warmups" in s:
    print("already patched"); sys.exit(0)
old = """        kimi_k3_triton_warmup(worker)
        watermark_sample_warmup(worker)
        qwen4_exp_qsa_triton_warmup(worker)
"""
new = """        for _fn in (kimi_k3_triton_warmup, watermark_sample_warmup, qwen4_exp_qsa_triton_warmup):
            try:
                _fn(worker)
            except Exception as e:  # local patch: skip unrelated warmups
                logger.warning("Skipping %s: %r", _fn.__name__, e)
"""
assert old in s, "pattern not found; vllm version changed?"
open(p, "w").write(s.replace(old, new))
print("patched", p)
PY
