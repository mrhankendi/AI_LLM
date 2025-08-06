#!/usr/bin/env python3
"""
cleanup_gpu_memory.py
Forcefully free GPU VRAM after unfinished ML scripts.
"""

import os
import subprocess
import time
import gc

def kill_python_gpu_processes():
    print("[INFO] Checking for Python processes using the GPU...")
    try:
        # Find processes using GPU via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 2:
                pid, pname = parts
                if "python" in pname.lower():
                    print(f"[INFO] Killing process PID={pid}, Name={pname}")
                    try:
                        os.kill(int(pid), 9)
                    except ProcessLookupError:
                        pass
    except Exception as e:
        print(f"[WARN] Could not query nvidia-smi: {e}")

def clear_cuda_cache():
    try:
        import torch
        print("[INFO] Clearing PyTorch CUDA cache...")
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        print("[INFO] PyTorch not available, skipping torch cache clear.")

def main():
    kill_python_gpu_processes()
    clear_cuda_cache()
    print("[INFO] Waiting for VRAM to be released...")
    time.sleep(2)
    subprocess.run(["nvidia-smi"])

if __name__ == "__main__":
    main()

