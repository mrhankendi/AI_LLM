#!/bin/bash
# kill_gpu_jobs.sh
# Forcefully free GPU VRAM from leftover Python processes

echo "[INFO] Checking for Python processes using the GPU..."
PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits | grep -i python | awk -F',' '{print $1}')

if [ -z "$PIDS" ]; then
	    echo "[INFO] No Python GPU processes found."
    else
	        echo "[INFO] Killing the following PIDs: $PIDS"
		    for PID in $PIDS; do
			            kill -9 "$PID" 2>/dev/null
				        done
fi

echo "[INFO] Waiting for VRAM to be released..."
sleep 2

nvidia-smi

