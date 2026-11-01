#!/bin/sh
set -e
export BASILISK=/home/zorka/basilisk/src
CUDA=/usr/local/cuda-13.1/targets/x86_64-linux
export PATH="$PATH:$BASILISK"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d"

rm -rf diagnostics.dat omega3d* vtk3d finale_t50.png resa_max.dat log-* carica_vti.log
echo "=== build (tubi inclinati a X, TILT=2) ==="
./build.sh

echo "=== lancio in background ==="
nohup ./saetta3d_gpu 2> log-saetta &
echo $! > run.pid
sleep 60
echo "=== stato dopo 60s ==="
ps -p "$(cat run.pid)" -o pid,etime,stat 2>/dev/null || echo "PROCESSO TERMINATO"
tail -3 log-saetta
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
