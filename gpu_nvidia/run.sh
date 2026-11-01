#!/bin/sh
# Runs the GPU solver (single precision) in the current directory.
# The outputs land here (as in the CPU run, but in this subfolder):
#   omega.mp4        vorticity movie (zoomed view)
#   omega.ppm        final frame
#   diagnostics.dat  time series for the plots
#   omega_matrix.bin final field on a regular grid
#   resa_max.dat     level, Poisson residual maximum and its time
#   log-saetta       solver output
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
# ppm2mp4 (a Basilisk script) must be in PATH, otherwise the
# movie falls back to raw PPM frames.
export PATH="$PATH:$BASILISK"
# libnvrtc is needed for kernel JIT; libcuda.so.1 comes from the
# Windows driver exposed by WSL (/usr/lib/wsl/lib).
CUDA="${CUDA:-/usr/local/cuda-13.1/targets/x86_64-linux}"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Avvio GPU: ./saetta_gpu  (log in log-saetta)"
./saetta_gpu 2> log-saetta
echo "Fatto."
