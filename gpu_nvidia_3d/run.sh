#!/bin/sh
# Runs the 3D GPU solver (single precision) in the current folder.
# Output:
#   omega3d_node.mp4  x-y slices at z = LZ/2 (node, early merging)
#   omega3d_anti.mp4  x-y slices at z = 0 (antinode)
#   omega3d.ppm       final frame (node slice)
#   vtk3d/*.vti       3D fields for ParaView (only with -DVTK=1)
#   carica_vti.log    log of the uploads to Drive (with -DVTK_UPLOAD=1)
#   diagnostics.dat   time series
#   resa_max.dat      level and max residual of the pressure Poisson solve
#   log-saetta        solver output
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
# ppm2mp4 (a Basilisk script) must be in PATH, otherwise the
# movie falls back to raw PPM frames.
export PATH="$PATH:$BASILISK"
# libnvrtc is needed to JIT the kernels; libcuda.so.1 comes from the
# Windows driver exposed by WSL (/usr/lib/wsl/lib).
CUDA="${CUDA:-/usr/local/cuda-13.1/targets/x86_64-linux}"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Avvio GPU 3D: ./saetta3d_gpu  (log in log-saetta)"
./saetta3d_gpu 2> log-saetta
echo "Fatto."
