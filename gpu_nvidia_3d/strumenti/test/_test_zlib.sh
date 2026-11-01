#!/bin/sh
set -e
export BASILISK=/home/zorka/basilisk/src
CUDA=/usr/local/cuda-13.1/targets/x86_64-linux
export PATH="$PATH:$BASILISK"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d"

echo "=== build di prova (MAXLEVEL=5, TMAX=0.3) con zlib ==="
EXTRA="-DMAXLEVEL=5 -DTMAX=0.3 -DVTK=1" ./build.sh
mv saetta3d_gpu saetta3d_test

echo "=== eseguo in /tmp/vti_prova ==="
rm -rf /tmp/vti_prova; mkdir -p /tmp/vti_prova
cd /tmp/vti_prova
time "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/saetta3d_test" 2> log
grep "vtk" log
ls -la vtk3d/vtk3d_*.vti

echo "=== verifico il file compresso con pyvista ==="
/home/zorka/.venvs/vis/bin/python3 - <<EOF
import glob, pyvista as pv, numpy as np
g = pv.read(sorted(glob.glob("vtk3d/vtk3d_*.vti"))[0])
wz = g["omega_z"]; wm = g["omega_mag"]
print("dims:", g.dimensions, "punti:", g.n_points)
print("omega_z  max=%.4f min=%.4f media=%.2e" % (wz.max(), wz.min(), wz.mean()))
print("omega_mag max=%.4f" % wm.max())
EOF
