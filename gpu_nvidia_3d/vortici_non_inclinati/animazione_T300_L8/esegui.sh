#!/bin/bash
# Animazione (movie) dei vortici NON inclinati (TILT=0), griglia fissa L8,
# TMAX=300, single precision, GPU. Alla fine scrive chk_300.000.bin per
# poter ripartire (vedi riparti.sh).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$ROOT/strumenti:$HOME/bin:$PATH:$BASILISK"
export CUDA="${CUDA:-$HOME/cuda-13.1}"
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXTRA="-DTILT=0 -DNDIV0=1 -DMAXLEVEL=8 -DTMAX=300 -DMOVIE=1 -DCFL_USER=0.5 \
       -DVTK=0 -DMATRIX=1 -DCHECKPOINT=1 -DMOVIE_TSTEP=2."
echo "=== build (MAXLEVEL=8, TILT=0, TMAX=300) ==="
if ! EXTRA="$EXTRA" bash "$ROOT/build.sh" > build.log 2>&1; then
    echo "BUILD FALLITA"; tail -20 build.log; exit 1
fi
cp -f "$ROOT/saetta3d_gpu" "$HERE/saetta3d_gpu"
cd "$HERE"
echo "=== run T=300, movie + checkpoint finale ==="
./saetta3d_gpu > log.txt 2>&1
echo "=== FINE: $(date) ==="
ls -la chk_*.bin omega3d*.mp4 2>/dev/null
