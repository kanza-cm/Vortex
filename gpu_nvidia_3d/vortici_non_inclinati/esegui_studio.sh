#!/bin/bash
# Studio completo (fissa + adattiva) per i vortici non_inclinati, TMAX=30.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"
export CUDA="${CUDA:-$HOME/cuda-13.1}"
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$HERE/studio_errore_fissa" "$HERE/heatmap_adattiva"

echo "=== TILT=0: errore griglia fissa (L4-8, ref L8, T=30) ==="
TILT=0 LMIN=4 LMAX=8 T=30 CFL=0.5 NDIV0=1 CUDA="$CUDA" \
  OUT_DIR="$HERE/studio_errore_fissa" \
  bash "$ROOT/gpu_nvidia_3d/strumenti/studi/error_study_fissa_3d_gpu.sh" \
  > "$HERE/studio_errore_fissa.log" 2>&1
echo "   log: $HERE/studio_errore_fissa.log"

echo "=== TILT=0: heatmap griglia adattiva (L4-8, ref L8, T=30) ==="
TILT=0 MINLEVEL=4 LMIN=4 LMAX=8 TMAX=30 REF_RATIO=4 CFL=0.5 MAX_CELLS=15e6 \
  THR_LIST="0.3 0.1 3e-2" OUT="$HERE/heatmap_adattiva" \
  bash "$ROOT/gpu_amr_3d/strumenti/heatmap_adattiva_3d.sh" \
  > "$HERE/heatmap_adattiva.log" 2>&1
echo "   log: $HERE/heatmap_adattiva.log"
echo "=== TILT=0: FINE ==="
