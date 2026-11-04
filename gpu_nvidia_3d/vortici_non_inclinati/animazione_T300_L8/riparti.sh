#!/bin/bash
# Riparte dallo stato salvato in chk_<T>.bin (default 300) e continua fino a TMAX2.
#   ./riparti.sh [T_da_cui_ripartire] [TMAX_nuovo]
# Esempio: ./riparti.sh 300 600
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
T0="${1:-300}"; T1="${2:-600}"
export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$ROOT/strumenti:$HOME/bin:$PATH:$BASILISK"
export CUDA="${CUDA:-$HOME/cuda-13.1}"
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RUN="$HERE/continuazione_T${T1}"
mkdir -p "$RUN"
cp -f "$HERE/saetta3d.c" "$RUN/" 2>/dev/null || cp -f "$ROOT/saetta3d.c" "$RUN/"
cp -f "$HERE/chk_$(printf '%.3f' "$T0").bin" "$RUN/"
EXTRA="-DTILT=0 -DNDIV0=1 -DMAXLEVEL=8 -DTMAX=$T1 -DMOVIE=1 -DCFL_USER=0.5 \
       -DVTK=0 -DMATRIX=1 -DCHECKPOINT=1 -DCHK_START=$T0"
cd "$RUN"
cp -f "$ROOT/saetta3d.c" . 2>/dev/null
if ! EXTRA="$EXTRA" bash "$ROOT/build.sh" > build.log 2>&1; then
    echo "BUILD FALLITA"; tail -20 build.log; exit 1
fi
cp -f "$ROOT/saetta3d_gpu" .
echo "=== riparto da t=$T0 fino a t=$T1 ==="
./saetta3d_gpu > log.txt 2>&1
echo "=== FINE: $(date) ==="
ls -la chk_*.bin omega3d*.mp4 2>/dev/null
