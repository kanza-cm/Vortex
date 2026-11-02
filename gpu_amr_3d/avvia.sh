#!/bin/sh
# Build + launch in background + status (same as _avvia_x.sh in gpu_nvidia_3d).
#
# Supports MINLEVEL/MAXLEVEL/REF_RATIO like run.sh:
#   ./avvia.sh                  # MINLEVEL=6 MAXLEVEL=8 (REF_RATIO=4)
#   ./avvia.sh 4 6
#   MINLEVEL=4 MAXLEVEL=6 REF_RATIO=4 ./avvia.sh
#
# Note: it deletes the previous outputs (plt*, log, pid).
set -e
cd "$(dirname "$0")"

rm -rf plt* log-saetta run.pid
echo "=== build (incflo AMR GPU, SINGLE precisione) ==="
./build.sh

echo "=== lancio in background ==="
. ./livelli.sh
./run.sh "$@"

sleep 60
echo "=== stato dopo 60s  (MINLEVEL=$MINLEVEL MAXLEVEL=$MAXLEVEL REF_RATIO=$REF_RATIO -> n_cell = $AMR_NCELL, max_level = $AMR_MAXLEV, ref_ratio = \"$AMR_REFRATIO\") ==="
ps -p "$(cat run.pid)" -o pid,etime,stat 2>/dev/null || echo "PROCESSO TERMINATO"
tail -3 log-saetta
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
