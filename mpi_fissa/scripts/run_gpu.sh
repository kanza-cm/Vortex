#!/bin/bash
# run_gpu.sh -- esegue il prototipo CUDA.
#   ./scripts/run_gpu.sh 1 -tmax 10                          # 1 GPU (1 rank)
#   ./scripts/run_gpu.sh 2 -tmax 1                           # 2 rank sulla STESSA GPU
#                                                            # (test della decomposizione)
# Per il run su 2 PC vedi run_2pc.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
NP="${1:?uso: run_gpu.sh <1|2> [opzioni]}"; shift
BIN=${BIN:-bin/mpi_fissa_gpu}
[ -x "$BIN" ] || { echo "manca $BIN: lancia prima ./scripts/build.sh gpu"; exit 2; }
CUDA="${CUDA:-$HOME/cuda-13.1}"
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec mpirun -np "$NP" --bind-to none "$BIN" "$@"
