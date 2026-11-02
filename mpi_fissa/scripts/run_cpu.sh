#!/bin/bash
# run_cpu.sh -- esegue il prototipo CPU con 1 o 2 rank sulla stessa macchina.
# Serve a validare la decomposizione di dominio senza GPU (risultati identici
# a meno di arrotondamento fp32).
#   ./scripts/run_cpu.sh 1 -tmax 1
#   ./scripts/run_cpu.sh 2 -tmax 1 -out out/cpu_np2
set -euo pipefail
cd "$(dirname "$0")/.."
NP="${1:?uso: run_cpu.sh <1|2> [opzioni]}"; shift
BIN=${BIN:-bin/mpi_fissa_cpu}
[ -x "$BIN" ] || { echo "manca $BIN: lancia prima ./scripts/build.sh cpu"; exit 2; }
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS
exec mpirun -np "$NP" --bind-to none "$BIN" "$@"
