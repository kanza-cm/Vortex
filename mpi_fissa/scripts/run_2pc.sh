#!/bin/bash
# run_2pc.sh -- una SINGOLA simulazione su 2 PC (un rank per PC) via SSH+MPI.
#
# Prerequisiti (una volta):
#   1. ssh master -> slave1 senza password (vedi cluster/README.md);
#   2. lo stesso progetto in un percorso SENZA SPAZI su entrambi i PC:
#        master: scp/rsync di questa cartella in ~/mpi_fissa
#        slave1: idem, NELLO STESSO PERCORSO ($HOME/mpi_fissa);
#   3. binario GPU compilato su ogni PC (./scripts/build.sh gpu), stessa NX;
#   4. rete: WSL mirrored networking (consigliato) oppure port-proxy.
#
# Uso:
#   ./scripts/run_2pc.sh -tmax 1 -out out/2pc_test
#   HOSTFILE=scripts/hostfile_2pc.txt ./scripts/run_2pc.sh -tmax 10 -out out/2pc_T10
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
HOSTFILE="${HOSTFILE:-$HERE/scripts/hostfile_2pc.txt}"
BIN="${BIN:-bin/mpi_fissa_gpu}"
CUDA="${CUDA:-$HOME/cuda-13.1}"
[ -f "$HOSTFILE" ] || { echo "manca $HOSTFILE"; exit 2; }
# il percorso deve essere identico sull'altro PC e senza spazi
case "$HERE" in *" "*) echo "ERRORE: percorso con spazi ($HERE): copia il progetto in ~/mpi_fissa"; exit 2;; esac
cat "$HOSTFILE"
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec mpirun --hostfile "$HOSTFILE" -np 2 \
    --map-by ppr:1:node --bind-to none --wdir "$HERE" \
    -x LD_LIBRARY_PATH -x OMP_NUM_THREADS \
    "$BIN" "$@"
