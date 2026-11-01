#!/bin/sh
# Builds saetta.c as the MPI version with 16 processes.
# The grid is the one set in saetta.c (grid/multigrid.h, uniform
# and fixed). _MPI=16 means the binary is built for 16 processes.
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"

FLAGS="-O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow"

echo "Compilo la versione MPI (16 processi) ..."
qcc -source -grid=multigrid -D_MPI=16 -O3 -march=native saetta.c
mpicc $FLAGS -o saetta _saetta.c -lm
rm -f _saetta.c              # intermediate source, no need to keep it

echo
echo "Build OK.  Avvia con: ./run.sh"
