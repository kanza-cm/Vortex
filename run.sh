#!/bin/sh
# Runs the simulation with MPI on 16 processes (see build.sh).
# Output files land in this folder:
#   omega.mp4        vorticity movie (zoomed view)
#   omega.ppm        final frame
#   diagnostics.dat  time series for the plots
#   log-saetta       solver output
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"

echo "Avvio: mpirun --oversubscribe -np 16 ./saetta"
echo "(log in log-saetta)"
mpirun --oversubscribe -np 16 ./saetta 2> log-saetta
echo "Fatto. Grafici con: gnuplot plot.gp"
