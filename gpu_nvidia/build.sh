#!/bin/sh
# Builds solver saetta.c (2nd order, AB2) for an NVIDIA GPU using the
# Basilisk CUDA backend, in SINGLE PRECISION.
#
# Prerequisites (one time):
#   sudo apt install cuda-driver-dev-13-1 cuda-nvrtc-dev-13-1 \
#                    cuda-nvrtc-13-1 cuda-cudart-dev-13-1
#   make -C $BASILISK/grid/gpu liberrors.a
#   CPATH=/usr/local/cuda-13.1/targets/x86_64-linux/include \
#       make -C $BASILISK/grid/cuda libbuda.a
#
# The cuda/multigrid grid uses the same equations and the same scheme as
# the CPU run (see ../build.sh): only the backend changes, and it builds
# the kernels at run time with NVRTC. Precision is single by default
# (SINGLE_PRECISION in the CUDA backend; no -DDOUBLE_PRECISION), and MPI
# is not used: it runs on a single GPU.
set -e
cd "$(dirname "$0")"
HERE=$(pwd)

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"
CUDA="${CUDA:-/usr/local/cuda-13.1/targets/x86_64-linux}"

# Basilisk GPU libraries (rebuilt only when missing).
[ -f "$BASILISK/grid/gpu/liberrors.a" ] || make -C "$BASILISK/grid/gpu" liberrors.a
[ -f "$BASILISK/grid/cuda/libbuda.a" ] || \
    CPATH="$CUDA/include" make -C "$BASILISK/grid/cuda" libbuda.a

# Source shared with the CPU build. With SRC=... you can build
# another variant (e.g. SRC=../solutori_secondo_ordine/saetta_adattiva.c).
# EXTRA=... passes extra flags to qcc (e.g. EXTRA="-DMAXLEVEL=13").
SRC="${SRC:-$HERE/../saetta.c}"
SRC=$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")
TOPDIR=$(cd "$HERE/.." && pwd)
EXTRA="${EXTRA:-}"

# qcc does not handle spaces in paths: it builds from a copy of the
# sources in a temp directory without spaces, then moves the
# binary here.
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT INT TERM
cp "$SRC" "$BUILD/"
cp -r "$TOPDIR/solutori_secondo_ordine" "$BUILD/" 2>/dev/null || true

echo "Compilo per GPU (CUDA, singola precisione): $SRC ..."
(
    cd "$BUILD"
    qcc -autolink -O3 -grid=cuda/multigrid \
        -I"$CUDA/include" -L"$CUDA/lib" -L"$CUDA/lib/stubs" \
        $EXTRA "$(basename "$SRC")" -o saetta_gpu -lm
)
mv "$BUILD/saetta_gpu" "$HERE/saetta_gpu"

echo
echo "Build OK.  Avvia con: ./run.sh"
