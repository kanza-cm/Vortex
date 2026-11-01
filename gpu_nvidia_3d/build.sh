#!/bin/sh
# Builds saetta3d.c (3D vortex coalescing, incompressible
# velocity-pressure Navier-Stokes) for NVIDIA GPUs with the Basilisk CUDA
# backend, in SINGLE PRECISION.
#
# Prerequisites (once, see ../gpu_nvidia/build.sh for the details):
#   sudo apt install cuda-driver-dev-13-1 cuda-nvrtc-dev-13-1 \
#                    cuda-nvrtc-13-1 cuda-cudart-dev-13-1
#   make -C $BASILISK/grid/gpu liberrors.a
#   CPATH=/usr/local/cuda-13.1/targets/x86_64-linux/include \
#       make -C $BASILISK/grid/cuda libbuda.a
#
# Grid: cuda/multigrid3D, cubic cells with side 2^-MAXLEVEL,
# box 1 x 1 x LZ with periodic z.  EXTRA=... passes extra flags to qcc,
# e.g. EXTRA="-DMAXLEVEL=6 -DTMAX=1" for a quick test.
set -e
cd "$(dirname "$0")"
HERE=$(pwd)

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"
CUDA="${CUDA:-/usr/local/cuda-13.1/targets/x86_64-linux}"

# Basilisk GPU libraries (rebuilds them only when missing).
[ -f "$BASILISK/grid/gpu/liberrors.a" ] || make -C "$BASILISK/grid/gpu" liberrors.a
[ -f "$BASILISK/grid/cuda/libbuda.a" ] || \
    CPATH="$CUDA/include" make -C "$BASILISK/grid/cuda" libbuda.a

SRC="$HERE/saetta3d.c"
EXTRA="${EXTRA:-}"

# qcc does not handle spaces in paths: it compiles from a copy of the
# sources in a temporary directory without spaces, then moves the
# binary here.
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT INT TERM
cp "$SRC" "$BUILD/"

echo "Compilo per GPU (CUDA 3D, singola precisione): $SRC ..."
(
    cd "$BUILD"
    # -disable-dimensions: the static dimension checker in qcc cannot
    # cope with the hand-written formulas of the initial conditions (it
    # does not affect the generated code, only the checks).
    qcc -autolink -disable-dimensions -O3 -grid=cuda/multigrid3D \
        -I"$CUDA/include" -L"$CUDA/lib" -L"$CUDA/lib/stubs" \
        $EXTRA saetta3d.c -o saetta3d_gpu -lm -lz
)
mv "$BUILD/saetta3d_gpu" "$HERE/saetta3d_gpu"

echo
echo "Build OK.  Avvia con: ./run.sh"
