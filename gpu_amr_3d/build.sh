#!/bin/sh
# Builds the incflo 3D AMR + GPU solver for the "vortex tubes" case.
#
# The incflo sources live in ~/src/incflo (local clone) and contain
# probtype 1001 (init_vortex_tubes, src/prob/prob_init_fluid_usr.cpp)
# and the vorticity refinement (incflo.vorterr, src/incflo_tagging.cpp).
# AMReX and AMReX-Hydro live in ~/src/amrex and ~/src/AMReX-Hydro.
#
# User-space toolchain (no system packages):
#   ~/cuda-13.1   nvcc 13.1 (extracted from the Ubuntu .deb files + NVVM packages)
#   ~/opt/cmake   CMake 3.31
#   see ~/cuda-env.sh
#
# Useful variables: INCFLO_DIR (default ~/src/incflo), JOBS (default 6).
set -e
cd "$(dirname "$0")"
. "$HOME/cuda-env.sh"

INCFLO_DIR="${INCFLO_DIR:-$HOME/src/incflo}"
BUILD_DIR="$INCFLO_DIR/build"

if [ ! -f "$BUILD_DIR/Makefile" ]; then
    echo "Configuro CMake (superbuild AMReX + AMReX-Hydro + incflo)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    "$CMAKE_BIN" .. \
        -DINCFLO_DIM=3 \
        -DINCFLO_MPI=ON \
        -DINCFLO_OMP=OFF \
        -DINCFLO_EB=OFF \
        -DINCFLO_CUDA=ON \
        -DAMReX_GPU_BACKEND=CUDA \
        -DAMReX_CUDA_ARCH=86 \
        -DAMReX_PRECISION=SINGLE \
        -DAMREX_HOME="$HOME/src/amrex" \
        -DAMREX_HYDRO_HOME="$HOME/src/AMReX-Hydro" \
        -DCMAKE_CUDA_COMPILER="$HOME/cuda-13.1/bin/nvcc" \
        -DCUDAToolkit_ROOT="$HOME/cuda-13.1" \
        -DCMAKE_BUILD_TYPE=Release
fi

echo "Compilo (CUDA, sm_86, SINGOLA precisione come i run GPU di Basilisk)..."
"$CMAKE_BIN" --build "$BUILD_DIR" -j"${JOBS:-6}"

echo
echo "Build OK.  Esegui con: ./run.sh   (oppure ./_avvia_x.sh)"
