#!/bin/bash
# build.sh -- compila il prototipo mpi_fissa: versione CPU (per validare la
# decomposizione senza GPU) e versione CUDA (per il run di produzione).
#
# Uso:
#   ./scripts/build.sh              # compila CPU + GPU con NX=128
#   NX=64 ./scripts/build.sh cpu    # solo CPU, griglia 64x64x256
#   NX=256 ./scripts/build.sh gpu   # solo GPU, griglia 256x256x1024 (L8)
#   ARCH=86 ./scripts/build.sh gpu  # architettura CUDA (RTX 3070 = sm_86)
#   EXTRA="-DPOISSON_NIT=2" ...     # (non serve: si passa a runtime)
#
# Variabili d'ambiente:
#   NX        griglia: nx=ny=NX, nz=4*NX, Delta=1/NX   (default 128)
#   CUDA      radice del toolkit CUDA (default ~/cuda-13.1)
#   ARCH      compute capability senza "sm_"          (default 86)
#   MPICXX    wrapper MPI per C++                     (default mpicxx)
#   OUT       cartella dei binari                     (default bin)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

NX="${NX:-128}"
CUDA="${CUDA:-$HOME/cuda-13.1}"
ARCH="${ARCH:-86}"
MPICXX="${MPICXX:-mpicxx}"
OUT="${OUT:-bin}"
SRC=src/mpi_fissa.cu
NVCC="${NVCC:-$CUDA/bin/nvcc}"
[ -x "$NVCC" ] || NVCC=$(command -v nvcc || true)

what="${1:-all}"
mkdir -p "$OUT"

if [ ! -f "$SRC" ]; then echo "ERRORE: $SRC non trovato (lancia da qualunque posto)"; exit 2; fi

build_cpu() {
    echo "== CPU: $MPICXX -O3 -fopenmp -DMF_NX=$NX =="
    $MPICXX -O3 -std=c++17 -fopenmp -x c++ -DMF_USE_CUDA=0 -DMF_NX="$NX" \
        "$SRC" -o "$OUT/mpi_fissa_cpu"
    echo "   -> $OUT/mpi_fissa_cpu"
}

build_gpu() {
    if [ -z "${NVCC:-}" ] || [ ! -x "$NVCC" ]; then
        echo "ERRORE: nvcc non trovato; imposta CUDA=... oppure NVCC=..." >&2
        exit 2
    fi
    echo "== GPU: $NVCC -arch=sm_$ARCH -DMF_NX=$NX (host compiler: $MPICXX) =="
    # -ccbin mpicxx: nvcc compila il codice host con il wrapper MPI, cosi'
    # mpi.h e' visibile e il link include le librerie MPI.
    if "$NVCC" -O3 -std=c++17 --extended-lambda -arch="sm_$ARCH" \
            -ccbin "$MPICXX" -DMF_USE_CUDA=1 -DMF_NX="$NX" \
            "$SRC" -o "$OUT/mpi_fissa_gpu" -lcudart; then
        echo "   -> $OUT/mpi_fissa_gpu (-ccbin $MPICXX)"
    else
        echo "   -ccbin non ha funzionato: provo la compilazione in due passi..." >&2
        "$NVCC" -O3 -std=c++17 --extended-lambda -arch="sm_$ARCH" -c \
            -DMF_USE_CUDA=1 -DMF_NX="$NX" "$SRC" -o "$OUT/mpi_fissa_gpu.o"
        "$MPICXX" -O3 "$OUT/mpi_fissa_gpu.o" -o "$OUT/mpi_fissa_gpu" \
            -lcudart -L"$CUDA/lib64" -L"$CUDA/lib"
        echo "   -> $OUT/mpi_fissa_gpu (due passi)"
    fi
}

case "$what" in
    cpu) build_cpu ;;
    gpu) build_gpu ;;
    all) build_cpu; build_gpu ;;
    *) echo "uso: $0 [cpu|gpu|all]"; exit 1 ;;
esac

echo
echo "Build OK (NX=$NX -> griglia $NX x $NX x $((4*NX)))."
echo "Test veloce CPU:  mpirun -np 1 $OUT/mpi_fissa_cpu -tmax 0.1 -out out/smoke_cpu1"
echo "Test MPI locale:  OMP_NUM_THREADS=1 mpirun -np 2 $OUT/mpi_fissa_cpu -tmax 0.1 -out out/smoke_cpu2"
