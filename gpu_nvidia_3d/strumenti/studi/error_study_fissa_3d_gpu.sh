#!/bin/sh
# Error study (grid convergence) at FIXED GRID with the 3D GPU solver
# (saetta3d.c, CUDA backend, SINGLE precision).
#
# For L = LMIN..LMAX it runs a uniform run at MAXLEVEL=L and compares the
# WZ field with the reference run at LREF = LMAX:
#
#     ||err||_inf = max |wz_L - wz_LREF|
#
# (the finer reference is restricted in 2x2x2 blocks to level L).
# The last row of error.dat is L = LMAX-1 (LMAX is the reference run).
# Here dt ~ h (fixed CFL), so space and time both contribute at second
# order: the measured order estimates the scheme directly.
#
# Grid: the base divisions NDIV0 (x,y) double at every MAXLEVEL -> the
# run at level L has NDIV0*2^L x NDIV0*2^L x NDIV0*2^L*4 cells (cubic
# cells, domain 1x1x4). Default NDIV0=1.
# E.g. NDIV0=8 LMIN=3 -> starting grid 64x64x256, as L=6 with NDIV0=1.
#
# GPU note (8 GB): LMAX=8 -> 256x256x1024 = 67M cells (~4.6 GB of VRAM);
# LMAX=9 does not fit in memory. The runs are sequential (single GPU).
# T is the final physical time of each run: longer = more physics but more
# steps (dt ~ h); t=10 is a good compromise, t=50 = 3D movies.
#
# Output in risultati/error_study/error_study_fissa_3d_gpu/:
#   run<L>/   (log.txt, diagnostics.dat, omega3d_matrix.bin, resa_max.dat)
#   error.dat L  ||err||_inf  ||err||_2  cells
#   error.png log10 of the errors vs MAXLEVEL
set -e
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"
# libnvrtc is needed to JIT the kernels; libcuda.so.1 comes from WSL.
CUDA="${CUDA:-/usr/local/cuda-13.1/targets/x86_64-linux}"
export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LMIN="${LMIN:-6}"      # first level
LMAX="${LMAX:-8}"      # last level = reference run (the finest one)
T="${T:-10}"           # final time of each run
CFL="${CFL:-0.3}"      # CFL (as in the production runs)
NDIV0="${NDIV0:-1}"    # base divisions in x,y at MAXLEVEL=0 (z = 4x)
TILT="${TILT:-2}"      # inclinazione a X (2 = ~8 gradi, 0 = tubi paralleli)
POISSON_ITER="${POISSON_ITER:-}"   # se impostato: V-cycle fissi del solver (es. 5)

LREF=$LMAX

out="${OUT_DIR:-risultati/error_study/error_study_fissa_3d_gpu}"
rm -rf "$out"
mkdir -p "$out"

# -------- runs from LMIN to LREF --------
L=$LMIN
while [ "$L" -le "$LREF" ]; do
  run="$out/run$L"
  mkdir -p "$run"

  echo "=== MAXLEVEL=$L   (griglia fissa GPU, single, CFL=$CFL, T=$T)"
  echo "    divisioni: $((NDIV0 << L)) x $((NDIV0 << L)) x $((NDIV0 << L)) * 4 ==="

  # qcc does not handle spaces in paths: it compiles from a copy of the
  # sources in a temporary directory without spaces, then moves the
  # binary into the run folder (same logic as build.sh).
  BUILD=$(mktemp -d)
  cp "$HERE/saetta3d.c" "$BUILD/"
  cp "$HERE/../solutori_secondo_ordine/centered_cn.h" "$BUILD/" 2>/dev/null || {
      echo "errore: centered_cn.h non trovato in ../solutori_secondo_ordine/" >&2; exit 1; }
  (
    cd "$BUILD"
    qcc -autolink -disable-dimensions -O3 -grid=cuda/multigrid3D \
        -I"$CUDA/include" -L"$CUDA/lib" -L"$CUDA/lib/stubs" \
        -DCFL_USER=$CFL \
        -DMOVIE=0 -DVTK=0 -DMATRIX=1 \
        -DNDIV0=$NDIV0 -DMAXLEVEL=$L -DTMAX=$T -DTILT=$TILT \
        ${POISSON_ITER:+-DPOISSON_ITER=$POISSON_ITER} \
        saetta3d.c -o saetta3d_gpu -lm -lz
  )
  mv "$BUILD/saetta3d_gpu" "$run/saetta3d_gpu"
  rm -rf "$BUILD"

  ( cd "$run" && ./saetta3d_gpu > log.txt 2>&1 )

  L=$((L + 1))
done

# -------- errors against the LREF reference --------
ref="$out/run$LREF/omega3d_matrix.bin"
echo "# L err_inf err_L2 cells" > "$out/error.dat"
L=$LMIN
while [ "$L" -lt "$LREF" ]; do
  vals=$(python3 "$HERE/strumenti/error_norm_3d.py" "$out/run$L/omega3d_matrix.bin" "$ref")
  cells=$(awk '$1 ~ /^[0-9]/ {print $12}' "$out/run$L/diagnostics.dat" | tail -1)
  echo "$L $vals $cells" >> "$out/error.dat"
  echo "   L=$L   ||err|| = $vals   celle = $cells"
  L=$((L + 1))
done

# -------- plot --------
cat > "$out/error.gp" <<EOF
set terminal pngcairo size 1000,700 enhanced font ",12"
set output "error.png"
set grid
set xlabel "MAXLEVEL"
set ylabel "log10 ||wz_L - wz_ref||_{inf}   (fissa 3D GPU, CFL $CFL, ref = L$LREF)"
set key top left
plot "error.dat" u 1:(log10(\$2)) w lp lw 2 pt 7 t "||.||_{inf}", \\
     "error.dat" u 1:(log10(\$3)) w lp lw 2 pt 5 t "||.||_2"
EOF
( cd "$out" && gnuplot error.gp )

# -------- summary with estimated order between consecutive levels --------
echo
echo "MAXLEVEL   ||err||_inf    ||err||_2     celle     ordine"
awk 'NR>1 {
       printf "   %2d     %-13.3e %-13.3e %-9d", $1, $2, $3, $4;
       if (prev > 0) printf "   %.2f", log(prev/$2)/log(2);
       printf "\n"; prev=$2
     }' "$out/error.dat"
echo
echo "Fatto: $out/error.dat / $out/error.png"
