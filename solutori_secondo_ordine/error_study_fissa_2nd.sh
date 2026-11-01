#!/bin/sh
# Error study (L2 and L_inf) on a fixed grid for the merging of two
# vortices, with reference MAXLEVEL=LMAX.
#
# Usage:  ./error_study_fissa_2nd.sh
#   e.g.: NP=16 TMAX=100 LMAX=10 ./error_study_fissa_2nd.sh
#   e.g.: NIT=12 TMAX=200 OUT=error_study_fissa_2nd_nit12_t200 \
#            ./error_study_fissa_2nd.sh
#
# Default: LMIN=6, LMAX=10 (reference), NP=16, TMAX=100, CFL=0.8, NIT=6,
#          OUT=error_study_fissa_2nd.
# Output: <OUT>/run<L>/ and <OUT>/error.dat
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"

LMIN="${LMIN:-6}"
LMAX="${LMAX:-10}"
TMAX="${TMAX:-100}"
NP="${NP:-16}"
CFL="${CFL:-0.8}"
NIT="${NIT:-6}"
OUT="${OUT:-error_study_fissa_2nd}"

out="$OUT"
mkdir -p "$out"

extra=""
[ -n "$NIT" ] && extra="-DPOISSON_ITER=$NIT"

L=$LMIN
while [ "$L" -le "$LMAX" ]; do
    run="$out/run$L"
    mkdir -p "$run"
    echo "=== fissa L=$L  np=$NP  TMAX=$TMAX  CFL=$CFL  NIT=$NIT ==="
    qcc -source -grid=multigrid -D_MPI=$NP -O3 -march=native \
        -DCFL_USER=$CFL -DMOVIE=0 -DMAXLEVEL=$L -DTMAX=$TMAX \
        -DMATRIX_N=$((1 << L)) $extra saetta.c
    mpicc -O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow \
        -o "$run/saetta" _saetta.c -lm
    rm -f _saetta.c
    ( cd "$run" && mpirun --oversubscribe -np $NP ./saetta > log.txt 2>&1 )
    L=$((L + 1))
done

ref="$out/run$LMAX/omega_matrix.bin"
[ -f "$ref" ] || { echo "manca il riferimento $ref"; exit 1; }

echo "# L err_inf err_L2 celle" > "$out/error.dat"
L=$LMIN
while [ "$L" -le "$LMAX" ]; do
    dir="$out/run$L"
    cells=$(awk '$1 ~ /^[0-9]/ {print $18}' "$dir/diagnostics.dat" | tail -1)
    if [ "$L" = "$LMAX" ]; then
        echo "$L NaN NaN $cells" >> "$out/error.dat"
        echo "   L=$L -> RIFERIMENTO (celle=$cells)"
    else
        read -r ei e2 <<EOF
$(python3 ../error_norm.py "$dir/omega_matrix.bin" "$ref")
EOF
        echo "$L $ei $e2 $cells" >> "$out/error.dat"
        echo "   L=$L -> err_inf=$ei  err_L2=$e2  celle=$cells"
    fi
    L=$((L + 1))
done

echo
echo "Ordini (p = log2(e_L / e_{L+1})):"
awk 'NR > 1 && $2 != "NaN" {
    n++; L[n] = $1; ei[n] = $2; e2[n] = $3
}
END {
    for (i = 1; i < n; i++)
        printf "  L%d->L%d: p_inf=%.2f  p_L2=%.2f\n", L[i], L[i+1], \
            log(ei[i]/ei[i+1])/log(2), log(e2[i]/e2[i+1])/log(2)
    if (n > 1)
        printf "  globale L%d->L%d: p_inf=%.2f  p_L2=%.2f\n", L[1], L[n], \
            log(ei[1]/ei[n])/((n-1)*log(2)), \
            log(e2[1]/e2[n])/((n-1)*log(2))
}' "$out/error.dat"

echo
echo "Fatto: $out/error.dat"
