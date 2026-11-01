#!/bin/sh
# Error heatmap for the ADAPTIVE grid (saetta_adattiva.c):
# for each MAXLEVEL L = LMIN..LMAX and for each EPS in EPS_LIST it runs
# one case and saves omega_matrix.bin (or MATRIX_NAME). The reference is
# the run with MAXLEVEL=LMAX and the smallest EPS. The errors (L_inf and
# L2) are computed with error_norm.py.
#
# Usage:  ./heatmap_adattiva.sh
#   e.g.: NP=16 TMAX=100 LMAX=10 ./heatmap_adattiva.sh
#   e.g.: TMAX=20 ./heatmap_adattiva.sh        (duration of the original heatmap)
#   e.g.: NIT=12 TMAX=20 OUT=heatmap_nit12 ./heatmap_adattiva.sh
#   e.g.: QCC_EXTRA=-DUX_MATRIX MATRIX_NAME=ux_matrix.bin FIELD_LABEL=u_x \
#            NIT=3 TMAX=15 LMIN=4 OUT=heatmap_ux ./heatmap_adattiva.sh
#   e.g.: SKIP_RUNS=1 DATA_DIR=heatmap_ux MATRIX_NAME=omega_matrix.bin \
#            FIELD_LABEL=omega OUT=heatmap_omega ./heatmap_adattiva.sh
#
# Default: LMIN=6, LMAX=10, TMAX=20, NP=16, CFL=0.8, NIT=6,
#          EPS_LIST="1e-3 5e-4 2.5e-4 1.25e-4", OUT=heatmap_adattiva,
#          MATRIX_NAME=omega_matrix.bin, FIELD_LABEL=omega, QCC_EXTRA="",
#          DATA_DIR=$OUT, SKIP_RUNS=0.
# Output: <OUT>/heatmap.dat and heatmap.png (L_inf), heatmap_l2.dat and
#         heatmap_l2.png (L2), plus L<L>_eps<EPS>/.
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"

LMIN="${LMIN:-6}"
LMAX="${LMAX:-10}"
TMAX="${TMAX:-${HEAT_T:-20}}"
NP="${NP:-16}"
CFL="${CFL:-0.8}"
NIT="${NIT:-6}"
EPS_LIST="${EPS_LIST:-1e-3 5e-4 2.5e-4 1.25e-4}"
OUT="${OUT:-heatmap_adattiva}"
MATRIX_NAME="${MATRIX_NAME:-omega_matrix.bin}"
FIELD_LABEL="${FIELD_LABEL:-omega}"
QCC_EXTRA="${QCC_EXTRA:-}"
DATA_DIR="${DATA_DIR:-$OUT}"   # existing runs (with SKIP_RUNS=1)
SKIP_RUNS="${SKIP_RUNS:-0}"    # 1 = do not rerun, use DATA_DIR

hm="$OUT"
mkdir -p "$hm"

extra=""
[ -n "$NIT" ] && extra="-DPOISSON_ITER=$NIT"
[ -n "$QCC_EXTRA" ] && extra="$extra $QCC_EXTRA"

if [ "$SKIP_RUNS" = "0" ]; then
    for EPS in $EPS_LIST; do
        L=$LMIN
        while [ "$L" -le "$LMAX" ]; do
            run="$hm/L${L}_eps${EPS}"
            mkdir -p "$run"
            if [ "$L" -le 8 ]; then NPL=4; else NPL=$NP; fi
            echo "=== L=$L  EPS=$EPS  np=$NPL  TMAX=$TMAX ==="
            qcc -source -grid=quadtree -D_MPI=$NPL -O3 -march=native \
                -DCFL_USER=$CFL -DMOVIE=0 -DMINLEVEL=$LMIN -DMAXLEVEL=$L \
                -DEPS=$EPS -DTMAX=$TMAX -DMATRIX_N=$((1 << L)) $extra \
                saetta_adattiva.c
            mpicc -O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow \
                -o "$run/saetta" _saetta_adattiva.c -lm
            rm -f _saetta_adattiva.c
            ( cd "$run" && mpirun --oversubscribe -np $NPL ./saetta > log.txt 2>&1 )
            L=$((L + 1))
        done
    done
fi

ref_eps=$(echo "$EPS_LIST" | awk '{print $NF}')
ref="$DATA_DIR/L${LMAX}_eps${ref_eps}/$MATRIX_NAME"
[ -f "$ref" ] || { echo "manca il riferimento $ref"; exit 1; }

echo "# x_idx L log10_err celle ref" > "$hm/heatmap.dat"
echo "# x_idx L log10_err celle ref" > "$hm/heatmap_l2.dat"
L=$LMIN
while [ "$L" -le "$LMAX" ]; do
    jx=1
    for EPS in $EPS_LIST; do
        dir="$DATA_DIR/L${L}_eps${EPS}"
        cells=$(awk '$1 ~ /^[0-9]/ {print $18}' "$dir/diagnostics.dat" | tail -1)
        if [ "$L" = "$LMAX" ] && [ "$EPS" = "$ref_eps" ]; then
            echo "$jx $L NaN $cells 1" >> "$hm/heatmap.dat"
            echo "$jx $L NaN $cells 1" >> "$hm/heatmap_l2.dat"
            echo "   L=$L EPS=$EPS -> RIFERIMENTO (celle=$cells)"
        else
            errs=$(python3 ../error_norm.py "$dir/$MATRIX_NAME" "$ref")
            ei=$(echo "$errs" | awk '{print $1}')
            e2=$(echo "$errs" | awk '{print $2}')
            li=$(awk -v e="$ei" 'BEGIN{printf "%.2f", (e>0)?log(e)/log(10):-99}')
            l2=$(awk -v e="$e2" 'BEGIN{printf "%.2f", (e>0)?log(e)/log(10):-99}')
            echo "$jx $L $li $cells 0" >> "$hm/heatmap.dat"
            echo "$jx $L $l2 $cells 0" >> "$hm/heatmap_l2.dat"
            echo "   L=$L EPS=$EPS -> log10(L_inf)=$li  log10(L2)=$l2  celle=$cells"
        fi
        jx=$((jx + 1))
    done
    L=$((L + 1))
done

tics=""
j=1
for EPS in $EPS_LIST; do
    lab=$(awk "BEGIN{printf \"%.0e\", 1/$EPS}")
    tics="$tics \"$lab\" $j,"
    j=$((j + 1))
done
nx=$((j - 1))

make_plot() {
    dat="$1"; png="$2"; normlabel="$3"; cblabel="$4"; gp="$5"
    rmin=$(awk '$5==0 && (mn=="" || $3<mn){mn=$3} END{print mn}' "$hm/$dat")
    rmax=$(awk '$5==0 && (mx=="" || $3>mx){mx=$3} END{print mx}' "$hm/$dat")
    cat > "$hm/$gp" <<EOF
set terminal pngcairo size 1100,850 enhanced font ",11"
set output "$png"
set title "log10 ($normlabel error) - t = $TMAX" font ",13"
set xlabel "1/EPS"
set ylabel "MAXLEVEL"
set xtics ($tics)
set ytics 1
set xrange [0.5:$(awk "BEGIN{print $nx+0.5}")]
set yrange [$(awk "BEGIN{print $LMIN-0.5}"):$(awk "BEGIN{print $LMAX+0.5}")]
set palette defined (0 "#440154", 0.25 "#3b528b", 0.5 "#21918c", 0.75 "#5ec962", 1 "#fde725")
set cbrange [$(awk "BEGIN{print $rmin-0.1}"):$(awk "BEGIN{print $rmax+0.1}")]
set colorbox
set cblabel "$cblabel" font ",11"
unset key
plot "$dat" using 1:2:3 with image, \\
     "$dat" using 1:2:((\$5==1) ? \\
              sprintf("Reference solution\\n(%d)", \$4) : \\
              sprintf("%.2f\\n(%d)", \$3, \$4)) \\
          with labels font ",9"
EOF
    ( cd "$hm" && gnuplot "$gp" )
    echo "   $hm/$png"
}

make_plot heatmap.dat heatmap.png "L_{inf}" \
    "log10 (||$FIELD_LABEL - ${FIELD_LABEL}_{ref}||_{inf})" heatmap.gp
make_plot heatmap_l2.dat heatmap_l2.png "L_2" \
    "log10 (||$FIELD_LABEL - ${FIELD_LABEL}_{ref}||_2)" heatmap_l2.gp

echo
echo "Fatto: $hm/heatmap.png (L_inf) e $hm/heatmap_l2.png (L2)"
