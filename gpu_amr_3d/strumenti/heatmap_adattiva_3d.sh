#!/bin/sh
# Error heatmap for the ADAPTIVE grid of the 3D AMR solver (gpu_amr_3d):
# for each MAXLEVEL L = LMIN..LMAX and each VORTERR threshold in THR_LIST
# it runs one case up to TMAX and saves the last plotfile. The reference is
# the run at MAXLEVEL=LMAX with the smallest threshold (last in THR_LIST).
# Errors (L_inf and L2 on the common MINLEVEL grid, field = vort) are
# computed with error_norm_amr3d.py.
#
# Usage:  ./strumenti/heatmap_adattiva_3d.sh
#   e.g.: TMAX=10 LMAX=7 OUT=heatmap_T10 ./strumenti/heatmap_adattiva_3d.sh
#   e.g.: SKIP_RUNS=1 DATA_DIR=heatmap_adattiva_3d ./strumenti/heatmap_adattiva_3d.sh
#
# Default: MINLEVEL=4 LMIN=4 LMAX=6 TMAX=5 CFL=0.5 REF_RATIO=4
#          THR_LIST="0.3 0.1 3e-2 1e-2" OUT=heatmap_adattiva_3d
# Output: <OUT>/heatmap.dat + heatmap.png (L_inf),
#         <OUT>/heatmap_l2.dat + heatmap_l2.png (L2)
set -e
cd "$(dirname "$0")/.."

. "$HOME/cuda-env.sh"
EXE="${INCFLO_DIR:-$HOME/src/incflo}/build/incflo.ex"
[ -x "$EXE" ] || { echo "manca $EXE: esegui prima ./build.sh"; exit 1; }

MINLEVEL="${MINLEVEL:-4}"
LMIN="${LMIN:-4}"
LMAX="${LMAX:-6}"
TMAX="${TMAX:-5}"
TILT="${TILT:-2.0}"      # inclinazione a X: 2 = ~8 gradi, 0 = tubi paralleli
CFL="${CFL:-0.5}"
REF_RATIO="${REF_RATIO:-4}"
THR_LIST="${THR_LIST:-0.3 0.1 3e-2 1e-2}"
OUT="${OUT:-heatmap_adattiva_3d}"
DATA_DIR="${DATA_DIR:-$OUT}"
mkdir -p "$OUT" "$DATA_DIR"
OUT=$(cd "$OUT" && pwd)              # absolute paths (gnuplot changes directory)
DATA_DIR=$(cd "$DATA_DIR" && pwd)
SKIP_RUNS="${SKIP_RUNS:-0}"
PLOT_PER="${PLOT_PER:-$(awk "BEGIN{print $TMAX/2}")}"
MAX_CELLS="${MAX_CELLS:-12e6}"

[ "$LMAX" -ge "$LMIN" ] || { echo "serve LMAX >= LMIN"; exit 1; }

K=0; TMP=$REF_RATIO
while [ "$TMP" -gt 1 ]; do TMP=$((TMP / 2)); K=$((K + 1)); done
NX=$((1 << MINLEVEL)); NZ=$((4 * NX)); NCELL="$NX $NX $NZ"

amr_levels() {  # $1 = MAXLEVEL -> echo "max_level ref_ratio"
    D=$(( $1 - MINLEVEL ))
    if [ "$D" -le 0 ]; then echo "0"; return; fi
    NFULL=$((D / K)); REM=$((D % K))
    RATIOS=""; i=0
    while [ "$i" -lt "$NFULL" ]; do
        [ -z "$RATIOS" ] && RATIOS="$REF_RATIO" || RATIOS="$RATIOS $REF_RATIO"
        i=$((i + 1))
    done
    if [ "$REM" -gt 0 ]; then
        R2=$((1 << REM)); [ -z "$RATIOS" ] && RATIOS="$R2" || RATIOS="$RATIOS $R2"
    fi
    echo "$((NFULL + (REM > 0 ? 1 : 0))) $RATIOS"
}

last_plotfile() {  # $1 = run dir -> echo path of plotfile with max time
    python3 - "$1" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strumenti"))
from video_amr_plotfile import read_header
d = sys.argv[1]
pfs = [os.path.join(d, x) for x in os.listdir(d) if x.startswith("plt") and os.path.isdir(os.path.join(d, x))]
if not pfs:
    raise SystemExit("no plotfile in " + d)
print(max(pfs, key=lambda p: read_header(p)["time"]))
PY
}

mk_run() {  # $1 = L  $2 = THR
    L=$1; THR=$2
    run="$OUT/L${L}_thr${THR}"
    mkdir -p "$run"
    cp -f saetta3d_inputs "$run"/
    sed -i -e "s|^stop_time.*|stop_time               =   $TMAX|" \
           -e "s|^amr.plot_per_approx.*|amr.plot_per_approx     =   $PLOT_PER|" \
           -e "s|^amr.plotVariables.*|amr.plotVariables       =   vort|" "$run/saetta3d_inputs"
    set -- $(amr_levels "$L")
    MLEV=$1; RAT=$2
    echo "=== L=$L thr=$THR  max_level=$MLEV ref_ratio=\"$RAT\" TMAX=$TMAX ==="
    ( cd "$run" && "$EXE" saetta3d_inputs amr.n_cell="$NCELL" amr.max_level="$MLEV" \
          ${RAT:+amr.ref_ratio="$RAT"} stop_time="$TMAX" incflo.vorterr="$THR" saetta.TILT="$TILT" \
          incflo.cfl="$CFL" incflo.max_cells="$MAX_CELLS" > log-saetta 2>&1 )
    if grep -qi "amrex::abort\|MPI_ABORT" "$run/log-saetta"; then
        echo "errore: run abortita ($run)"; tail -20 "$run/log-saetta"; exit 1
    fi
}

if [ "$SKIP_RUNS" = "0" ]; then
    L=$LMIN
    while [ "$L" -le "$LMAX" ]; do
        for THR in $THR_LIST; do mk_run "$L" "$THR"; done
        L=$((L + 1))
    done
fi

ref_thr=$(echo "$THR_LIST" | awk '{print $NF}')
ref_dir="$DATA_DIR/L${LMAX}_thr${ref_thr}"
ref_pf=$(last_plotfile "$ref_dir") || { echo "manca il riferimento $ref_dir"; exit 1; }

mkdir -p "$OUT"
echo "# x_idx L log10_err celle ref" > "$OUT/heatmap.dat"
echo "# x_idx L log10_err celle ref" > "$OUT/heatmap_l2.dat"

L=$LMIN
while [ "$L" -le "$LMAX" ]; do
    jx=1
    for THR in $THR_LIST; do
        dir="$DATA_DIR/L${L}_thr${THR}"
        pf=$(last_plotfile "$dir")
        if [ "$L" = "$LMAX" ] && [ "$THR" = "$ref_thr" ]; then
            cells=$(python3 strumenti/error_norm_amr3d.py "$pf" "$ref_pf" --minlevel "$MINLEVEL" | awk '{print $3}')
            echo "$jx $L NaN $cells 1" >> "$OUT/heatmap.dat"
            echo "$jx $L NaN $cells 1" >> "$OUT/heatmap_l2.dat"
            echo "   L=$L thr=$THR -> RIFERIMENTO (celle=$cells)"
        else
            read ei e2 cells _cref <<EOF
$(python3 strumenti/error_norm_amr3d.py "$pf" "$ref_pf" --minlevel "$MINLEVEL")
EOF
            li=$(awk -v e="$ei" 'BEGIN{printf "%.2f", (e>0)?log(e)/log(10):-99}')
            l2=$(awk -v e="$e2" 'BEGIN{printf "%.2f", (e>0)?log(e)/log(10):-99}')
            echo "$jx $L $li $cells 0" >> "$OUT/heatmap.dat"
            echo "$jx $L $l2 $cells 0" >> "$OUT/heatmap_l2.dat"
            echo "   L=$L thr=$THR -> log10(L_inf)=$li  log10(L2)=$l2  celle=$cells"
        fi
        jx=$((jx + 1))
    done
    L=$((L + 1))
done

# --- grafici (stile della heatmap 2D) ---
tics=""; j=1
for THR in $THR_LIST; do
    lab=$(awk "BEGIN{printf \"%.0e\", 1/$THR}")
    tics="$tics \"$lab\" $j,"
    j=$((j + 1))
done
nx=$((j - 1))

make_plot() {
    dat="$1"; png="$2"; gp="$3"; cb="$4"
    rmin=$(awk '$5==0 && (mn=="" || $3<mn){mn=$3} END{print mn}' "$OUT/$dat")
    rmax=$(awk '$5==0 && (mx=="" || $3>mx){mx=$3} END{print mx}' "$OUT/$dat")
    cat > "$OUT/$gp" <<EOF
set terminal pngcairo size 1100,850 enhanced font ",11"
set output "$OUT/$png"
set title "log10 (|vort| error) - t = $TMAX" font ",13"
set xlabel "1 / VORTERR"
set ylabel "MAXLEVEL"
set xtics ($tics)
set ytics 1
set xrange [0.5:$(awk "BEGIN{print $nx+0.5}")]
set yrange [$(awk "BEGIN{print $LMIN-0.5}"):$(awk "BEGIN{print $LMAX+0.5}")]
set palette defined (0 "#440154", 0.25 "#3b528b", 0.5 "#21918c", 0.75 "#5ec962", 1 "#fde725")
set cbrange [$(awk "BEGIN{print $rmin-0.1}"):$(awk "BEGIN{print $rmax+0.1}")]
set colorbox
set cblabel "$cb" font ",11"
unset key
plot "$OUT/$dat" using 1:2:3 with image, \\
     "$OUT/$dat" using 1:2:((\$5==1) ? \\
              sprintf("%s\\n(%d)", "ref", \$4) : \\
              sprintf("%.2f\\n(%d)", \$3, \$4)) \\
          with labels font ",9"
EOF
    ( cd "$OUT" && gnuplot "$gp" )
    echo "   $OUT/$png"
}

make_plot heatmap.dat heatmap.png heatmap.gp "log10 (||vort - vort_ref||_inf)"
make_plot heatmap_l2.dat heatmap_l2.png heatmap_l2.gp "log10 (||vort - vort_ref||_2)"

echo
echo "Fatto: $OUT/heatmap.png (L_inf) e $OUT/heatmap_l2.png (L2)"
