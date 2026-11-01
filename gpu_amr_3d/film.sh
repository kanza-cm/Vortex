#!/bin/sh
# ============================================================================
# film.sh: 3D AMR simulation (incflo) up to TMAX + automatic videos
# ============================================================================
#
# Runs the case with periodic plotfiles and, as soon as it ends, generates:
#   risultati/video_T<TMAX>_2d.mp4   x-y slice (z = LZ/2) with |omega| + AMR boxes
#   risultati/video_T<TMAX>_3d.mp4   rotating 3D view: AMR boxes + |omega| isosurface
#   risultati/preview_*.png          previews
#
# Usage:
#   ./film.sh                        # MINLEVEL=4 MAXLEVEL=6, TMAX=200
#   ./film.sh 4 8                    # another level pair
#   TMAX=50 PLOT_PER=1.0 ./film.sh 4 6
#   VORTERR="5e-3 0.30" ./film.sh 4 6
#
# Parameters (env): MINLEVEL MAXLEVEL REF_RATIO TMAX (200) PLOT_PER (2.0)
#   VORTERR (0.05) CFL (empty = from the file) RUN_DIR (/tmp/amr_T200)
#   N2D (512) N3D (640) THR3D (0.2) FPS (5) OUT (<dir>/risultati)
# ============================================================================
set -e
cd "$(dirname "$0")"
W=$(pwd)

# levels: positional arguments or environment variables
if [ $# -ge 2 ] && [ -z "${MINLEVEL:-}" ] && [ -z "${MAXLEVEL:-}" ]; then
    MINLEVEL=$1; MAXLEVEL=$2; shift 2
fi
MINLEVEL="${MINLEVEL:-5}"
MAXLEVEL="${MAXLEVEL:-6}"
export MINLEVEL MAXLEVEL
. ./_livelli.sh
[ "$AMR_MAXLEV" -gt 0 ] || { echo "errore: serve MAXLEVEL > MINLEVEL"; exit 1; }

TMAX="${TMAX:-200}"
PLOT_PER="${PLOT_PER:-2.0}"
PLOTVARS="${PLOTVARS:-}"      # e.g. "vort": writes only that variable in the plotfiles
RUN_TAG="M${MINLEVEL}-${MAXLEVEL}_T${TMAX}"
RUN_DIR="${RUN_DIR:-/tmp/amr_${RUN_TAG}}"
OUT="${OUT:-$W/risultati/run_${RUN_TAG}}"
N2D="${N2D:-512}"; N3D="${N3D:-640}"; THR3D="${THR3D:-0.2}"; FPS="${FPS:-5}"
VORTERR="${VORTERR:-0.05}"          # threshold that makes the adaptation visible

echo "=== film.sh ==="
echo "  livelli: MINLEVEL=$MINLEVEL MAXLEVEL=$MAXLEVEL REF_RATIO=$REF_RATIO -> n_cell=\"$AMR_NCELL\" max_level=$AMR_MAXLEV ref_ratio=\"$AMR_REFRATIO\""
echo "  TMAX=$TMAX  plot ogni ~$PLOT_PER  vorterr=$VORTERR  CFL=${CFL:-dal file}"
echo "  run: $RUN_DIR   output: $OUT"

mkdir -p "$RUN_DIR" "$OUT"
cp -f run.sh _livelli.sh saetta3d_inputs "$RUN_DIR"/
sed -i "s|^amr\.plot_per_approx.*|amr.plot_per_approx     =   $PLOT_PER|" "$RUN_DIR/saetta3d_inputs"
if [ -n "$PLOTVARS" ]; then
    sed -i "s|^amr\.plotVariables.*|amr.plotVariables       =   $PLOTVARS|" "$RUN_DIR/saetta3d_inputs"
fi

cd "$RUN_DIR"
if [ -n "${CFL:-}" ]; then export CFL; fi
STOP_TIME="$TMAX" MINLEVEL="$MINLEVEL" MAXLEVEL="$MAXLEVEL" VORTERR="$VORTERR" ./run.sh

PID=$(cat run.pid)
echo "=== run avviata (pid $PID): attendo la fine (T=$TMAX) ==="
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    LAST=$(grep -E "Step [0-9]+:" log-saetta 2>/dev/null | tail -1 || true)
    [ -n "$LAST" ] && echo "  ... $LAST"
done
echo "=== run terminata ==="
if grep -qi "amrex::abort\|MPI_ABORT" log-saetta; then
    echo "ERRORE: run abortita"; tail -25 log-saetta; exit 1
fi
tail -3 log-saetta
NPLT=$(ls -d plt* 2>/dev/null | wc -l)
echo "plotfile: $NPLT"
[ "$NPLT" -ge 2 ] || { echo "errore: troppo pochi plotfile"; exit 1; }

# --- video ---
K=$(awk "BEGIN{printf \"%g\", $FPS*$PLOT_PER}")   # simulated time = K * t_video
FONT=$(fc-match -f '%{file}' sans 2>/dev/null | head -1)
echo "=== rendering video (t_sim = ${K} * t_video) ==="
python3 "$W/strumenti/video_amr_plotfile.py" . "$OUT/_raw_2d.mp4" --z 2.0 --n "$N2D" --fps "$FPS"
python3 "$W/strumenti/video_amr_3d.py" . "$OUT/_raw_3d.mp4" --n "$N3D" --thr "$THR3D" --fps "$FPS"
if [ -n "$FONT" ] && [ -f "$FONT" ]; then
    VF="drawtext=fontfile=$FONT:text='t = %{eif\\:$K*t\\:d}':x=14:y=14:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.45"
else
    VF="null"
fi
ffmpeg -y -loglevel error -i "$OUT/_raw_2d.mp4" -vf "$VF" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/video_2d.mp4"
ffmpeg -y -loglevel error -i "$OUT/_raw_3d.mp4" -vf "$VF" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/video_3d.mp4"
rm -f "$OUT/_raw_2d.mp4" "$OUT/_raw_3d.mp4" "$OUT"/video*_frame_*.png

for spec in "0:00" "$((NPLT/2)):mid" "$((NPLT-1)):end"; do
    idx=${spec%%:*}; tag=${spec##*:}
    ffmpeg -y -loglevel error -i "$OUT/video_2d.mp4" -vf "select=eq(n\,$idx)" -frames:v 1 "$OUT/preview_2d_t${tag}.png"
    ffmpeg -y -loglevel error -i "$OUT/video_3d.mp4" -vf "select=eq(n\,$idx)" -frames:v 1 "$OUT/preview_3d_t${tag}.png"
done

cat > "$OUT/LEGGIMI.txt" <<EOF
film.sh — configurazione usata ($RUN_TAG)
MINLEVEL=$MINLEVEL  MAXLEVEL=$MAXLEVEL  REF_RATIO=$REF_RATIO
n_cell="$AMR_NCELL"  amr.max_level=$AMR_MAXLEV  amr.ref_ratio="$AMR_REFRATIO"
TMAX=$TMAX  plot_per_approx=$PLOT_PER  vorterr=$VORTERR  CFL=${CFL:-dal file}
N2D=$N2D  N3D=$N3D  THR3D=$THR3D  FPS=$FPS
plotfile in: $RUN_DIR
EOF

echo "=== fatto ==="
ls -la "$OUT"
