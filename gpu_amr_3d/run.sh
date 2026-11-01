#!/bin/sh
# Runs the 3D AMR + GPU case (incflo) with MINLEVEL/MAXLEVEL (and REF_RATIO)
# in the same convention as the 2D solver.
#
#   level L -> dx = 2^-L ; n_cell = 2^MINLEVEL 2^MINLEVEL 4*2^MINLEVEL
#   each AMR level refines by REF_RATIO (default 4 = two 2D steps)
#   amr.max_level = ceil((MAXLEVEL-MINLEVEL)/log2(REF_RATIO))
#   e.g.: ./run.sh 4 6 -> base 16 16 64, 1 level "4" -> max 64 64 256
#
# Usage:
#   ./run.sh                        # MINLEVEL=6 MAXLEVEL=8, REF_RATIO=4
#   MINLEVEL=4 MAXLEVEL=6 ./run.sh
#   ./run.sh 4 6
#   REF_RATIO=2 ./run.sh 6 8        # classic ratio 2 (2 levels)
#   ./run.sh 4 7                    # odd difference -> ref_ratio "4 2"
#   STOP_TIME=50 ./run.sh 4 6       # stops at t=50 (default: 200 from the file)
#   MAX_STEP=100 ./run.sh 4 6       # or cap the number of steps
#   VORTERR="0.05 0.30" ./run.sh    # refinement thresholds (|omega|), one per AMR level
#   CFL=0.5 ./run.sh 4 6            # adaptive step CFL (default: 0.5 from the file; max 0.5 with MOL)
#   REGRID_INT=3 ./run.sh 4 6       # every how many steps the AMR grid is rebuilt
#   MAX_CELLS=20e6 ./run.sh 4 6     # cell budget with feedback on the thresholds (0 = off)
#
# Output: run.pid, log-saetta, plt*/ (AMReX plotfile).
set -e
cd "$(dirname "$0")"

. ./_livelli.sh                      # sets AMR_NCELL / AMR_MAXLEV / AMR_REFRATIO
. "$HOME/cuda-env.sh"

INCFLO_DIR="${INCFLO_DIR:-$HOME/src/incflo}"
EXE="$INCFLO_DIR/build/incflo.ex"
[ -x "$EXE" ] || { echo "manca $EXE: esegui prima ./build.sh"; exit 1; }

echo "Livelli 2D: MINLEVEL=$MINLEVEL  MAXLEVEL=$MAXLEVEL   (REF_RATIO=$REF_RATIO)"
if [ "$AMR_MAXLEV" -eq 0 ]; then
    echo "         -> amr.n_cell = \"$AMR_NCELL\"   amr.max_level = 0 (nessun raffinamento)"
else
    echo "         -> amr.n_cell = \"$AMR_NCELL\"   amr.max_level = $AMR_MAXLEV   amr.ref_ratio = \"$AMR_REFRATIO\"   (dx_min = 2^-$MAXLEVEL)"
    if [ "$AMR_MAXLEV" -eq 1 ]; then
        echo "         nota: con un solo livello AMR conta solo la PRIMA soglia di vorterr."
    fi
fi

# Warning: with max_level >= 2 you need max_level vorterr thresholds; if some are missing
# the last one is repeated (risk of refining almost everything -> VRAM).
if [ "$AMR_MAXLEV" -gt 0 ]; then
    if [ -n "${VORTERR:-}" ]; then
        NVORT=$(printf '%s\n' $VORTERR | wc -w)
    else
        NVORT=$(sed -n 's/^[[:space:]]*incflo\.vorterr[[:space:]]*=[[:space:]]*//p' saetta3d_inputs | sed 's/#.*//' | wc -w)
    fi
    if [ "$NVORT" -lt "$AMR_MAXLEV" ]; then
        echo "attenzione: amr.max_level=$AMR_MAXLEV ma ci sono $NVORT soglie vorterr:"
        echo "            l'ultima viene ripetuta (possibile sovra-raffinamento/VRAM)."
        echo "            Usa VORTERR=\"...\" con $AMR_MAXLEV valori."
    fi
fi

set -- amr.n_cell="$AMR_NCELL" amr.max_level="$AMR_MAXLEV"
if [ "$AMR_MAXLEV" -gt 0 ]; then
    set -- "$@" amr.ref_ratio="$AMR_REFRATIO"
fi
if [ -n "${STOP_TIME:-}" ]; then set -- "$@" stop_time="$STOP_TIME"; fi
if [ -n "${MAX_STEP:-}" ];  then set -- "$@" max_step="$MAX_STEP"; fi
if [ -n "${VORTERR:-}" ];   then set -- "$@" incflo.vorterr="$VORTERR"; fi
if [ -n "${REGRID_INT:-}" ]; then set -- "$@" amr.regrid_int="$REGRID_INT"; fi
if [ -n "${MAX_CELLS:-}" ];  then set -- "$@" incflo.max_cells="$MAX_CELLS"; fi
if [ -n "${CFL:-}" ];        then set -- "$@" incflo.cfl="$CFL"; fi

if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: $(basename "$EXE") saetta3d_inputs $*"
    exit 0
fi

echo "Avvio: $(basename "$EXE") saetta3d_inputs amr.n_cell=\"$AMR_NCELL\" amr.max_level=$AMR_MAXLEV amr.ref_ratio=\"$AMR_REFRATIO\""
nohup "$EXE" saetta3d_inputs "$@" > log-saetta 2>&1 &
echo $! > run.pid
echo "pid $(cat run.pid)  (log: log-saetta)"
