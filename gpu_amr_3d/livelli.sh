#!/bin/sh
# ============================================================================
# MINLEVEL / MAXLEVEL / REF_RATIO "Basilisk 2D style" (3D AMR GPU solver).
# ============================================================================
#
# 2D convention (saetta_adattiva.c): level L -> dx = 2^-L.
# The domain is 1 x 1 x LZ with LZ = 4, so at level L there are 2^L
# divisions in x and y and 4*2^L in z (ZDIV = 4):
#
#     n_cell = 2^MINLEVEL  2^MINLEVEL  4*2^MINLEVEL
#
# One AMR level refines by REF_RATIO (default 4 = "two 2D levels" in one
# shot), so:
#
#     amr.max_level = ceil( (MAXLEVEL - MINLEVEL) / log2(REF_RATIO) )
#     amr.ref_ratio = list of ratios per level, e.g. "4" or "4 2"
#                     (the last is 2 when the difference is odd)
#
# Examples (REF_RATIO=4, default):
#   MINLEVEL=4 MAXLEVEL=6 -> base 16 16 64,   1 level "4"     -> max 64 64 256    (dx_min 2^-6)
#   MINLEVEL=6 MAXLEVEL=8 -> base 64 64 256,  1 level "4"     -> max 256 256 1024 (dx_min 2^-8)  [default]
#   MINLEVEL=4 MAXLEVEL=7 -> base 16 16 64,   2 levels "4 2"  -> max 128 128 512  (dx_min 2^-7)
#
# With REF_RATIO=2 it goes back to the classic ratio 2 (one level per
# MAXLEVEL step):
#   REF_RATIO=2 MINLEVEL=6 MAXLEVEL=8 -> base 64 64 256, 2 levels "2 2"
#
# Usage: sourced by the launchers (. ./livelli.sh). Reads MINLEVEL/MAXLEVEL
# from the environment or from the first two positional arguments. Exports:
#   MINLEVEL  MAXLEVEL  ZDIV  REF_RATIO
#   AMR_NCELL  AMR_MAXLEV  AMR_REFRATIO
#
# If you change LZ in saetta3d_inputs (geometry.prob_hi) update ZDIV = LZ.
# ----------------------------------------------------------------------------
if [ $# -ge 2 ] && [ -z "${MINLEVEL:-}" ] && [ -z "${MAXLEVEL:-}" ]; then
    MINLEVEL=$1
    MAXLEVEL=$2
fi
MINLEVEL="${MINLEVEL:-4}"
MAXLEVEL="${MAXLEVEL:-7}"
ZDIV="${ZDIV:-4}"
REF_RATIO="${REF_RATIO:-4}"

case "${MINLEVEL}${MAXLEVEL}${ZDIV}${REF_RATIO}" in
    *[!0-9]*)
        echo "errore: MINLEVEL/MAXLEVEL/ZDIV/REF_RATIO devono essere interi (ricevuti: $MINLEVEL/$MAXLEVEL/$ZDIV/$REF_RATIO)" >&2
        exit 1 ;;
esac
[ "$MAXLEVEL" -ge "$MINLEVEL" ] || {
    echo "errore: serve MAXLEVEL >= MINLEVEL (ricevuti: $MINLEVEL/$MAXLEVEL)" >&2
    exit 1; }
[ "$REF_RATIO" -ge 2 ] || {
    echo "errore: REF_RATIO deve essere >= 2 (ricevuto: $REF_RATIO)" >&2
    exit 1; }

# K = log2(REF_RATIO)
K=0; TMP=$REF_RATIO
while [ "$TMP" -gt 1 ]; do TMP=$((TMP / 2)); K=$((K + 1)); done
[ $((1 << K)) -eq "$REF_RATIO" ] || {
    echo "errore: REF_RATIO deve essere una potenza di 2 >= 2 (ricevuto: $REF_RATIO)" >&2
    exit 1; }

NX=$((1 << MINLEVEL))
NZ=$((ZDIV * NX))
AMR_NCELL="$NX $NX $NZ"

# Decomposition: D = number of "2D steps" to cover.
D=$((MAXLEVEL - MINLEVEL))
AMR_MAXLEV=0
AMR_REFRATIO=""
if [ "$D" -gt 0 ]; then
    NFULL=$((D / K))
    REM=$((D % K))
    RATIOS=""
    i=0
    while [ "$i" -lt "$NFULL" ]; do
        if [ -z "$RATIOS" ]; then RATIOS="$REF_RATIO"; else RATIOS="$RATIOS $REF_RATIO"; fi
        i=$((i + 1))
    done
    if [ "$REM" -gt 0 ]; then
        R2=$((1 << REM))
        if [ -z "$RATIOS" ]; then RATIOS="$R2"; else RATIOS="$RATIOS $R2"; fi
    fi
    AMR_MAXLEV=$NFULL
    [ "$REM" -gt 0 ] && AMR_MAXLEV=$((AMR_MAXLEV + 1))
    AMR_REFRATIO="$RATIOS"
fi
export MINLEVEL MAXLEVEL ZDIV REF_RATIO AMR_NCELL AMR_MAXLEV AMR_REFRATIO
