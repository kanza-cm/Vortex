#!/bin/sh
# Scheduler for the fixed grid error study:
# it waits DELAY seconds (default 2h30 = 9000s) and then starts
# error_study_fissa_2nd.sh with NIT=12, TMAX=200, LMAX=10.
# If the adaptive heatmap (tmux session heat12) is still running,
# it waits for it to finish so the 16 CPUs are not overloaded.
set -e
cd "$(dirname "$0")"

DELAY="${DELAY:-9000}"
TARGET=$(( $(date +%s) + DELAY ))
TARGET_H=$(date -d "@$TARGET" '+%F %T')
echo "[$(date '+%F %T')] avvio programmato per $TARGET_H (attesa ${DELAY}s)"

while [ "$(date +%s)" -lt "$TARGET" ]; do
    sleep 30
done

while tmux has-session -t heat12 2>/dev/null; do
    echo "[$(date '+%F %T')] heat12 ancora attiva, attendo la fine..."
    sleep 60
done

echo "[$(date '+%F %T')] lancio lo studio fissa NIT=12 TMAX=200 LMAX=10"
if NIT=12 TMAX=200 LMAX=10 OUT=error_study_fissa_2nd_nit12_t200 \
        ./error_study_fissa_2nd.sh > error_study_fissa_2nd_nit12_t200.log 2>&1; then
    echo "[$(date '+%F %T')] studio completato"
else
    rc=$?
    echo "[$(date '+%F %T')] studio FALLITO (rc=$rc), vedi error_study_fissa_2nd_nit12_t200.log"
    exit $rc
fi
