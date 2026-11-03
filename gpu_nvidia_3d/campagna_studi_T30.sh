#!/bin/bash
# Campagna notturna: entrambe le configurazioni, errore fisso + heatmap adattiva a T=30.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
for cfg in vortici_inclinati vortici_non_inclinati; do
    echo "########## $cfg ##########"
    bash "$HERE/$cfg/esegui_studio.sh"
done
echo "########## CAMPAGNA COMPLETATA ##########"
