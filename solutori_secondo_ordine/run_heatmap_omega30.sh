#!/bin/sh
# L_inf and L2 heatmap on omega for T=30, NIT=3, same axes as T=15.
set -e
cd "$(dirname "$0")"

NIT=3 TMAX=30 LMIN=4 LMAX=10 NP=16 \
EPS_LIST="1e-2 3e-3 1e-3 3e-4 1e-4" \
OUT=heatmap_adattiva_omega_t30_nit3 \
MATRIX_NAME=omega_matrix.bin FIELD_LABEL=omega \
./heatmap_adattiva.sh
