#!/bin/sh
# Comparison with the user's heatmap (t=15, L_inf on u_x, NIT=3):
# same axes (MAXLEVEL 4-10, 1/EPS 1e2-1e4), but our solver
# adapts on omega. The error is computed on u_x (ux_matrix.bin).
set -e
cd "$(dirname "$0")"

NIT=3 TMAX=15 LMIN=4 LMAX=10 NP=16 \
EPS_LIST="1e-2 3e-3 1e-3 3e-4 1e-4" \
OUT=heatmap_adattiva_ux_t15_nit3 \
MATRIX_NAME=ux_matrix.bin FIELD_LABEL="u_x" QCC_EXTRA="-DUX_MATRIX" \
./heatmap_adattiva.sh
