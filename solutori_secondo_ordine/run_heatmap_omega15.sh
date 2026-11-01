#!/bin/sh
# Error heatmap on omega with T=15 and NIT=3, reusing the same runs
# as the u_x heatmap (same LMIN/LMAX/EPS/TMAX/NIT): only the norm changes.
set -e
cd "$(dirname "$0")"

NIT=3 TMAX=15 LMIN=4 LMAX=10 NP=16 \
EPS_LIST="1e-2 3e-3 1e-3 3e-4 1e-4" \
SKIP_RUNS=1 DATA_DIR=heatmap_adattiva_ux_t15_nit3 \
OUT=heatmap_adattiva_omega_t15_nit3 \
MATRIX_NAME=omega_matrix.bin FIELD_LABEL=omega \
./heatmap_adattiva.sh
