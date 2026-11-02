#!/bin/bash
# check_env.sh -- verifica i prerequisiti del prototipo su questo PC.
#   ./scripts/check_env.sh
set -u
ok=0; ko=0
say() { printf '%-38s %s\n' "$1" "$2"; }
has() { if command -v "$1" >/dev/null 2>&1; then say "$1" "OK ($(command -v "$1"))"; ok=$((ok+1)); else say "$1" "MANCA"; ko=$((ko+1)); fi; }

echo "=== mpi_fissa: controllo ambiente ==="
has mpirun
has mpicxx
has python3
say "nvcc (PATH)" "$(command -v nvcc || echo 'non in PATH')"
[ -x "${HOME}/cuda-13.1/bin/nvcc" ] && say "nvcc (~/cuda-13.1)" "OK" || say "nvcc (~/cuda-13.1)" "non trovato"
echo
mpirun --version 2>/dev/null | head -2 || true
echo
echo "-- MPI: numero di slot locali --"
mpirun --hostfile "$(dirname "$0")/hostfile_locale.txt" -np 2 hostname 2>/dev/null || echo "mpirun -np 2 hostname FALLITO"
echo
echo "-- GPU --"
nvidia-smi --query-gpu=name,memory.total,driver_version,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi non disponibile"
echo
[ -f "$HOME/cuda-env.sh" ] && echo "suggerimento: source ~/cuda-env.sh (imposta CUDA_HOME/PATH)" || true
echo
if [ "$ko" -eq 0 ]; then echo "PREREQUISITI MINIMI OK"; else echo "$ko voci mancanti"; fi
