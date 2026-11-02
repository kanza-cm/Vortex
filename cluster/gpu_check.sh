#!/bin/bash
# Checks that this machine (WSL) can run the GPU solvers.
# Run it on the slave:  ./gpu_check.sh
set -u
ok=0; ko=0
say() { printf '%-34s %s\n' "$1" "$2"; }
check() {  # $1 label, $2 shell test
    if eval "$2" >/dev/null 2>&1; then say "$1" "OK"; ok=$((ok+1)); else say "$1" "MANCA"; ko=$((ko+1)); fi
}

echo "=== WSL GPU check ==="
uname -r
echo
check "nvidia-smi"            "command -v nvidia-smi"
check "GPU visibile"          "nvidia-smi --query-gpu=name --format=csv,noheader | grep -q ."
check "/dev/dxg (WSL2 GPU)"   "[ -e /dev/dxg ]"
check "libcuda WSL"           "ls /usr/lib/wsl/lib/libcuda.so* >/dev/null 2>&1"
[ -f "$HOME/cuda-env.sh" ] && say "cuda-env.sh" "OK" || { say "cuda-env.sh" "MANCA"; ko=$((ko+1)); }
check "nvcc"                  "command -v nvcc >/dev/null 2>&1 || [ -x $HOME/cuda-13.1/bin/nvcc ]"
check "qcc (Basilisk)"        "command -v qcc >/dev/null 2>&1 || [ -x $HOME/basilisk/src/qcc ]"
check "basilisk src"          "[ -d $HOME/basilisk/src ]"
check "incflo build"          "[ -x $HOME/src/incflo/build/incflo.ex ]"
echo
nvidia-smi --query-gpu=name,memory.total,driver_version,memory.used --format=csv,noheader 2>/dev/null
echo
if [ "$ko" -eq 0 ]; then
    echo "TUTTO OK: questa macchina puo' girare i solver GPU."
    echo "Test veloce:  cd gpu_nvidia_3d && EXTRA='-DMAXLEVEL=4 -DTMAX=0.05 -DMOVIE=0' ./build.sh && ./run.sh"
else
    echo "$ko controlli mancanti (vedi cluster/README.md, sezione GPU su WSL)."
fi
exit "$ko"
