#!/usr/bin/env bash
# ============================================================================
# validate_tempo.sh -- studio di raffinamento temporale (CFL) del prototipo
# mpi_fissa contro saetta3d (Basilisk centered.h) alla STESSA griglia.
#
# Ambito:
#   griglia 64 x 64 x 256 (NX=64; saetta: NDIV0=1, MAXLEVEL=6)
#   TMAX=1.0, nu=1e-5, poisson_nit=2, np=1
#   CFL = 0.3, 0.15, 0.1
#
# Per ogni CFL:
#   1) compila saetta3d in /tmp/saetta3d_cfl<tag> con i flag del riferimento;
#   2) esegue saetta3d e il prototipo (GPU, fallback CPU);
#   3) confronta i campi finali (confronta_campi.py) e le diagnostiche
#      (confronta_diagnostics.py);
#   4) scrive una riga in tabella.txt.
# Alla fine stima l'ordine osservato (ordine.txt) e lo appende a tabella.txt.
#
# Garanzie:
#   - nessuna sovrascrittura: prima esecuzione in out/validazione_tempo/ se e'
#     vuota, altrimenti sottocartella con timestamp; build in /tmp;
#   - timeout per ogni comando esterno;
#   - una riga incompleta riporta NA + motivo, senza fermare le altre.
#
# Uso:
#   bash /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/validazione/validate_tempo.sh
#
# Variabili (tutte opzionali):
#   PROJECT=...        percorso di mpi_fissa (default: ricerca automatica)
#   OUTROOT=...        cartella risultati (default: $PROJECT/out/validazione_tempo)
#   SESSION=...        nome sottocartella (default: timestamp se OUTROOT non vuota)
#   FORCE_CPU=1        salta GPU e usa il prototipo CPU
#   NX=64 TMAX_RUN=1.0 NU=1e-5 POISSON_NIT=2
#   CFL_LIST="0.3 0.15 0.1"
#   T_BUILD_PROTO=1200 T_SMOKE=300 T_BUILD_SAETTA=1800
#   T_RUN_PROTO=1800 T_RUN_SAETTA=3600 T_POST=600 T_BUILD_LIBS=1800
#
# Note:
#   - il prototipo NON viene compilato in mpi_fissa/bin: i binari vanno in
#     /tmp/mpi_fissa_bin_<timestamp>_<pid>;
#   - qcc non digerisce i percorsi con spazi: saetta compila in /tmp;
#   - il confronto diagnostiche completo include t=0, dove il prototipo ha
#     separation=0 per un artefatto noto (due massimi simmetrici). Il
#     confronto che conta e' quello a t=1; gli altri sono record.
# ============================================================================

set -uo pipefail
export LC_ALL=C
umask 022

# ------------------------------- parametri ---------------------------------
NX="${NX:-64}"
TMAX_RUN="${TMAX_RUN:-1.0}"
NU="${NU:-1e-5}"
POISSON_NIT="${POISSON_NIT:-2}"
CFL_LIST="${CFL_LIST:-0.3 0.15 0.1}"

SAETTA_NDIV0="${SAETTA_NDIV0:-1}"
SAETTA_MAXLEVEL="${SAETTA_MAXLEVEL:-6}"

T_BUILD_PROTO="${T_BUILD_PROTO:-1200}"
T_SMOKE="${T_SMOKE:-300}"
T_BUILD_SAETTA="${T_BUILD_SAETTA:-1800}"
T_RUN_PROTO="${T_RUN_PROTO:-1800}"
T_RUN_SAETTA="${T_RUN_SAETTA:-3600}"
T_POST="${T_POST:-600}"
T_BUILD_LIBS="${T_BUILD_LIBS:-1800}"

EXPECTED_CELLS=$(( NX * NX * 4 * NX ))

# ------------------------------- utilita' ----------------------------------
msg()  { printf '%s\n' "$*"; }
die()  { printf 'ERRORE: %s\n' "$*" >&2; exit 2; }

SCRIPT_LOG=""
say() {
    if [ -n "$SCRIPT_LOG" ]; then
        printf '%s\n' "$*" | tee -a "$SCRIPT_LOG"
    else
        printf '%s\n' "$*"
    fi
}
section() {
    say ""
    say "=================================================================="
    say "$*"
    say "=================================================================="
}
tail_log() {
    local f="$1"
    if [ -f "$f" ]; then
        tail -n 20 "$f" 2>/dev/null | sed 's/^/    /'
    fi
}
close_to() {
    # close_to A B [tol] -> 0 se |A-B| <= tol
    awk -v a="$1" -v b="$2" -v tol="${3:-1e-6}" \
        'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= tol) }'
}

# --------------------------- risoluzione percorsi --------------------------
PROJECT="${PROJECT:-}"
PROTO=""
find_proto() {
    local cand
    for cand in "$PROJECT" "$PWD" "$PWD/mpi_fissa" "/home/zorka/simulazioni/Test 1"; do
        [ -n "$cand" ] || continue
        if [ -f "$cand/mpi_fissa/src/mpi_fissa.cu" ] && \
           [ -f "$cand/mpi_fissa/scripts/build.sh" ]; then
            PROTO="$cand/mpi_fissa"
            return 0
        fi
        if [ -f "$cand/src/mpi_fissa.cu" ] && \
           [ -f "$cand/scripts/build.sh" ]; then
            PROTO="$cand"
            return 0
        fi
    done
    return 1
}

find_proto || die "non trovo mpi_fissa (usa PROJECT=/percorso/di/mpi_fissa)"
PROTO=$(cd "$PROTO" && pwd) || die "PROTO non accessibile: $PROTO"
REPO=$(dirname "$PROTO")

if [ -z "${SAETTA_SRC:-}" ]; then
    if [ -f "$REPO/gpu_nvidia_3d/saetta3d.c" ]; then
        SAETTA_SRC="$REPO/gpu_nvidia_3d/saetta3d.c"
    elif [ -f "$PWD/gpu_nvidia_3d/saetta3d.c" ]; then
        SAETTA_SRC="$PWD/gpu_nvidia_3d/saetta3d.c"
    else
        SAETTA_SRC="$REPO/gpu_nvidia_3d/saetta3d.c"
    fi
fi
[ -f "$SAETTA_SRC" ] || die "non trovo saetta3d.c: $SAETTA_SRC"
SAETTA_SRC=$(cd "$(dirname "$SAETTA_SRC")" && pwd)/$(basename "$SAETTA_SRC")

# ------------------------------ cartella output -----------------------------
OUTROOT="${OUTROOT:-$PROTO/out/validazione_tempo}"
mkdir -p "$OUTROOT" || die "non posso creare OUTROOT=$OUTROOT"
OUTROOT=$(cd "$OUTROOT" && pwd) || die "OUTROOT non accessibile: $OUTROOT"

REUSE_ROOT=0
if [ -n "${SESSION:-}" ]; then
    OUT="$OUTROOT/$SESSION"
elif [ -z "$(ls -A "$OUTROOT" 2>/dev/null)" ]; then
    OUT="$OUTROOT"
    REUSE_ROOT=1
else
    OUT="$OUTROOT/$(date +%Y%m%d_%H%M%S)"
fi
if [ "$REUSE_ROOT" -eq 0 ]; then
    while [ -e "$OUT" ]; do
        OUT="${OUT}_$$"
    done
fi
mkdir -p "$OUT" || die "non posso creare OUT=$OUT"
SCRIPT_LOG="$OUT/validate_tempo.log"
: > "$SCRIPT_LOG" || die "non posso scrivere $SCRIPT_LOG"

# ------------------------------- ambiente -----------------------------------
BASILISK="${BASILISK:-$HOME/basilisk/src}"
export BASILISK
[ -d "$BASILISK" ] || die "BASILISK non trovato: $BASILISK (esporta BASILISK=...)"
export PATH="$PATH:$BASILISK"

{
    echo "# ambiente validate_tempo.sh"
    echo "# data: $(date -Is 2>/dev/null || date)"
    echo "# host: $(hostname 2>/dev/null || echo sconosciuto)"
    echo "# pwd: $PWD"
    echo "# PROTO=$PROTO"
    echo "# REPO=$REPO"
    echo "# SAETTA_SRC=$SAETTA_SRC"
    echo "# OUT=$OUT"
    echo "# NX=$NX TMAX_RUN=$TMAX_RUN NU=$NU POISSON_NIT=$POISSON_NIT CFL_LIST=$CFL_LIST"
    echo "# EXPECTED_CELLS=$EXPECTED_CELLS"
    echo "#"
    uname -a 2>/dev/null || true
    echo "#"
    for c in mpirun mpicxx python3 qcc nvcc make timeout; do
        printf '# %s: ' "$c"
        command -v "$c" 2>/dev/null || echo "NON TROVATO"
    done
    echo "#"
    ( mpirun --version 2>/dev/null | head -n 3 ) || true
    echo "#"
    ( python3 --version 2>/dev/null ) || true
    echo "#"
    ( nvidia-smi -L 2>/dev/null ) || true
    echo "#"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$PROTO/src/mpi_fissa.cu" "$SAETTA_SRC" 2>/dev/null || true
    fi
} > "$OUT/ambiente.txt" 2>&1

# ------------------------------- prerequisiti -------------------------------
for c in mpirun mpicxx python3 timeout awk sed grep head tail mkdir cp date hostname uname dirname basename readlink tee; do
    command -v "$c" >/dev/null 2>&1 || die "comando richiesto mancante: $c"
done

if [ -n "${QCC:-}" ]; then
    [ -x "${QCC}" ] || die "QCC non eseguibile: $QCC"
elif [ -x "$BASILISK/qcc" ]; then
    QCC="$BASILISK/qcc"
else
    QCC=$(command -v qcc 2>/dev/null || true)
fi
[ -n "${QCC:-}" ] && [ -x "${QCC:-}" ] || \
    die "qcc non trovato: aggiungi \$BASILISK al PATH o imposta QCC=..."

CUDA_TARGET="${CUDA_TARGET:-}"
if [ -z "$CUDA_TARGET" ]; then
    for cand in "${CUDA:-}" \
                "$HOME/cuda-13.1/targets/x86_64-linux" \
                "$HOME/cuda-13.1" \
                "/usr/local/cuda-13.1/targets/x86_64-linux" \
                "/usr/local/cuda-13.1" \
                "/usr/local/cuda" \
                "$HOME/cuda"; do
        [ -n "$cand" ] || continue
        if [ -f "$cand/targets/x86_64-linux/include/cuda_runtime.h" ]; then
            CUDA_TARGET="$cand/targets/x86_64-linux"
            break
        fi
        if [ -f "$cand/include/cuda_runtime.h" ]; then
            CUDA_TARGET="$cand"
            break
        fi
    done
fi
[ -n "$CUDA_TARGET" ] || die "header CUDA non trovati: imposta CUDA=... o CUDA_TARGET=..."
[ -f "$CUDA_TARGET/include/cuda_runtime.h" ] || die "cuda_runtime.h non trovato in $CUDA_TARGET/include"
export LD_LIBRARY_PATH="$CUDA_TARGET/lib:$CUDA_TARGET/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# nvcc (solo per la build GPU del prototipo; se manca si usa la CPU)
CUDA_ROOT="${CUDA:-}"
if [ -n "$CUDA_ROOT" ] && \
   [ ! -x "$CUDA_ROOT/bin/nvcc" ] && \
   [ ! -x "$CUDA_ROOT/targets/x86_64-linux/bin/nvcc" ]; then
    # CUDA era puntato alla directory target (include/lib), non alla radice
    CUDA_ROOT=""
fi
if [ -z "$CUDA_ROOT" ]; then
    for cand in "$HOME/cuda-13.1" "/usr/local/cuda-13.1" "/usr/local/cuda" "$HOME/cuda"; do
        [ -n "$cand" ] || continue
        if [ -x "$cand/bin/nvcc" ]; then
            CUDA_ROOT="$cand"
            break
        fi
        if [ -x "$cand/targets/x86_64-linux/bin/nvcc" ]; then
            CUDA_ROOT="$cand"
            break
        fi
    done
fi
if [ -z "$CUDA_ROOT" ] && command -v nvcc >/dev/null 2>&1; then
    _nv=$(readlink -f "$(command -v nvcc)" 2>/dev/null || command -v nvcc)
    CUDA_ROOT=$(dirname "$(dirname "$_nv")")
fi
if [ -n "$CUDA_ROOT" ]; then
    export CUDA="$CUDA_ROOT"
else
    say "ATTENZIONE: nvcc non trovato; la build GPU del prototipo fallira' e si usera' la CPU."
fi

# librerie Basilisk GPU (se mancano, prova a costruirle una volta)
ensure_basilisk_libs() {
    local rc=0
    if [ ! -f "$BASILISK/grid/gpu/liberrors.a" ]; then
        say "  manca $BASILISK/grid/gpu/liberrors.a: provo 'make liberrors.a'"
        ( cd "$BASILISK/grid/gpu" && timeout -k 30 "$T_BUILD_LIBS" make liberrors.a ) \
            >"$OUT/build_liberrors.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -f "$BASILISK/grid/gpu/liberrors.a" ]; then
            say "  FALLITA (exit $rc); log: $OUT/build_liberrors.log"
            return 1
        fi
    fi
    if [ ! -f "$BASILISK/grid/cuda/libbuda.a" ]; then
        say "  manca $BASILISK/grid/cuda/libbuda.a: provo 'make libbuda.a'"
        ( cd "$BASILISK/grid/cuda" && \
          CPATH="$CUDA_TARGET/include" timeout -k 30 "$T_BUILD_LIBS" make libbuda.a ) \
            >"$OUT/build_libbuda.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -f "$BASILISK/grid/cuda/libbuda.a" ]; then
            say "  FALLITA (exit $rc); log: $OUT/build_libbuda.log"
            return 1
        fi
    fi
    return 0
}

# prerequisito di saetta3d: le librerie GPU di Basilisk (prima della build
# del prototipo, cosi' un errore qui non fa perdere tempo)
ensure_basilisk_libs || die "librerie Basilisk GPU mancanti: impossibile compilare saetta3d"

# --------------------------- build prototipo -------------------------------
BIN_DIR="${BIN_DIR:-/tmp/mpi_fissa_bin_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$BIN_DIR" || die "non posso creare BIN_DIR=$BIN_DIR"

try_build_proto() {
    local kind="$1"
    local log="$OUT/build_proto_${kind}.log"
    local bin="$BIN_DIR/mpi_fissa_${kind}"
    local rc=0
    section "Build prototipo $kind (NX=$NX)"
    ( cd "$PROTO" && NX="$NX" OUT="$BIN_DIR" \
        timeout -k 30 "$T_BUILD_PROTO" bash ./scripts/build.sh "$kind" ) \
        >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -x "$bin" ]; then
        say "  OK: $bin"
        return 0
    fi
    say "  FALLITA (exit $rc); ultime righe del log:"
    tail_log "$log"
    return 1
}

try_smoke_proto() {
    local kind="$1"
    local bin="$BIN_DIR/mpi_fissa_${kind}"
    local smoke="$OUT/smoke_${kind}"
    local log="$OUT/smoke_${kind}.log"
    local rc=0
    section "Smoke test prototipo $kind (-tmax 0)"
    ( cd "$PROTO" && BIN="$bin" \
        timeout -k 30 "$T_SMOKE" bash "./scripts/run_${kind}.sh" 1 \
        -tmax 0 -out "$smoke" ) >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$smoke/diagnostics.dat" ]; then
        say "  OK"
        return 0
    fi
    say "  FALLITO (exit $rc); ultime righe del log:"
    tail_log "$log"
    return 1
}

PROTO_KIND=""
if [ "${FORCE_CPU:-0}" = "1" ]; then
    say "FORCE_CPU=1: salto la build/test GPU."
else
    if try_build_proto gpu; then
        if try_smoke_proto gpu; then
            PROTO_KIND="gpu"
        else
            say "Smoke GPU fallito: passo al prototipo CPU."
        fi
    else
        say "Build GPU fallita: passo al prototipo CPU."
    fi
fi
if [ -z "$PROTO_KIND" ]; then
    try_build_proto cpu || die "build CPU del prototipo fallita"
    try_smoke_proto cpu || die "smoke test CPU del prototipo fallito"
    PROTO_KIND="cpu"
fi
say "Prototipo da usare: $PROTO_KIND (binari in $BIN_DIR)"

# --------------------------- funzioni per i run -----------------------------
# saetta_case: imposta NOTE (globale) e ritorna 0 se build+run+file ok.
saetta_case() {
    local cfl="$1"
    local tag="$2"
    local cdir="$3"
    local tmpbase="/tmp/saetta3d_cfl${tag}"
    local rc=0

    if [ -e "$tmpbase" ]; then
        tmpbase="${tmpbase}_$(date +%s)_$$"
    fi
    if ! mkdir "$tmpbase" 2>/dev/null; then
        NOTE="${NOTE}tmp_dir_failed=${tmpbase}; "
        return 1
    fi
    if ! cp "$SAETTA_SRC" "$tmpbase/saetta3d.c"; then
        NOTE="${NOTE}cp_saetta_failed; "
        return 1
    fi

    say "  build saetta3d in $tmpbase"
    ( cd "$tmpbase" && \
      timeout -k 30 "$T_BUILD_SAETTA" "$QCC" \
        -autolink -disable-dimensions -O3 -grid=cuda/multigrid3D \
        -I"$CUDA_TARGET/include" \
        "${CUDA_LINK[@]}" \
        -DCFL_USER="$cfl" -DNU="$NU" -DMOVIE=0 -DVTK=0 -DMATRIX=1 \
        -DNDIV0="$SAETTA_NDIV0" -DMAXLEVEL="$SAETTA_MAXLEVEL" -DTMAX="$TMAX_RUN" \
        saetta3d.c -o saetta3d_gpu -lm -lz ) \
        >"$cdir/saetta_build.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -x "$tmpbase/saetta3d_gpu" ]; then
        NOTE="${NOTE}saetta_build_exit=${rc}; "
        return 1
    fi

    say "  run saetta3d (tmax=$TMAX_RUN)"
    ( cd "$tmpbase" && timeout -k 30 "$T_RUN_SAETTA" ./saetta3d_gpu ) \
        >"$tmpbase/log_run.txt" 2>&1
    rc=$?

    # copia quello che c'e' anche in caso di fallimento (per debug)
    mkdir -p "$cdir/saetta" 2>/dev/null || true
    for f in diagnostics.dat omega3d_matrix.bin resa_max.dat log_run.txt; do
        if [ -f "$tmpbase/$f" ]; then
            cp "$tmpbase/$f" "$cdir/saetta/" 2>/dev/null || true
        fi
    done

    if [ "$rc" -ne 0 ]; then
        NOTE="${NOTE}saetta_run_exit=${rc}; "
        return 1
    fi
    if [ ! -s "$tmpbase/diagnostics.dat" ]; then
        NOTE="${NOTE}saetta_no_diagnostics; "
        return 1
    fi
    if [ ! -s "$tmpbase/omega3d_matrix.bin" ]; then
        NOTE="${NOTE}saetta_no_matrix; "
        return 1
    fi
    return 0
}

CUDA_LINK=(-L"$CUDA_TARGET/lib")
if [ -d "$CUDA_TARGET/lib64" ]; then
    CUDA_LINK+=(-L"$CUDA_TARGET/lib64")
fi
if [ -d "$CUDA_TARGET/lib/stubs" ]; then
    CUDA_LINK+=(-L"$CUDA_TARGET/lib/stubs")
fi
if [ -d "$CUDA_TARGET/lib64/stubs" ]; then
    CUDA_LINK+=(-L"$CUDA_TARGET/lib64/stubs")
fi

run_diag() {
    # run_diag A B LOG LABEL_A LABEL_B
    local a="$1"
    local b="$2"
    local log="$3"
    local la="${4:-A}"
    local lb="${5:-B}"
    local rc=0
    timeout -k 10 "$T_POST" python3 "$PROTO/scripts/confronta_diagnostics.py" \
        "$a" "$b" --label "$la" --label "$lb" >"$log" 2>&1
    rc=$?
    return "$rc"
}

filter_time() {
    # filter_time SRC LO HI DST (include l'header; righe con LO <= t <= HI)
    local src="$1"
    local lo="$2"
    local hi="$3"
    local dst="$4"
    {
        head -n 1 "$src"
        awk -v lo="$lo" -v hi="$hi" \
            '!/^#/ && $1+0 >= lo && $1+0 <= hi' "$src"
    } > "$dst"
}

# ------------------------------- tabella -----------------------------------
{
    printf '%-5s | %-13s | %-11s | %-11s | %-10s | %-11s | %s\n' \
        "cfl" "dt_proto_medio" "rms_rel" "linf" "passi_proto" "passi_saetta" "note"
} > "$OUT/tabella.txt"
printf '# cfl rms_rel dt_proto_medio\n' > "$OUT/dati_ordine.txt"
printf '# cfl saetta_dt0\n' > "$OUT/saetta_dt0.txt"

TECH_FAIL=0

for cfl in $CFL_LIST; do
    # CFL_LIST e' volutamente word-split: "0.3 0.15 0.1"
    TAG=$(awk -v c="$cfl" 'BEGIN { printf "%03d", int(c*100 + 0.5) }')
    [ -n "$TAG" ] || die "CFL non valido: $cfl"
    CDIR="$OUT/cfl$TAG"
    mkdir -p "$CDIR" || die "non posso creare $CDIR"

    section "CFL=$cfl  (tag $TAG, griglia ${NX}x${NX}x$((4*NX)))"

    note=""
    proto_ok=0
    saetta_ok=0
    dtmean="NA"
    dt_saetta="NA"
    rms="NA"
    linf="NA"
    psteps="NA"
    ssteps="NA"
    st_dt0="NA"

    # ---------------- prototipo ----------------
    PDIR="$CDIR/proto"
    PLOG="$CDIR/proto_run.log"
    rc=0
    ( cd "$PROTO" && BIN="$BIN_DIR/mpi_fissa_$PROTO_KIND" \
      timeout -k 30 "$T_RUN_PROTO" bash "./scripts/run_${PROTO_KIND}.sh" 1 \
        -tmax "$TMAX_RUN" -cfl "$cfl" -nu "$NU" -poisson_nit "$POISSON_NIT" \
        -out "$PDIR" ) >"$PLOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$PDIR/done_0.txt" ] && \
       [ -s "$PDIR/diagnostics.dat" ] && [ -s "$PDIR/omega3d_matrix.bin" ]; then
        proto_ok=1
        say "  prototipo OK ($PDIR)"
    else
        note="${note}proto_run_exit=${rc}; "
        say "  prototipo FALLITO (exit $rc); log: $PLOG"
        tail_log "$PLOG"
    fi

    if [ "$proto_ok" -eq 1 ]; then
        vals=$(awk '!/^#/ && NF>=12 { t=$1; it=$3; cells=$12; n++ }
                    END { if (n > 0) print t, it, cells }' "$PDIR/diagnostics.dat")
        if [ -n "$vals" ]; then
            read -r pt psteps pcells <<< "$vals"
        else
            pt="NA"; psteps="NA"; pcells="NA"
        fi
        if [ -f "$PDIR/timing.dat" ]; then
            dtmean=$(awk 'NR>1 && $3+0 > 0 { s+=$3; n++ }
                           END { if (n > 0) printf "%.6e", s/n; else printf "NA" }' \
                         "$PDIR/timing.dat" 2>/dev/null)
        fi
        [ -n "${dtmean:-}" ] || dtmean="NA"

        if ! close_to "${pt:-NA}" "$TMAX_RUN" 1e-6; then
            note="${note}proto_t_finale=${pt}; "
            proto_ok=0
        fi
        if [ "${pcells:-NA}" != "$EXPECTED_CELLS" ]; then
            note="${note}proto_cells=${pcells}; "
            proto_ok=0
        fi
    fi

    # ---------------- saetta3d ----------------
    NOTE=""
    if saetta_case "$cfl" "$TAG" "$CDIR"; then
        saetta_ok=1
        say "  saetta3d OK ($CDIR/saetta)"
    else
        note="${note}${NOTE}"
        say "  saetta3d FALLITO"
        tail_log "$CDIR/saetta_build.log"
    fi

    if [ "$saetta_ok" -eq 1 ]; then
        svals=$(awk '!/^#/ && NF>=12 { t=$1; it=$3; cells=$12; n++ }
                     END { if (n > 0) print t, it, cells }' \
                   "$CDIR/saetta/diagnostics.dat")
        if [ -n "$svals" ]; then
            read -r st ssteps scells <<< "$svals"
        else
            st="NA"; ssteps="NA"; scells="NA"
        fi
        st_dt0=$(awk '!/^#/ && $1+0 == 0 { print $2; exit }' \
                     "$CDIR/saetta/diagnostics.dat")
        [ -n "${st_dt0:-}" ] || st_dt0="NA"
        if close_to "${st:-NA}" "$TMAX_RUN" 1e-6; then :; else
            note="${note}saetta_t_finale=${st}; "
            saetta_ok=0
        fi
        if [ "${scells:-NA}" != "$EXPECTED_CELLS" ]; then
            note="${note}saetta_cells=${scells}; "
            saetta_ok=0
        fi
        if [ "${ssteps:-0}" -gt 0 ] 2>/dev/null; then
            dt_saetta=$(awk -v it="$ssteps" -v tm="$TMAX_RUN" \
                'BEGIN { printf "%.6e", tm/it }')
        else
            dt_saetta="NA"
        fi
    fi

    # ---------------- confronti ----------------
    if [ "$proto_ok" -eq 1 ] && [ "$saetta_ok" -eq 1 ]; then
        CPLOG="$CDIR/confronto_campi.txt"
        rc=0
        timeout -k 10 "$T_POST" python3 "$PROTO/scripts/confronta_campi.py" \
            "$PDIR/omega3d_matrix.bin" "$CDIR/saetta/omega3d_matrix.bin" \
            --tol-rms 1e-2 \
            --label "proto cfl=$cfl" --label "saetta cfl=$cfl" \
            >"$CPLOG" 2>&1
        rc=$?
        rms=$(awk '/^rms_rel/ { print $3 }' "$CPLOG" 2>/dev/null)
        linf=$(awk '/^linf/ { print $3 }' "$CPLOG" 2>/dev/null)
        if [ -z "${rms:-}" ]; then
            rms="NA"
            note="${note}confronto_campi_exit=${rc}; "
        fi
        [ -n "${linf:-}" ] || linf="NA"

        # diagnostiche: completo, t=0, t=1
        run_diag "$PDIR/diagnostics.dat" "$CDIR/saetta/diagnostics.dat" \
            "$CDIR/confronto_diagnostics.txt" "proto cfl=$cfl" "saetta cfl=$cfl" \
            || true

        filter_time "$PDIR/diagnostics.dat" -1e-9 1e-9 "$CDIR/proto_diag_t0.dat"
        filter_time "$PDIR/diagnostics.dat" 0.999999 1.000001 "$CDIR/proto_diag_t1.dat"
        filter_time "$CDIR/saetta/diagnostics.dat" -1e-9 1e-9 "$CDIR/saetta_diag_t0.dat"
        filter_time "$CDIR/saetta/diagnostics.dat" 0.999999 1.000001 "$CDIR/saetta_diag_t1.dat"

        run_diag "$CDIR/proto_diag_t0.dat" "$CDIR/saetta_diag_t0.dat" \
            "$CDIR/confronto_diagnostics_t0.txt" "proto t=0" "saetta t=0" || true
        run_diag "$CDIR/proto_diag_t1.dat" "$CDIR/saetta_diag_t1.dat" \
            "$CDIR/confronto_diagnostics_t1.txt" "proto t=1" "saetta t=1" || true

        diag_max=$(awk '($4 == "OK" || $4 == "FUORI") { v=$2+0; if (v>m) m=v; n++ }
                        END { if (n > 0) printf "%.3e", m; else print "NA" }' \
                       "$CDIR/confronto_diagnostics_t1.txt" 2>/dev/null)
        zrel=$(awk '$1 == "enstrophy" { print $2 }' \
                   "$CDIR/confronto_diagnostics_t1.txt" 2>/dev/null)
        srel=$(awk '$1 == "separation" { print $2 }' \
                   "$CDIR/confronto_diagnostics_t1.txt" 2>/dev/null)
        [ -n "${diag_max:-}" ] || diag_max="NA"
        [ -n "${zrel:-}" ] || zrel="NA"
        [ -n "${srel:-}" ] || srel="NA"
        if [ "$diag_max" != "NA" ]; then
            note="${note}diag_t1_max=${diag_max}(Z=${zrel},sep=${srel}); "
        fi
    fi

    # ---------------- nota e riga ----------------
    if [ "$dtmean" != "NA" ] && [ "$dt_saetta" != "NA" ]; then
        ratio=$(awk -v a="$dtmean" -v b="$dt_saetta" \
            'BEGIN { if (b > 0) printf "%.2f", a/b; else print "NA" }')
        note="${note}dt_saetta~${dt_saetta}; dt_ratio~${ratio}; "
    fi
    if [ "$st_dt0" != "NA" ]; then
        note="${note}saetta_dt0=${st_dt0}; "
        printf '%s %s\n' "$cfl" "$st_dt0" >> "$OUT/saetta_dt0.txt"
    fi
    note="${note}backend=${PROTO_KIND}"
    note=$(printf '%s' "$note" | sed 's/;[[:space:]]*$//')
    [ -n "$note" ] || note="-"

    printf '%-5s | %-13s | %-11s | %-11s | %-10s | %-11s | %s\n' \
        "$cfl" "${dtmean:-NA}" "${rms:-NA}" "${linf:-NA}" \
        "${psteps:-NA}" "${ssteps:-NA}" "$note" >> "$OUT/tabella.txt"

    if [ "${rms:-NA}" != "NA" ]; then
        printf '%s %s %s\n' "$cfl" "$rms" "${dtmean:-NA}" >> "$OUT/dati_ordine.txt"
    fi

    if [ "$proto_ok" -eq 1 ] && [ "$saetta_ok" -eq 1 ] && [ "${rms:-NA}" != "NA" ]; then
        say "  confronto: rms_rel=$rms linf=$linf passi=$psteps/$ssteps"
    else
        TECH_FAIL=$((TECH_FAIL + 1))
        say "  riga incompleta (proto_ok=$proto_ok saetta_ok=$saetta_ok rms=$rms)"
    fi
done

# controllo che -DCFL_USER abbia davvero cambiato dt in saetta
if [ -f "$OUT/saetta_dt0.txt" ]; then
    read -r NDT NUNIQ <<< "$(awk '!/^#/ && $2 != "NA" { n++; v[$2]=1 }
                                   END { for (k in v) u++; print n+0, u+0 }' \
                                 "$OUT/saetta_dt0.txt")"
    if [ "${NDT:-0}" -ge 2 ] && [ "${NUNIQ:-0}" -le 1 ]; then
        say ""
        say "ATTENZIONE: i dt0 di saetta sono identici per CFL diversi."
        say "  -DCFL_USER potrebbe non essere efficace in questa build di centered.h:"
        say "  la colonna saetta della matrice sarebbe a CFL costante."
        say "  Controlla $OUT/saetta_dt0.txt e i log di build."
    fi
fi

# ------------------------- analisi dell'ordine ------------------------------
section "Analisi dell'ordine osservato"
python3 - "$OUT" <<'PY' | tee -a "$SCRIPT_LOG"
import math
import os
import sys

out = sys.argv[1]
rows = []
path = os.path.join(out, "dati_ordine.txt")
with open(path) as fp:
    for line in fp:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            cfl = float(parts[0])
            rms = float(parts[1])
            dt = float(parts[2]) if len(parts) > 2 and parts[2] != "NA" else None
        except (ValueError, IndexError):
            continue
        if cfl > 0 and rms > 0:
            rows.append((cfl, rms, dt))

rows.sort(key=lambda r: r[0], reverse=True)
lines = []
lines.append("")
lines.append("# " + "-" * 75)
lines.append("# Analisi dell'ordine osservato (stima su %d punti; non e' una misura)"
             % len(rows))
lines.append("# p_ij = ln(rms_i/rms_j) / ln(cfl_i/cfl_j)")
lines.append("# " + "-" * 75)

if len(rows) < 2:
    lines.append("# ATTENZIONE: meno di 2 righe valide: ordine non stimabile.")
    verdict = "NA"
else:
    def p_between(a, b):
        return math.log(a[1] / b[1]) / math.log(a[0] / b[0])

    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        p = p_between(a, b)
        lines.append("# p(%.4g -> %.4g) = %+.3f" % (a[0], b[0], p))

    p13 = p_between(rows[0], rows[-1])
    lines.append("# p(%.4g -> %.4g) = %+.3f   [stima principale]"
                 % (rows[0][0], rows[-1][0], p13))

    xs = [math.log(r[0]) for r in rows]
    ys = [math.log(r[1]) for r in rows]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    den = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / den if den else float("nan")
    lines.append("# pendenza log-log (LS, vs CFL) = %+.3f" % slope)

    if all(r[2] and r[2] > 0 for r in rows):
        xs = [math.log(r[2]) for r in rows]
        xbar = sum(xs) / len(xs)
        den = sum((x - xbar) ** 2 for x in xs)
        slope_dt = (sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / den
                    if den else float("nan"))
        lines.append("# pendenza log-log (LS, vs dt_proto_medio) = %+.3f" % slope_dt)

    mono = all(rows[i][1] > rows[i + 1][1] for i in range(len(rows) - 1))
    decr = [(rows[i][1] - rows[i + 1][1]) / rows[i][1]
            for i in range(len(rows) - 1)]
    weak = any(d < 0.10 for d in decr)
    ratio = rows[0][1] / rows[-1][1]
    lines.append("# monotonia stretta (rms decrescente al diminuire di CFL): %s"
                 % ("SI" if mono else "NO"))
    lines.append("# riduzioni relative tra CFL consecutivi: %s"
                 % ", ".join("%.1f%%" % (100.0 * d) for d in decr))
    lines.append("# rms(CFL max)/rms(CFL min) = %.3g "
                 "(con p=2 e CFL 3x atteso ~9; con p=1 ~3)" % ratio)

    if not mono:
        verdict = "NO: rms non decrescente"
    elif weak:
        verdict = "PARZIALE: almeno una riduzione <10% (possibile plateau)"
    elif p13 >= 1.5:
        verdict = "SI: ordine osservato %+.2f (compatibile con 2o ordine)" % p13
    elif p13 >= 0.8:
        verdict = "PARZIALE: monotono ma ordine %+.2f (indagare proiezione/dt)" % p13
    else:
        verdict = "NO: plateau (ordine %+.2f)" % p13
    lines.append("# criterio proposto (vedi PROTOCOLLO.md sez. 4): " + verdict)

text = "\n".join(lines) + "\n"
with open(os.path.join(out, "ordine.txt"), "w") as fp:
    fp.write(text)
with open(os.path.join(out, "tabella.txt"), "a") as fp:
    fp.write(text)
print(text, end="")
PY
PY_RC=${PIPESTATUS[0]:-1}
if [ "$PY_RC" -ne 0 ]; then
    say "ATTENZIONE: analisi dell'ordine fallita (exit $PY_RC); vedi $OUT/tabella.txt"
    TECH_FAIL=$((TECH_FAIL + 1))
fi

# ------------------------------- riepilogo ---------------------------------
say ""
say "Sessione:  $OUT"
say "Tabella:   $OUT/tabella.txt"
say "Ordine:    $OUT/ordine.txt"
say "Log:       $SCRIPT_LOG"
say "Ambiente:  $OUT/ambiente.txt"
if [ "$TECH_FAIL" -gt 0 ]; then
    say "Esito tecnico: $TECH_FAIL riga/righe CFL incomplete (vedi note in tabella)."
    exit 1
fi
say "Esito tecnico: tutte le righe CFL complete."
exit 0
