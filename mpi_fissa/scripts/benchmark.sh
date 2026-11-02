#!/bin/bash
# benchmark.sh -- campagna di misure np=1 (1 GPU) vs np=2 (2 PC) per mpi_fissa.
#
# Fa, per ogni -poisson_nit di NIT_LIST e per REPS ripetizioni:
#   * un run a 1 GPU (np=1, locale)      -> out/perf_2pc/np1_nit<N>_r<R>/
#   * un run a 2 PC  (np=2, hostfile)    -> out/perf_2pc/np2_nit<N>_r<R>/
# e per ognuno salva: log, timing.dat, diagnostics.dat, campionamento
# nvidia-smi di TUTTI gli host, piu' una tabella finale:
#
#   out/perf_2pc/tabella.txt   <- leggibile
#   out/perf_2pc/dati.csv      <- una riga per run (per grafici/script)
#   out/perf_2pc/note.txt      <- ambiente (versioni, GPU, md5, hostfile)
#
# Scelte che contano (leggere prima di cambiare i default):
#   * `-diag 1e9`: nessuna diagnostica a meta' run. La diagnostica costa
#     `download_u` (3x137 MB per rank a L8) + un loop host single-thread su
#     ~34M celle: dentro una media di pochi passi falsa il risultato del 4-20%.
#   * l'ULTIMO passo di ogni run contiene comunque la diagnostica di fine run
#     (residual_lvl + bc_u + download_u): viene SCARTATO se ci sono >=3 passi.
#   * `comm_MB` e `n_exch` in timing.dat sono CUMULATIVI: qui sono differenziati.
#   * `timeout`: se le porte dati MPI sono chiuse dal firewall il run resta
#     appeso per sempre; con TIMEOUT=300 si annota rc=124 e si va avanti.
#   * il dump finale va su /dev/null (KEEP_DUMP=1 per tenerlo: a L8 sono 268 MB
#     per run, e il MPI_Gather costa 1-3 s a run). Non entra nel timing.
#   * TMAX e' in unita' di tempo: a L8 (dt~0.0204) 0.1 = 5 passi; a L6
#     (dt~0.082) servono TMAX=0.5 per avere 5 passi.
#
# Uso:
#   cd ~/progetto/mpi_fissa
#   NX=256 ./scripts/benchmark.sh                       # L8 completo (default)
#   FAST=1 ./scripts/benchmark.sh                        # 1 passo, 1 reps, nit 1-2
#   NIT_LIST="1 2" REPS=3 ./scripts/benchmark.sh
#   SKIP_2PC=1 ./scripts/benchmark.sh                    # solo 1 GPU
#   KEEP_DUMP=1 NX=256 ./scripts/benchmark.sh            # tiene i campi finali
#   HOSTFILE=scripts/hostfile_2pc.txt ./scripts/benchmark.sh
#
# Variabili: BIN HOSTFILE TMAX NIT_LIST REPS NP2 OUTROOT TIMEOUT CUDA
#            MPI_EXTRA SKIP_1GPU SKIP_2PC FAST KEEP_DUMP NX(etichetta)
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIN="${BIN:-bin/mpi_fissa_gpu}"
HOSTFILE="${HOSTFILE:-$ROOT/scripts/hostfile_2pc.txt}"
NIT_LIST="${NIT_LIST:-1 2 3}"
REPS="${REPS:-3}"
NP2="${NP2:-2}"
OUTROOT="${OUTROOT:-$ROOT/out/perf_2pc}"
TIMEOUT="${TIMEOUT:-300}"
CUDA="${CUDA:-$HOME/cuda-13.1}"
MPI_EXTRA="${MPI_EXTRA:-}"
SKIP_1GPU="${SKIP_1GPU:-0}"
SKIP_2PC="${SKIP_2PC:-0}"
FAST="${FAST:-0}"
KEEP_DUMP="${KEEP_DUMP:-0}"   # 1 = tiene omega3d_matrix.bin (268 MB/run a L8!)
if [ "$FAST" = "1" ]; then
  TMAX="${TMAX:-0.02}"; REPS="${REPS:-1}"; NIT_LIST="${NIT_LIST:-1 2}"
else
  TMAX="${TMAX:-0.1}"      # a L8: 5 passi (4 utili scartando l'ultimo)
fi

say() { printf '%-30s %s\n' "$1" "$2"; }
die() { printf '\nERRORE: %s\n' "$*" >&2; exit 2; }
hr()  { printf '%s\n' "================================================================================"; }

# ------------------------------------------------------------------ preflight
[ -x "$BIN" ] || die "manca $BIN (lancia prima ./scripts/build.sh gpu)"
if [ "$SKIP_2PC" = "0" ]; then
  [ -f "$HOSTFILE" ] || die "manca l'hostfile $HOSTFILE (o usa SKIP_2PC=1)"
fi
mkdir -p "$OUTROOT" || die "non riesco a creare $OUTROOT"
CSV="$OUTROOT/dati.csv"
NOTE="$OUTROOT/note.txt"
TABLE="$OUTROOT/tabella.txt"
: > "$CSV"
echo "mode,nit,rep,rc,steps,wall_med,wall_mean,comm_med,comm_mean,comm_pct,mb_step,exch_step,gpu_util_m,gpu_mem_m,gpu_util_s,gpu_mem_s" > "$CSV"

export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

HOSTS=()
if [ -f "$HOSTFILE" ]; then
  mapfile -t HOSTS < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTFILE" | awk '{print $1}')
fi
LOCAL_ALIASES="localhost 127.0.0.1 $(hostname) $(hostname -s 2>/dev/null)"
REMOTES=()
for h in "${HOSTS[@]:-}"; do
  [ -z "$h" ] && continue
  case " $LOCAL_ALIASES " in *" $h "*) : ;; *) REMOTES+=("$h") ;; esac
done

# ------------------------------------------------------------------ ambiente
{
  echo "# note.txt — ambiente della campagna"
  echo "data            : $(date)"
  echo "root            : $ROOT"
  echo "binario         : $BIN"
  echo "md5             : $(md5sum "$BIN" | awk '{print $1}')"
  echo "mpirun          : $(mpirun --version 2>/dev/null | head -1)"
  echo "CUDA            : $CUDA"
  echo "OMP_NUM_THREADS : $OMP_NUM_THREADS"
  echo "TMAX            : $TMAX   (-diag 1e9)"
  echo "NIT_LIST        : $NIT_LIST"
  echo "REPS            : $REPS"
  echo "TIMEOUT         : $TIMEOUT s"
  echo "MPI_EXTRA       : ${MPI_EXTRA:-<vuoto>}"
  echo "hostfile        : $HOSTFILE"
  sed -e 's/^/  | /' "$HOSTFILE" 2>/dev/null
  echo
  echo "GPU master:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  for h in "${REMOTES[@]:-}"; do
    [ -z "$h" ] && continue
    echo "GPU $h:"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
      'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' 2>/dev/null | sed 's/^/  /'
    echo "  OpenMPI : $(ssh -o BatchMode=yes "$h" 'mpirun --version 2>/dev/null | head -1' 2>/dev/null)"
    echo "  md5 bin : $(ssh -o BatchMode=yes "$h" "md5sum '$ROOT/$BIN' 2>/dev/null" | awk '{print $1}')"
    echo "  HOME    : $(ssh -o BatchMode=yes "$h" 'echo $HOME' 2>/dev/null)"
  done
} > "$NOTE" 2>&1

hr; echo " mpi_fissa — benchmark np=1 vs np=2"; hr
say "data"      "$(date '+%Y-%m-%d %H:%M')"
say "binario"   "$BIN (md5 $(md5sum "$BIN" | awk '{print $1}' | cut -c1-12)…)"
say "hostfile"  "$HOSTFILE"
say "TMAX"      "$TMAX  (REPS=$REPS, nit=$NIT_LIST, timeout=${TIMEOUT}s)"
say "output"    "$OUTROOT"
[ -n "${NX:-}" ] && say "NX" "$NX"
echo

# ------------------------------------------------------------ campionatore GPU
# sample_gpu <local|host> <file> : campiona util/mem ogni 0.5 s per max 120 s
sample_gpu() {
  local where="$1" out="$2"
  if [ "$where" = "local" ]; then
    ( for _ in $(seq 1 240); do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
        sleep 0.5
      done ) >> "$out" 2>/dev/null &
  else
    ( ssh -o BatchMode=yes -o ConnectTimeout=10 "$where" \
        'for _ in $(seq 1 240); do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 0.5; done' \
      ) >> "$out" 2>/dev/null &
  fi
  echo $!
}

gpu_agg() {   # <file> -> "maxutil maxmem_GiB"
  if [ ! -s "$1" ]; then echo "0 0.00"; return; fi
  awk -F',' 'NF>=2 { u=$1+0; m=$2+0; if (u>mu) mu=u; if (m>mm) mm=m }
             END { printf "%d %.2f", mu, mm/1024 }' "$1" 2>/dev/null
}

parse_timing() {   # <timing.dat> -> "n wall_med wall_mean comm_med comm_mean pct mb ex"
  awk '
    NR>1 { r++; W[r]=$4; C[r]=$5; M[r]=$6; E[r]=$7 }
    END {
      if (r==0) { print "0 0 0 0 0 0 0 0"; exit }
      lo=1; hi=r
      if (r>=3) hi=r-1                     # scarta l ultimo (diagnostica di fine run)
      n=hi-lo+1
      for (i=1;i<=n;i++) { tw[i]=W[lo+i-1]; tc[i]=C[lo+i-1] }
      for (i=2;i<=n;i++) { k=tw[i]; j=i-1; while (j>=1 && tw[j]>k) { tw[j+1]=tw[j]; j-- } tw[j+1]=k }
      for (i=2;i<=n;i++) { k=tc[i]; j=i-1; while (j>=1 && tc[j]>k) { tc[j+1]=tc[j]; j-- } tc[j+1]=k }
      if (n%2) { wm=tw[(n+1)/2]; cm=tc[(n+1)/2] }
      else     { wm=0.5*(tw[n/2]+tw[n/2+1]); cm=0.5*(tc[n/2]+tc[n/2+1]) }
      ws=0; cs=0
      for (i=lo;i<=hi;i++) { ws+=W[i]; cs+=C[i] }
      base = (lo>1) ? M[lo-1] : 0
      eb   = (lo>1) ? E[lo-1] : 0
      printf "%d %.4f %.4f %.4f %.4f %.1f %.2f %.1f\n",
             n, wm, ws/n, cm, cs/n, (ws>0 ? 100*cs/ws : 0), (M[hi]-base)/n, (E[hi]-eb)/n
    }' "$1" 2>/dev/null
}

last_row() { awk '!/^#/ && NF>5 { row=$0 } END { print row }' "$1" 2>/dev/null; }

cmp_rows() {   # <riga A> <riga B> -> max differenza relativa su enstrophy..separation
  awk -v a="$1" -v b="$2" 'BEGIN{
    na=split(a,A," "); nb=split(b,B," "); m=0
    for (i=4; i<=9 && i<=na && i<=nb; i++) {
      x=A[i]+0; y=B[i]+0
      ax=(x<0?-x:x); ay=(y<0?-y:y)
      s=(ax>ay?ax:ay); if (s<1e-30) s=1e-30
      d=(x>y?x-y:y-x); r=d/s
      if (r>m) m=r
    }
    printf "%.2e", m }'
}

# ------------------------------------------------------------------- un run
# run_one <np1|np2> <nit> <rep>
run_one() {
  local mode="$1" nit="$2" rep="$3"
  local dir="$OUTROOT/${mode}_nit${nit}_r${rep}"
  local dump="/dev/null"
  [ "$KEEP_DUMP" = "1" ] && dump="$dir/omega3d_matrix.bin"
  local args=(-tmax "$TMAX" -diag 1e9 -poisson_nit "$nit" -dump "$dump" -out "$dir")
  local p_m p_s util_m mem_m util_s mem_s rc a b

  rm -rf "$dir"; mkdir -p "$dir"
  p_m=$(sample_gpu local "$dir/gpu_master.csv")
  if [ "$mode" = "np2" ] && [ "${#REMOTES[@]}" -gt 0 ]; then
    p_s=$(sample_gpu "${REMOTES[0]}" "$dir/gpu_slave.csv")
  else
    p_s=""
  fi

  printf '  -> %-3s nit=%s rep=%s ... ' "$mode" "$nit" "$rep"
  if [ "$mode" = "np1" ]; then
    timeout -k 5 "$TIMEOUT" mpirun -np 1 --bind-to none \
      -x LD_LIBRARY_PATH -x OMP_NUM_THREADS \
      "$BIN" "${args[@]}" > "$dir/log.txt" 2>&1
    rc=$?
  elif [ -x scripts/run_2pc.sh ]; then
    HOSTFILE="$HOSTFILE" LOG="$dir/run_2pc.log" SKIP_CHECK=1 MPI_EXTRA="$MPI_EXTRA" \
      timeout -k 5 "$TIMEOUT" ./scripts/run_2pc.sh "${args[@]}" > "$dir/log.txt" 2>&1
    rc=$?
  else
    # shellcheck disable=SC2086
    timeout -k 5 "$TIMEOUT" mpirun --hostfile "$HOSTFILE" -np "$NP2" \
      --map-by ppr:1:node --bind-to none --wdir "$ROOT" \
      --mca plm_rsh_agent ssh \
      --mca plm_rsh_args "-o BatchMode=yes -o ConnectTimeout=10" \
      -x LD_LIBRARY_PATH -x OMP_NUM_THREADS $MPI_EXTRA \
      "$BIN" "${args[@]}" > "$dir/log.txt" 2>&1
    rc=$?
  fi

  [ -n "$p_m" ] && kill "$p_m" 2>/dev/null
  [ -n "$p_s" ] && kill "$p_s" 2>/dev/null
  sleep 0.2

  read -r util_m mem_m <<< "$(gpu_agg "$dir/gpu_master.csv")"
  read -r util_s mem_s <<< "$(gpu_agg "$dir/gpu_slave.csv")"
  : "${util_m:=0}"; : "${mem_m:=0.00}"; : "${util_s:=0}"; : "${mem_s:=0.00}"

  local n=0 wm=0 wmean=0 cm=0 cmean=0 pct=0 mb=0 ex=0
  if [ -f "$dir/timing.dat" ]; then
    read -r n wm wmean cm cmean pct mb ex <<< "$(parse_timing "$dir/timing.dat")"
  fi
  if [ -f "$dir/diagnostics.dat" ]; then
    last_row "$dir/diagnostics.dat" > "$dir/diag_last.txt"
  fi

  echo "rc=$rc passi=$n wall=${wm}s comm=${cm}s (${pct}%) MB=${mb} scambi=${ex}"
  if [ "$rc" = "124" ]; then
    echo "     TIMEOUT dopo ${TIMEOUT}s. Se e' il run a 2 PC: porte dati MPI chiuse"
    echo "     dal firewall (CHECKLIST_2PC.md §4) oppure ssh/orted sul secondo PC."
  elif [ "$rc" != "0" ]; then
    tail -3 "$dir/log.txt" 2>/dev/null | sed 's/^/     /'
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$mode" "$nit" "$rep" "$rc" "$n" "$wm" "$wmean" "$cm" "$cmean" "$pct" "$mb" "$ex" \
    "$util_m" "$mem_m" "$util_s" "$mem_s" >> "$CSV"
}

# ---------------------------------------------------------------- pre-flight
if [ "$SKIP_2PC" = "0" ]; then
  echo "-- pre-flight 2 PC (un run breve, ~10-30 s: se fallisce si prosegue con la sola 1 GPU)"
  PF="$OUTROOT/_preflight"
  rm -rf "$PF"
  if [ -x scripts/run_2pc.sh ]; then
    HOSTFILE="$HOSTFILE" LOG="$OUTROOT/preflight.log" SKIP_CHECK=0 \
      timeout -k 5 240 ./scripts/run_2pc.sh -tmax 0.02 -diag 1e9 -dump /dev/null -out "$PF" \
      > "$OUTROOT/preflight.log" 2>&1
    PF_RC=$?
  else
    timeout -k 5 240 mpirun --hostfile "$HOSTFILE" -np "$NP2" --map-by ppr:1:node \
      --bind-to none --wdir "$ROOT" "$BIN" -tmax 0.02 -diag 1e9 -dump /dev/null -out "$PF" \
      > "$OUTROOT/preflight.log" 2>&1
    PF_RC=$?
  fi
  grep -E '^(ERRORE|ATTENZIONE)|^  |^-- ' "$OUTROOT/preflight.log" 2>/dev/null | head -30 | sed 's/^/   /'
  if [ "$PF_RC" -ne 0 ]; then
    echo
    echo "   pre-flight FALLITO (rc=$PF_RC): proseguo con la sola 1 GPU."
    echo "   Diagnosi: CHECKLIST_2PC.md — cause tipiche: ssh, orted, versione"
    echo "   OpenMPI, md5 del binario, firewall sulle porte dati."
    SKIP_2PC=1
  else
    echo "   pre-flight OK"
  fi
  echo
fi
rm -rf "$OUTROOT/_preflight"

# ------------------------------------------------------------------ campagna
for rep in $(seq 1 "$REPS"); do
  for nit in $NIT_LIST; do
    [ "$SKIP_1GPU" = "0" ] && run_one np1 "$nit" "$rep"
    [ "$SKIP_2PC"  = "0" ] && run_one np2 "$nit" "$rep"
  done
done 2>&1 | tee "$OUTROOT/campagna.log"
echo

# ------------------------------------------------------------------- tabella
{
  hr
  echo " mpi_fissa — np=1 (1 GPU) vs np=2 (2 PC)   $(date '+%Y-%m-%d %H:%M')"
  hr
  echo " binario : $BIN"
  echo " md5     : $(md5sum "$BIN" | awk '{print $1}')"
  echo " mpirun  : $(mpirun --version 2>/dev/null | head -1)"
  echo " hostfile: $HOSTFILE"
  grep -vE '^[[:space:]]*(#|$)' "$HOSTFILE" 2>/dev/null | sed -e 's/^/           /'
  echo " TMAX    : $TMAX   (-diag 1e9: nessuna diagnostica a meta' run)"
  echo " REPS    : $REPS    nit: $NIT_LIST    timeout: ${TIMEOUT}s"
  if [ "$SKIP_2PC" = "1" ]; then
    echo " NB      : run np=2 NON eseguiti (SKIP_2PC=1 o pre-flight fallito)"
  fi
  echo " nota    : wall/comm = mediana sui passi utili; per ogni run l'ultimo"
  echo "           passo (diagnostica di fine run) e' escluso se i passi sono >=3."
  echo "           comm_MB e n_exch di timing.dat sono cumulativi: qui differenziati."
  hr
  echo
} > "$TABLE"

# --- tabella 1: un riga per run
awk -F',' '
  BEGIN { printf " %-4s %4s %4s %6s %10s %10s %7s %9s %8s %8s\n",
                 "modo","nit","rep","passi","wall_s","comm_s","comm%","MB/passo","scambi","GPU% M/S" }
  $1=="mode" { next }
  { printf " %-4s %4s %4s %6d %10.4f %10.4f %6.1f%% %9.2f %8.0f %8s\n",
           $1, $2, $3, ($5+0), ($6+0), ($8+0), ($10+0), ($11+0), ($12+0),
           (($13+0)"/"($15+0)) }
' "$CSV" >> "$TABLE"

# --- tabella 2: mediana sulle reps + speedup
awk -F',' -v nits="$NIT_LIST" '
  function med(v, n,   i,j,k) {
    if (n<=0) return 0
    for (i=2;i<=n;i++) { k=v[i]; j=i-1; while (j>=1 && v[j]>k) { v[j+1]=v[j]; j-- } v[j+1]=k }
    if (n%2) return v[(n+1)/2]
    return 0.5*(v[n/2]+v[n/2+1])
  }
  $1=="mode" { next }
  {
    key=$1" "$2; k=++cnt[key]
    W[key,k]=$6+0; C[key,k]=$8+0; MB[key,k]=$11+0; EX[key,k]=$12+0
    seen[key]=1
  }
  END {
    printf "\n %-4s %4s %8s %8s %7s %9s %8s %9s\n", \
           "modo","nit","wall_s","comm_s","comm%","MB/passo","scambi","speedup"
    nn=split(nits, N, " ")
    for (mi=1; mi<=2; mi++) {
      mode = (mi==1) ? "np1" : "np2"
      for (ii=1; ii<=nn; ii++) {
        nit = N[ii]; key = mode" "nit
        if (!seen[key]) continue
        n = cnt[key]
        for (i=1;i<=n;i++) { wv[i]=W[key,i]; cv[i]=C[key,i]; mv[i]=MB[key,i]; ev[i]=EX[key,i] }
        w = med(wv, n); c = med(cv, n); mb = med(mv, n); ex = med(ev, n)
        printf " %-4s %4s %8.4f %8.4f %6.1f%% %9.2f %8.0f", \
               mode, nit, w, c, (w>0?100*c/w:0), mb, ex
        if (mode=="np2") {
          k1 = "np1 "nit
          s1 = 0
          if (w>0 && cnt[k1]>0) {
            for (i=1;i<=cnt[k1];i++) tmp[i]=W[k1,i]
            s1 = med(tmp, cnt[k1])/w
          }
          if (s1>0) printf " %8.2fx", s1; else printf " %8s", "-"
        } else {
          printf " %8s", "-"
        }
        printf "\n"
      }
    }
  }
' "$CSV" >> "$TABLE" 2>/dev/null

# --- confronto campi np1 vs np2
{
  echo
  echo " Confronto campi np1 vs np2 (max differenza relativa su enstrophy..separation,"
  echo " ultima riga di diagnostics.dat). Atteso ~1e-6; allarme se > 1e-4."
  any=0
  for nit in $NIT_LIST; do
    worst=0; wrep=0; found=0
    for rep in $(seq 1 "$REPS"); do
      a="$OUTROOT/np1_nit${nit}_r${rep}/diag_last.txt"
      b="$OUTROOT/np2_nit${nit}_r${rep}/diag_last.txt"
      [ -s "$a" ] && [ -s "$b" ] || continue
      found=1; any=1
      r=$(cmp_rows "$(cat "$a")" "$(cat "$b")")
      if awk -v x="$r" -v y="$worst" 'BEGIN{exit !(x+0 > y+0)}'; then worst="$r"; wrep="$rep"; fi
    done
    if [ "$found" = "1" ]; then
      printf '   nit=%s  err_max = %s  (peggiore delle reps: %s)\n' "$nit" "$worst" "$wrep"
    else
      printf '   nit=%s  nessun dato (run mancanti o falliti)\n' "$nit"
    fi
  done
  [ "$any" = "0" ] && echo "   (nessuna coppia np1/np2 completa)"
  echo
  echo " Come leggere la tabella:"
  echo "  * speedup = wall(np1)/wall(np2) con lo stesso nit. A L8 oggi atteso"
  echo "    1.1-1.2x; con le ottimizzazioni O2-O5 di OTIMIZZAZIONI.md 1.4-1.6x."
  echo "  * comm% > 40% = il collo di bottiglia e' la comunicazione"
  echo "    (OTIMIZZAZIONI.md §6-7). Oltre 70%: 2 GPU non conviene per la velocita'."
  echo "  * se MB/passo vale ~7 e non ~17-28, il binario e' a NX=128, non a L8."
  echo "  * GPU% M/S = utilizzazione massima campionata sui due PC: se uno dei due"
  echo "    resta a 0-20%, quel rank aspetta MPI e non calcola."
  echo "  * err_max >> 1e-4 = la decomposizione non riproduce il run a 1 rank:"
  echo "    fermarsi e guardare gli aloni, non la prestazione."
} >> "$TABLE"

echo "== tabella: $TABLE"
echo "== csv    : $CSV"
echo "== note   : $NOTE"
echo
cat "$TABLE"
