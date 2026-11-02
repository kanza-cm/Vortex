#!/bin/bash
# run_2pc.sh -- una SINGOLA simulazione su 2 PC (un rank per PC) via SSH + MPI.
#
# Sostituisce mpi_fissa/scripts/run_2pc.sh. Differenze rispetto alla versione
# precedente (ognuna evita un guasto che altrimenti si scopre a run avviato):
#   * pre-flight: ssh, orted, stesso OpenMPI, stesso $HOME, STESSO binario
#     (md5) e stesso percorso sul secondo PC -> fallisce in 5 s, non a meta';
#   * test `mpirun -np 2 hostname` incluso: si vede subito se il 2o rank parte;
#   * `-mca plm_rsh_args` con BatchMode/ConnectTimeout: niente prompt appesi;
#   * niente `exec` e log su file: il riepilogo si stampa anche se il run muore.
#
# Uso:
#   ./scripts/run_2pc.sh -tmax 1 -out out/2pc_test
#   HOSTFILE=scripts/hostfile_2pc.txt ./scripts/run_2pc.sh -tmax 10 -out out/2pc_T10
#   DRY=1 ./scripts/run_2pc.sh -tmax 1          # stampa solo il comando mpirun
#   SKIP_CHECK=1 ./scripts/run_2pc.sh ...       # salta il pre-flight (sconsigliato)
#
# Variabili d'ambiente:
#   HOSTFILE   hostfile (default scripts/hostfile_2pc.txt)
#   BIN        binario (default bin/mpi_fissa_gpu)
#   CUDA       radice CUDA (default $HOME/cuda-13.1)
#   NP         numero di rank (default 2 = un rank per PC)
#   MPI_EXTRA  opzioni mpirun aggiuntive, es. "--mca btl tcp,self"
#   LOG        file di log (default out/run_2pc.log)
#   SKIP_CHECK=1  salta i controlli
#   DRY=1      stampa il comando e esce
#   ALLOW_MPI_MISMATCH=1  non blocca se le versioni OpenMPI differiscono
set -uo pipefail

cd "$(dirname "$0")/.."
HERE=$(pwd)
HOSTFILE="${HOSTFILE:-$HERE/scripts/hostfile_2pc.txt}"
BIN="${BIN:-bin/mpi_fissa_gpu}"
CUDA="${CUDA:-$HOME/cuda-13.1}"
NP="${NP:-2}"
MPI_EXTRA="${MPI_EXTRA:-}"
LOG="${LOG:-$HERE/out/run_2pc.log}"
SKIP_CHECK="${SKIP_CHECK:-0}"
DRY="${DRY:-0}"

say()  { printf '  %-16s %s\n' "$1" "$2"; }
die()  { printf '\nERRORE: %s\n' "$*" >&2; exit 2; }
warn() { printf 'ATTENZIONE: %s\n' "$*" >&2; }

# ---------------------------------------------------------------- 0. percorso
case "$HERE" in
  *" "*) die "percorso con spazi ($HERE).
  Il progetto deve stare in un percorso senza spazi, IDENTICO sui due PC:
      ~/progetto/mpi_fissa      (consigliato, coerente con cluster/push_project.sh)
  Copia con:  rsync -az --exclude out/ --exclude bin/ ./ slave1:~/progetto/mpi_fissa/" ;;
esac
[ -f "$HOSTFILE" ] || die "manca l'hostfile $HOSTFILE"
[ -x "$BIN" ]      || die "manca o non e' eseguibile $BIN (lancia prima ./scripts/build.sh gpu)"
mkdir -p "$(dirname "$LOG")"

# --------------------------------------------------------------- 1. hostfile
mapfile -t HOSTS < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTFILE" | awk '{print $1}')
[ "${#HOSTS[@]}" -eq "$NP" ] || die "l'hostfile $HOSTFILE ha ${#HOSTS[@]} host, ma -np $NP.
  Servono esattamente $NP righe 'host slots=1' (una per rank)."
LOCAL_ALIASES="localhost 127.0.0.1 $(hostname) $(hostname -s 2>/dev/null)"
case " $LOCAL_ALIASES " in
  *" ${HOSTS[0]} "*) : ;;
  *) die "la prima riga dell'hostfile deve essere la macchina locale
  (localhost o 127.0.0.1); trovato '${HOSTS[0]}'" ;;
esac
REMOTES=()
for h in "${HOSTS[@]}"; do
  case " $LOCAL_ALIASES " in *" $h "*) : ;; *) REMOTES+=("$h") ;; esac
done
[ "${#REMOTES[@]}" -gt 0 ] || warn "nessun host remoto nell'hostfile: sara' un run multi-rank su una sola macchina"

echo "=== run_2pc: $NP rank, $(date) ==="
sed -e 's/^/  | /' "$HOSTFILE"
echo

# ------------------------------------------------------------- 2. pre-flight
if [ "$SKIP_CHECK" -eq 0 ] && [ "$DRY" -eq 0 ]; then
  echo "-- locale"
  LOC_MD5=$(md5sum "$BIN" 2>/dev/null | awk '{print $1}')
  LOC_MPI=$(mpirun --version 2>/dev/null | head -1)
  [ -n "$LOC_MPI" ] || die "mpirun non trovato: sudo apt install openmpi-bin"
  command -v orted >/dev/null || die "orted non e' nel PATH: sudo apt install openmpi-bin"
  say "binario"  "$BIN (md5 ${LOC_MD5:0:12}…)"
  say "mpirun"   "$LOC_MPI"
  say "orted"    "$(command -v orted)"
  [ -d "$CUDA" ] && say "CUDA" "$CUDA" || warn "$CUDA non esiste (ok se CUDA e' in /usr/local/cuda)"
  say "GPU" "$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | head -1)"
  echo

  for h in "${REMOTES[@]}"; do
    echo "-- $h"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" true 2>/dev/null; then
      die "ssh $h non funziona in batch mode.
  Prova a mano:  ssh -o BatchMode=yes $h hostname
  Cause tipiche: chiave non copiata (ssh-copy-id); host key sconosciuta
  (StrictHostKeyChecking); banner/MOTD che rompe il protocollo orted."
    fi
    R_HOME=$(ssh -o BatchMode=yes "$h" 'echo $HOME' 2>/dev/null)
    R_MPI=$(ssh -o BatchMode=yes "$h" 'mpirun --version 2>/dev/null | head -1' 2>/dev/null)
    R_ORTED=$(ssh -o BatchMode=yes "$h" 'command -v orted' 2>/dev/null)
    R_MD5=$(ssh -o BatchMode=yes "$h" "md5sum '$HERE/$BIN' 2>/dev/null" | awk '{print $1}')
    R_GPU=$(ssh -o BatchMode=yes "$h" 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader' 2>/dev/null | head -1)
    say "ssh"      "OK"
    say "\$HOME"   "$R_HOME"
    say "mpirun"   "${R_MPI:-NON TROVATO}"
    say "orted"    "${R_ORTED:-NON TROVATO}"
    say "binario"  "${R_MD5:-MANCANTE} (locale ${LOC_MD5})"
    say "GPU"      "${R_GPU:-nvidia-smi non disponibile}"

    [ -n "$R_HOME" ] || die "$h: ssh non risponde"
    [ "$R_HOME" = "$HOME" ] || die "\$HOME diverso: locale '$HOME', $h '$R_HOME'.
  Con \$HOME diversi il percorso del progetto non puo' essere identico: usare lo
  stesso nome utente sui due PC."
    [ -n "$R_ORTED" ] || die "$h: orted non nel PATH del login shell.
  Su quel PC:  sudo apt install openmpi-bin    (stessa versione del master)"
    if [ "$R_MPI" != "$LOC_MPI" ] && [ "${ALLOW_MPI_MISMATCH:-0}" != "1" ]; then
      die "OpenMPI diverso:
      locale : $LOC_MPI
      $h : $R_MPI
  Devono coincidere (stesso pacchetto openmpi-bin sui due PC). Per forzare:
  ALLOW_MPI_MISMATCH=1 ./scripts/run_2pc.sh ..."
    fi
    [ -n "$R_MD5" ] || die "$h: non trovo $HERE/$BIN.
  Il progetto deve stare NELLO STESSO PERCORSO sui due PC. Copia con:
      rsync -az --exclude out/ --exclude bin/ ./ $h:$HERE/
  poi compila la' con la STESSA NX:  NX=256 ./scripts/build.sh gpu"
    [ "$R_MD5" = "$LOC_MD5" ] || die "binari diversi fra i due PC (md5 $LOC_MD5 vs $R_MD5).
  NX e' un valore di COMPILAZIONE: due binari con NX diversi danno griglie
  diverse e la decomposizione e' sbagliata. Ricompila su entrambi con la stessa
  NX e lo stesso ARCH (se le due GPU sono modelli diversi vedi CHECKLIST §3)."
  done
  echo

  # --------------------------------------------------- 3. test di connettivita'
  echo "-- test MPI end-to-end (2 s)"
  if mpirun --hostfile "$HOSTFILE" -np "$NP" \
        --mca plm_rsh_agent ssh \
        --mca plm_rsh_args "-o BatchMode=yes -o ConnectTimeout=10" \
        hostname; then
    echo "   (devono comparire $NP hostname diversi; se sono uguali OpenMPI ha"
    echo "    messo i rank sulla stessa macchina: controlla slots=1 nell'hostfile)"
  else
    die "mpirun -np $NP hostname FALLITO.
  'ssh: connect ... timed out'        -> rete/firewall (porta 22)
  'orted: command not found'          -> openmpi-bin mancante sul secondo PC
  resta appeso al primo scambio grosso -> firewall sulle porte dati MPI
                                          (CHECKLIST_2PC.md §4)"
  fi
  echo
fi

# ------------------------------------------------------------------- 4. run
export LD_LIBRARY_PATH="$CUDA/lib64:$CUDA/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CMD=(mpirun --hostfile "$HOSTFILE" -np "$NP"
     --map-by ppr:1:node --bind-to none --wdir "$HERE"
     --mca plm_rsh_agent ssh
     --mca plm_rsh_args "-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"
     -x LD_LIBRARY_PATH -x OMP_NUM_THREADS)
if [ -n "$MPI_EXTRA" ]; then
  # shellcheck disable=SC2206
  CMD+=($MPI_EXTRA)
fi
CMD+=("$BIN" "$@")

printf -- '-- comando\n   '
printf '%q ' "${CMD[@]}"; echo; echo
if [ "$DRY" = "1" ]; then echo "(DRY=1: non eseguo)"; exit 0; fi

"${CMD[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

# -------------------------------------------------------------- 5. riepilogo
OUT="out"; prev=""
for a in "$@"; do [ "$prev" = "-out" ] && OUT="$a"; prev="$a"; done
echo
if [ "$RC" -ne 0 ]; then
  echo "=== run FALLITO (exit $RC) — log: $LOG ==="
  echo "  grep -iE 'error|unable|refused|timed out|abort' $LOG | head -20"
  exit "$RC"
fi
if [ -f "$OUT/done_0.txt" ]; then
  echo "=== OK: $(cat "$OUT/done_0.txt") ==="
else
  warn "il run e' uscito con 0 ma manca $OUT/done_0.txt (rank 0 non ha finito?)"
fi
if [ -f "$OUT/timing.dat" ]; then
  echo "passi  wall_s/medio  comm_s/medio  comm%  MB/passo   [da $OUT/timing.dat]"
  awk '
    NR>1 { r++; W[r]=$4; C[r]=$5; M[r]=$6 }
    END {
      if (r==0) { print "  (nessun passo registrato)"; exit }
      lo=1; hi=r
      if (r>=3) hi=r-1            # l ultimo passo contiene la diagnostica di fine run
      w=0; c=0
      for (i=lo;i<=hi;i++) { w+=W[i]; c+=C[i] }
      n=hi-lo+1
      base = (lo>1) ? M[lo-1] : 0
      mb = (M[hi]-base)/n
      printf "%5d  %12.4f  %12.4f  %5.1f%%  %8.2f%s\n", n, w/n, c/n,
             (w>0 ? 100*c/w : 0), mb,
             (r>=3 ? "   (ultimo passo escluso)" : "   (attenzione: pochi passi)")
    }' "$OUT/timing.dat"
  echo
  echo "  comm_MB e n_exch in timing.dat sono CUMULATIVI: qui sono differenziati."
  echo "  Confronto a 1 GPU:  ./scripts/run_gpu.sh 1 $* -out ${OUT}_np1"
fi
echo
echo "log completo: $LOG"
