#!/bin/bash
# Launch the GPU jobs in jobs.txt on the slaves (in background) and follow them.
#
#   ./dispatch.sh                 # start all jobs, return immediately
#   ./dispatch.sh --wait          # start and wait, printing the last log line
#   JOBS=cluster/jobs.txt ./dispatch.sh --wait
#
# Each job line:  host|remote command
# Logs end up on the slave in  ~/job_logs/<host>-<n>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS_FILE="${JOBS:-$HERE/jobs.txt}"
WAIT=0
case "${1:-}" in
    --wait) WAIT=1 ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

mapfile -t JOBS < <(grep -vE '^[[:space:]]*(#|$)' "$JOBS_FILE")
[ "${#JOBS[@]}" -gt 0 ] || { echo "nessun job in $JOBS_FILE"; exit 1; }

declare -A LOGOF
i=0
for line in "${JOBS[@]}"; do
    host="${line%%|*}"; cmd="${line#*|}"
    [ -n "$host" ] && [ -n "$cmd" ] || { echo "riga non valida: $line"; exit 1; }
    i=$((i+1)); log="job_logs/${host}-${i}.log"; LOGOF["$host"]="$log"
    enc=$(printf '%s' "$cmd" | base64 -w0)
    wrapped="bash -lc \"\$(echo $enc | base64 -d)\"; echo __DONE__\$?"
    enc2=$(printf '%s' "$wrapped" | base64 -w0)
    echo "-> $host : $cmd"
    ssh -o BatchMode=yes "$host" "mkdir -p \$HOME/job_logs && nohup bash -lc \"\$(echo $enc2 | base64 -d)\" > \$HOME/$log 2>&1 & echo \$!" >/dev/null
done
echo
echo "job avviati. Log: ~/job_logs/*.log sui slave"
echo "stato:   ./cluster.sh run 'tail -n 5 ~/job_logs/*.log'"
echo "oppure:  ./dispatch.sh --wait"

if [ "$WAIT" -eq 1 ]; then
    echo
    echo "=== attendo la fine (Ctrl-C per smettere di seguire) ==="
    remaining=1
    while [ "$remaining" -gt 0 ]; do
        sleep 15; remaining=0
        for host in "${!LOGOF[@]}"; do
            log="${LOGOF[$host]}"
            last=$(ssh -o BatchMode=yes "$host" "tail -n 1 ~/$log 2>/dev/null" || true)
            case "$last" in
                __DONE__*) printf '%-10s fatto (%s)\n' "$host" "${last#__DONE__}" ;;
                *)         printf '%-10s %s\n' "$host" "$last"; remaining=$((remaining+1)) ;;
            esac
        done
    done
    echo "tutti i job sono terminati"
fi
