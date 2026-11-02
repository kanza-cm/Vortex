#!/bin/bash
# Run a queue of GPU jobs: hosts work in parallel, one job at a time per host
# (one GPU = one job), so nothing goes OOM and each GPU is always busy.
#
#   ./queue.sh              # start everything, return immediately
#   ./queue.sh --wait       # start and follow until all queues are done
#   QUEUE=cluster/queue.txt ./queue.sh --wait
#
# queue.txt lines:  host|command      ("local" = this PC, the others are ssh aliases)
# Logs: ~/job_logs/queue_<host>.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
QUEUE_FILE="${QUEUE:-$HERE/queue.txt}"
WAIT=0
case "${1:-}" in
    --wait) WAIT=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

declare -A JOBS
while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    host="${line%%|*}"; cmd="${line#*|}"
    [ -n "$host" ] && [ -n "$cmd" ] || { echo "riga non valida: $line"; exit 1; }
    JOBS["$host"]="${JOBS[$host]:-}"$'\n'"$cmd"
done < "$QUEUE_FILE"

mail_logs=""
for host in "${!JOBS[@]}"; do
    script="#!/bin/bash"
    n=0
    while IFS= read -r cmd; do
        [ -z "$cmd" ] && continue
        n=$((n+1))
        script+=$'\n'"echo __JOB${n}__START__"
        script+=$'\n'"$cmd"
        script+=$'\n'"echo __JOB${n}__DONE__\$?"
    done <<< "${JOBS[$host]}"
    script+=$'\n'"echo __QUEUE_DONE__"
    enc=$(printf '%s' "$script" | base64 -w0)
    log="job_logs/queue_$host.log"
    mail_logs+=" $host"
    echo "-> $host : $n job in coda"
    if [ "$host" = "local" ]; then
        mkdir -p "$HOME/job_logs"
        printf '%s\n' "$script" > "$HOME/job_logs/queue_local.sh"
        chmod +x "$HOME/job_logs/queue_local.sh"
        ( nohup bash "$HOME/job_logs/queue_local.sh" > "$HOME/job_logs/queue_local.log" 2>&1 & echo $! > /dev/null )
    else
        ssh -o BatchMode=yes "$host" "mkdir -p \$HOME/job_logs && echo $enc | base64 -d > \$HOME/job_logs/queue_$host.sh && chmod +x \$HOME/job_logs/queue_$host.sh && nohup bash \$HOME/job_logs/queue_$host.sh > \$HOME/job_logs/queue_$host.log 2>&1 &" >/dev/null
    fi
done

echo
echo "code avviate. Log:"
for host in "${!JOBS[@]}"; do
    if [ "$host" = "local" ]; then echo "  local   $HOME/job_logs/queue_local.log"
    else echo "  $host   ~/job_logs/queue_$host.log"; fi
done

if [ "$WAIT" -eq 1 ]; then
    echo
    echo "=== seguo le code (Ctrl-C per smettere di seguire) ==="
    running=1
    while [ "$running" -gt 0 ]; do
        sleep 15; running=0
        for host in "${!JOBS[@]}"; do
            if [ "$host" = "local" ]; then
                logf="$HOME/job_logs/queue_local.log"
                last=$(tail -n 1 "$logf" 2>/dev/null || true)
            else
                last=$(ssh -o BatchMode=yes "$host" "tail -n 1 ~/job_logs/queue_$host.log 2>/dev/null" || true)
            fi
            case "$last" in
                *__QUEUE_DONE__*) printf '%-10s coda finita\n' "$host" ;;
                *)                printf '%-10s %s\n' "$host" "$last"; running=$((running+1)) ;;
            esac
        done
    done
    echo "tutte le code sono terminate"
fi
