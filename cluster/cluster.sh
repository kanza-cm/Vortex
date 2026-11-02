#!/bin/bash
# master -> slaves: push / pull / run / status over ssh.
# hosts.txt: one ssh alias per line (configure them in ~/.ssh/config).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTS_FILE="${HOSTS_FILE:-$HERE/hosts.txt}"
PARALLEL="${PARALLEL:-4}"

hosts() { grep -vE '^[[:space:]]*(#|$)' "$HOSTS_FILE"; }

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
    echo "usage: $(basename "$0") push <local> <remote>"
    echo "       $(basename "$0") pull <remote> <local>"
    echo "       $(basename "$0") run '<remote command>'"
    echo "       $(basename "$0") status"
    exit 1
}

# run one command on every host, in parallel, one line of output per host
each() {
    local cmd="$1" pids=()
    while IFS= read -r h; do
        ( out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "$cmd" 2>&1) \
              && printf 'OK   %-10s %s\n' "$h" "$out" \
              || printf 'FAIL %-10s %s\n' "$h" "$out" ) &
        pids+=($!)
        while [ "${#pids[@]}" -ge "$PARALLEL" ]; do
            wait "${pids[0]}" 2>/dev/null || true
            pids=("${pids[@]:1}")
        done
    done < <(hosts)
    wait
}

cmd="${1:-}"; shift || true
case "$cmd" in
    push)
        src="${1:?local path}"; dst="${2:?remote path}"
        each "mkdir -p '$dst'" >/dev/null
        while IFS= read -r h; do
            ( rsync -az --info=progress2 "$src" "$dst" 2>/dev/null >/dev/null \
                && echo "OK   $h" || echo "FAIL $h (rsync)" ) &
        done < <(hosts)
        wait ;;
    pull)
        src="${1:?remote path}"; dst="${2:?local dir}"
        while IFS= read -r h; do
            mkdir -p "$dst/$h"
            ( rsync -az "$h:$src" "$dst/$h/" \
                && echo "OK   $h -> $dst/$h" || echo "FAIL $h (rsync)" ) &
        done < <(hosts)
        wait ;;
    run)
        [ $# -ge 1 ] || usage
        each "$1" ;;
    status)
        each 'printf "%s  " "$(hostname)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | head -1; uptime | sed "s/^ *//"' ;;
    *) usage ;;
esac
