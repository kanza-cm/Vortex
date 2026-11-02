#!/bin/bash
# Copy the project to every slave as ~/progetto (without .git and results).
#   ./push_project.sh
#   REMOTE_DIR=progetto ./push_project.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
HOSTS_FILE="${HOSTS_FILE:-$HERE/hosts.txt}"
REMOTE_DIR="${REMOTE_DIR:-progetto}"

while IFS= read -r h; do
    h="${h%%[[:space:]]*}"; [ -z "$h" ] && continue
    echo "-> $h:$REMOTE_DIR"
    rsync -az --delete \
        --exclude '.git/' --exclude 'risultati/' --exclude '__pycache__/' \
        --exclude '*.swp' \
        "$SRC/" "$h:$REMOTE_DIR/"
done < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTS_FILE")
echo "fatto"
