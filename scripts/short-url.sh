#!/usr/bin/env bash
# Optional: make the console reachable at http://q:8000 instead of
# http://localhost:8000.
#
# This edits /etc/hosts, so it needs sudo — run it yourself:
#     sudo ./scripts/short-url.sh
#
# Undo with:  sudo ./scripts/short-url.sh --remove
set -euo pipefail

NAME="${QSM_SHORT_NAME:-q}"
MARK="# added by qsm"

if [ "$(id -u)" -ne 0 ]; then
  echo "This edits /etc/hosts and must run as root:" >&2
  echo "  sudo $0 $*" >&2
  exit 1
fi

if [ "${1:-}" = "--remove" ]; then
  sed -i '' "/${MARK}\$/d" /etc/hosts
  dscacheutil -flushcache 2>/dev/null || true
  echo "removed. http://${NAME}:8000 no longer resolves."
  exit 0
fi

if grep -q "[[:space:]]${NAME}[[:space:]]*${MARK}\$" /etc/hosts 2>/dev/null; then
  echo "already present."
else
  # Both stacks, matching what the server listens on.
  printf '127.0.0.1\t%s\t%s\n::1\t\t%s\t%s\n' "$NAME" "$MARK" "$NAME" "$MARK" >> /etc/hosts
  dscacheutil -flushcache 2>/dev/null || true
fi

echo
echo "  now reachable at:  http://${NAME}:8000"
echo "  undo with:         sudo $0 --remove"
echo
