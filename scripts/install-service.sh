#!/usr/bin/env bash
# Install a per-user LaunchAgent so the qsm console is always running and
# http://qsm.localhost:8000 works whenever you type it.
#
# User-scope only: ~/Library/LaunchAgents, no sudo, no root, nothing
# system-wide. Undo at any time with scripts/uninstall-service.sh.
set -euo pipefail

PORT="${QSM_PORT:-8000}"
LABEL="com.qsm.console"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.venv/bin/qsm"
AGENTS="$HOME/Library/LaunchAgents"
PLIST="$AGENTS/$LABEL.plist"
LOGS="$ROOT/logs"

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found. Create the venv first:" >&2
  echo "  cd $ROOT && uv venv --python 3.12 && uv pip install -e '.[dev,web]'" >&2
  exit 1
fi

mkdir -p "$AGENTS" "$LOGS"

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN</string>
    <string>serve</string>
    <string>--port</string>
    <string>$PORT</string>
    <string>--no-open</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$LOGS/console.log</string>
  <key>StandardErrorPath</key>
  <string>$LOGS/console.err.log</string>
</dict>
</plist>
PLISTEOF

# Replace any previous copy, then start it. bootout returns before the job is
# actually gone, so bootstrap can race it and fail with "Input/output error";
# wait for the label to disappear before re-bootstrapping.
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
for _ in $(seq 1 20); do
  launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1 || break
  sleep 0.25
done

if ! launchctl bootstrap "gui/$UID" "$PLIST" 2>/dev/null; then
  # Already loaded (or a stale handle): reload the plist and restart in place.
  launchctl enable "gui/$UID/$LABEL" 2>/dev/null || true
fi
launchctl kickstart -k "gui/$UID/$LABEL" >/dev/null 2>&1 || true

printf 'waiting for the console to come up'
for _ in $(seq 1 25); do
  if curl -fs -o /dev/null "http://127.0.0.1:$PORT/"; then
    echo
    echo
    echo "  qsm console is live, and starts automatically at login:"
    echo
    echo "      http://localhost:$PORT"
    echo
    echo "  logs:      $LOGS/console.log"
    echo "  uninstall: $ROOT/scripts/uninstall-service.sh"
    echo
    exit 0
  fi
  printf '.'
  sleep 0.6
done

echo
echo "error: the service did not answer on port $PORT." >&2
echo "check $LOGS/console.err.log" >&2
exit 1
