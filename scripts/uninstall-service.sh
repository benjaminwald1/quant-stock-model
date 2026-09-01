#!/usr/bin/env bash
# Remove the qsm console LaunchAgent. Leaves the project, its venv and its runs
# completely untouched -- this only stops the console starting at login.
set -euo pipefail
LABEL="com.qsm.console"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
if [ -f "$PLIST" ]; then
  rm -f "$PLIST"
  echo "removed $PLIST"
else
  echo "no LaunchAgent installed at $PLIST"
fi
echo "the console no longer starts at login."
