#!/bin/bash
# Run a script fully detached from SSH session
# Usage: ./run_detached.sh <log_file> <script> [args...]
LOGFILE="$1"
shift
setsid bash "$@" > "$LOGFILE" 2>&1 < /dev/null &
disown
echo "Detached: PID=$! logfile=$LOGFILE"
