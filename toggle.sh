#!/bin/zsh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON="$DIR/env/bin/python"
SCRIPT="$DIR/Send_Receive.py"

# Search for the running process
PID=$(pgrep -f "$SCRIPT")

if [ -z "$PID" ]; then
    # Start if not running
    echo "🚀 Starting AirShare..."
    
    # Wayland environment variables
    export QT_QPA_PLATFORM=wayland
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
    
    # Launch in background
    nohup "$PYTHON" "$SCRIPT" > /tmp/airshare.log 2>&1 &
    
    notify-send "AirShare" "Service Started" --icon=camera-web
else
    # Stop if already running
    echo "🛑 Stopping AirShare (PID: $PID)..."
    
    # Kill the process
    kill -15 "$PID"
    
    notify-send "AirShare" "Service Stopped" --icon=camera-off
fi