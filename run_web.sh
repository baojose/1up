#!/bin/bash
# 1UP Web Application Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run ./setup_venv.sh first"
    exit 1
fi

# Use venv Python directly (more reliable after Mac update)
VENV_PYTHON="venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python del venv no encontrado"
    echo "   Ejecuta primero: bash setup_venv.sh"
    exit 1
fi

# Check if Flask is installed
if ! "$VENV_PYTHON" -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    "$VENV_PYTHON" -m pip install Flask>=3.0.0
fi

# Run web application
echo "🌐 Starting 1UP Web Application..."
echo "📍 Server will be available at: http://localhost:5001"
echo "   (Port 5000 is often used by AirPlay on macOS)"
echo "Press Ctrl+C to stop"
echo ""

"$VENV_PYTHON" web_app.py

