#!/bin/bash
# Activate venv and run list_cameras.py

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "❌ Entorno virtual no encontrado"
    echo "   Ejecuta primero: bash setup_venv.sh"
    exit 1
fi

# Use venv Python directly (more reliable after Mac update)
VENV_PYTHON="venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python del venv no encontrado"
    echo "   Ejecuta primero: bash setup_venv.sh"
    exit 1
fi

"$VENV_PYTHON" list_cameras.py

