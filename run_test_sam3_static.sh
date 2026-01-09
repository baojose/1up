#!/bin/bash
# Test SAM 3 with a static image (no camera required)

cd "$(dirname "$0")"

# Use venv Python directly (more reliable after Mac update)
VENV_PYTHON="venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python del venv no encontrado"
    echo "   Ejecuta primero: bash setup_venv.sh"
    exit 1
fi

echo "🧪 Testing SAM 3 with static image..."
echo ""

"$VENV_PYTHON" test_sam3_static.py

