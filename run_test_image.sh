#!/bin/bash
# Test SAM 3 on a specific image
# Usage: ./run_test_image.sh [image_path]

cd "$(dirname "$0")"

# Use venv Python directly (more reliable after Mac update)
VENV_PYTHON="venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python del venv no encontrado"
    echo "   Ejecuta primero: bash setup_venv.sh"
    exit 1
fi

if [ -z "$1" ]; then
    # Default image
    IMAGE="images/raw/scene_camCAM0_2025-12-01_20-36-47.jpg"
    echo "🧪 Testing SAM 3 on default image: $IMAGE"
else
    IMAGE="$1"
    echo "🧪 Testing SAM 3 on: $IMAGE"
fi

echo ""

"$VENV_PYTHON" test_image.py "$IMAGE"

