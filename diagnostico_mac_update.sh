#!/bin/bash
# Script de diagnóstico después de actualización de macOS

echo "🔍 Diagnóstico 1UP después de actualización de Mac"
echo "==================================================="
echo ""

echo "1. Verificando Python del sistema..."
python3 --version
python3.12 --version 2>/dev/null || echo "   ⚠️  python3.12 no encontrado en PATH"
echo ""

echo "2. Verificando entorno virtual..."
if [ -d "venv" ]; then
    echo "   ✅ venv existe"
    echo "   Versión configurada:"
    cat venv/pyvenv.cfg | grep version
    echo "   Ejecutable configurado:"
    cat venv/pyvenv.cfg | grep executable
else
    echo "   ❌ venv NO existe"
fi
echo ""

echo "3. Probando ejecutar Python del venv..."
if [ -f "venv/bin/python3" ]; then
    venv/bin/python3 --version 2>&1 || echo "   ❌ Python del venv NO funciona"
else
    echo "   ❌ python3 NO existe en venv/bin/"
fi
echo ""

echo "4. Verificando dependencias críticas..."
source venv/bin/activate 2>/dev/null || echo "   ⚠️  No se pudo activar venv"

echo "   - torch:"
python -c "import torch; print(f'      ✅ torch {torch.__version__}')" 2>&1 || echo "      ❌ torch NO disponible"

echo "   - cv2:"
python -c "import cv2; print(f'      ✅ opencv-python {cv2.__version__}')" 2>&1 || echo "      ❌ cv2 NO disponible"

echo "   - anthropic:"
python -c "import anthropic; print(f'      ✅ anthropic {anthropic.__version__}')" 2>&1 || echo "      ❌ anthropic NO disponible"

echo "   - SAM 3:"
python -c "from sam3 import SAM3ImagePredictor; print('      ✅ SAM 3 disponible')" 2>&1 || echo "      ❌ SAM 3 NO disponible"

echo ""
echo "5. Verificando rutas de Homebrew..."
if [ -d "/opt/homebrew" ]; then
    echo "   ✅ Homebrew está en /opt/homebrew (Apple Silicon)"
elif [ -d "/usr/local" ]; then
    echo "   ✅ Homebrew está en /usr/local (Intel)"
else
    echo "   ⚠️  Homebrew no encontrado en ubicaciones estándar"
fi
echo ""

echo "6. Verificando acceso a HuggingFace..."
if command -v hf &> /dev/null; then
    echo "   ✅ Comando 'hf' disponible"
    hf auth whoami 2>&1 | head -1 || echo "   ⚠️  No autenticado en HuggingFace"
else
    echo "   ⚠️  Comando 'hf' no disponible (instalar: pip install huggingface-hub)"
fi
echo ""

echo "==================================================="
echo "📋 Resumen:"
echo ""
echo "Si ves ❌ en alguna dependencia, el venv necesita ser recreado."
echo "Ejecuta: bash setup_venv.sh (y responde 'y' para recrear)"
echo ""

