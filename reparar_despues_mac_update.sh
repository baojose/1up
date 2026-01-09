#!/bin/bash
# Script para reparar el proyecto después de actualización de macOS

set -e

echo "🔧 Reparando 1UP después de actualización de macOS"
echo "=================================================="
echo ""

cd "$(dirname "$0")"

# 1. Verificar Python 3.12
echo "1. Verificando Python 3.12..."
if command -v python3.12 &> /dev/null; then
    echo "   ✅ Python 3.12 encontrado: $(python3.12 --version)"
else
    echo "   ⚠️  Python 3.12 no encontrado. Instalando..."
    brew install python@3.12
fi
echo ""

# 2. Verificar/Recrear venv si es necesario
echo "2. Verificando entorno virtual..."
if [ ! -d "venv" ]; then
    echo "   ⚠️  venv no existe. Creando nuevo..."
    python3.12 -m venv venv
    echo "   ✅ venv creado"
else
    echo "   ✅ venv existe"
    
    # Verificar que el Python del venv funciona
    if ! venv/bin/python3 --version &> /dev/null; then
        echo "   ⚠️  venv parece roto. Recreando..."
        rm -rf venv
        python3.12 -m venv venv
        echo "   ✅ venv recreado"
    fi
fi
echo ""

# 3. Activar venv
echo "3. Activando entorno virtual..."
source venv/bin/activate
echo "   ✅ venv activado"
echo ""

# 4. Actualizar pip
echo "4. Actualizando pip..."
pip install --upgrade pip --quiet
echo "   ✅ pip actualizado"
echo ""

# 5. Reinstalar dependencias básicas
echo "5. Reinstalando dependencias básicas..."
echo "   - numpy..."
pip install "numpy>=2.3.0" --only-binary=numpy --quiet
echo "   - torch y torchvision..."
pip install torch torchvision --quiet
echo "   - opencv-python, anthropic, pyyaml, pillow..."
pip install opencv-python anthropic pyyaml pillow --quiet
echo "   ✅ Dependencias básicas instaladas"
echo ""

# 6. Verificar/Reinstalar SAM 3
echo "6. Verificando SAM 3..."
if python -c "from sam3.model_builder import build_sam3_image_model" 2>/dev/null; then
    echo "   ✅ SAM 3 ya está instalado correctamente"
else
    echo "   ⚠️  SAM 3 no está instalado o está roto"
    
    if [ ! -d "sam3" ]; then
        echo "   ⚠️  Directorio sam3 no existe. Clonando repositorio..."
        cd ..
        if [ ! -d "sam3" ]; then
            git clone https://github.com/facebookresearch/sam3.git
        fi
        cd sam3
        echo "   Instalando SAM 3..."
        pip install -e . --quiet
        cd ../1UP_2
        echo "   ✅ SAM 3 instalado"
    else
        echo "   Directorio sam3 existe. Reinstalando..."
        cd sam3
        pip install -e . --quiet
        cd ..
        echo "   ✅ SAM 3 reinstalado"
    fi
fi
echo ""

# 7. Verificar instalación
echo "7. Verificando instalación completa..."
echo "   - torch:"
python -c "import torch; print(f'      ✅ torch {torch.__version__}')" || echo "      ❌ FALLO"

echo "   - cv2:"
python -c "import cv2; print(f'      ✅ opencv-python {cv2.__version__}')" || echo "      ❌ FALLO"

echo "   - anthropic:"
python -c "import anthropic; print(f'      ✅ anthropic {anthropic.__version__}')" || echo "      ❌ FALLO"

echo "   - SAM 3:"
python -c "from sam3.model_builder import build_sam3_image_model; print('      ✅ SAM 3 disponible')" || echo "      ❌ FALLO - Necesitas acceso a checkpoints de HuggingFace"

echo ""

# 8. Verificar HuggingFace
echo "8. Verificando HuggingFace..."
if command -v hf &> /dev/null; then
    if hf auth whoami &> /dev/null; then
        echo "   ✅ Autenticado en HuggingFace"
    else
        echo "   ⚠️  No autenticado en HuggingFace"
        echo "   Ejecuta: hf auth login"
    fi
else
    echo "   ⚠️  Cliente HuggingFace no instalado"
    echo "   Instalando huggingface-hub..."
    pip install huggingface-hub --quiet
    echo "   ✅ Instalado. Ejecuta: hf auth login"
fi
echo ""

echo "=================================================="
echo "✅ Reparación completa!"
echo ""
echo "Si todo está ✅, el proyecto debería funcionar ahora."
echo ""
echo "Prueba ejecutando:"
echo "  ./run_test_detection.sh"
echo ""

