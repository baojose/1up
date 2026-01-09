#!/bin/bash
# 1UP - Setup with Virtual Environment

set -e

echo "🍄 1UP - Setup con Entorno Virtual"
echo "===================================="
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Entorno virtual ya existe"
    read -p "¿Recrear? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Eliminando entorno virtual anterior..."
        rm -rf venv
    else
        echo "Usando entorno virtual existente"
        source venv/bin/activate
        echo "✅ Entorno virtual activado"
        exit 0
    fi
fi

# Check Python version and use 3.12 if available
echo "1. Verificando Python..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "   ✅ Python 3.12 encontrado"
elif python3 --version | grep -q "3.12"; then
    PYTHON_CMD="python3"
    echo "   ✅ Python 3.12 (python3)"
else
    echo "   ⚠️  Python 3.12 no encontrado"
    echo "   Instalando Python 3.12..."
    brew install python@3.12
    PYTHON_CMD="python3.12"
fi

# Create virtual environment
echo "   Creando entorno virtual con $PYTHON_CMD..."
$PYTHON_CMD -m venv venv
echo "✅ Entorno virtual creado"

# Activate virtual environment
echo ""
echo "2. Activando entorno virtual..."
source venv/bin/activate
echo "✅ Entorno virtual activado"

# Upgrade pip
echo ""
echo "3. Actualizando pip..."
pip install --upgrade pip
echo "✅ pip actualizado"

# Install basic dependencies
echo ""
echo "4. Instalando dependencias básicas..."
echo "   (Instalando numpy precompilado primero)..."
pip install "numpy>=2.3.0" --only-binary=numpy
echo "   ✅ numpy instalado"
echo "   Instalando torch y torchvision..."
pip install torch torchvision --no-deps || pip install torch torchvision
echo "   Instalando resto de dependencias..."
pip install opencv-python anthropic pyyaml pillow
echo "✅ Dependencias básicas instaladas"

# Install SAM 3
echo ""
echo "5. Instalando SAM 3..."
echo "   ⚠️  IMPORTANTE: SAM 3 requiere acceso a checkpoints en HuggingFace"
echo "   1. Ve a: https://huggingface.co/models?search=sam3"
echo "   2. Solicita acceso a los checkpoints"
echo "   3. Autentícate: hf auth login"
echo ""
read -p "¿Ya tienes acceso a SAM 3 checkpoints? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "   ⚠️  Instalando SAM 3 sin checkpoints (necesitarás acceso después)"
fi

if [ ! -d "../sam3" ] && [ ! -d "sam3" ]; then
    echo "   Clonando repositorio SAM 3..."
    cd ..
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    pip install -e .
    cd ../1UP_2
    echo "✅ SAM 3 instalado"
else
    echo "   ✅ SAM 3 ya está clonado"
    if [ -d "../sam3" ]; then
        cd ../sam3 && pip install -e . && cd ../1UP_2
    elif [ -d "sam3" ]; then
        cd sam3 && pip install -e . && cd ..
    fi
fi

# Check API key
echo ""
if [ -z "$CLAUDE_API_KEY" ]; then
    echo "⚠️  ADVERTENCIA: CLAUDE_API_KEY no está configurada"
    echo "   Configúrala con: export CLAUDE_API_KEY='sk-ant-api03-xxxxx'"
else
    echo "7. ✅ CLAUDE_API_KEY está configurada"
fi

echo ""
echo "🎉 Setup completo!"
echo ""
echo "Para usar el proyecto:"
echo "  source venv/bin/activate"
echo ""
echo "O ejecuta scripts con:"
echo "  ./run_list_cameras.sh"
echo "  ./run_test_detection.sh"
echo "  ./run_main.sh"
echo ""

