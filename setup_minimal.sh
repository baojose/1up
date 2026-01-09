#!/bin/bash
# 1UP - Setup Mínimo (solo para probar cámaras)

set -e

echo "🍄 1UP - Setup Mínimo (solo cámaras)"
echo "====================================="
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

# Create virtual environment
echo "1. Creando entorno virtual..."
python3 -m venv venv
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

# Install ONLY opencv for camera testing
echo ""
echo "4. Instalando OpenCV (solo para probar cámaras)..."
pip install opencv-python
echo "✅ OpenCV instalado"

echo ""
echo "🎉 Setup mínimo completo!"
echo ""
echo "Ahora puedes probar las cámaras:"
echo "  ./run_list_cameras.sh"
echo ""
echo "Para instalar el resto (SAM 3, etc.), necesitarás Python 3.11 o 3.12"
echo "Python 3.14 tiene problemas de compatibilidad con numpy/torch"
echo ""

