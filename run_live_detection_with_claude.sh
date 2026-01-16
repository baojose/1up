#!/bin/bash
# Activate venv and run live_detection.py with Claude API key

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

# Try to load API key from file first
API_KEY_FILE=".claude_api_key"
if [ -f "$API_KEY_FILE" ]; then
    export CLAUDE_API_KEY="$(cat "$API_KEY_FILE" | tr -d '\n\r ')"
    echo "✅ API key cargada desde $API_KEY_FILE"
fi

# Load HuggingFace token from file or environment
HF_TOKEN_FILE=".hf_token"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE" | tr -d '\n\r ')"
    echo "✅ HF token cargado desde $HF_TOKEN_FILE"
elif [ -n "$HF_TOKEN" ]; then
    # Token already set in environment (e.g., from ~/.zshrc)
    echo "✅ HF token cargado desde variable de entorno"
else
    echo "⚠️  HF_TOKEN no encontrado (necesario para SAM 3)"
    echo "   El token puede estar configurado en ~/.zshrc o en .hf_token"
fi

# Check if API key is set (from file or environment)
if [ -z "$CLAUDE_API_KEY" ]; then
    echo "⚠️  CLAUDE_API_KEY no está configurada"
    echo ""
    echo "Ejecuta primero:"
    echo "export CLAUDE_API_KEY='sk-ant-api03-...'"
    echo ""
    read -p "¿Quieres configurarla ahora? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        read -p "Pega tu API key: " api_key
        export CLAUDE_API_KEY="$api_key"
        
        # Save to file for future use
        echo "$api_key" > "$API_KEY_FILE"
        chmod 600 "$API_KEY_FILE"  # Secure: only owner can read/write
        echo "✅ API key configurada y guardada en $API_KEY_FILE"
    else
        echo "⚠️  Continuando sin API key (solo detección visual, sin análisis Claude)"
    fi
fi

echo ""
echo "🧹 Limpiando imágenes anteriores (testing mode)..."
echo ""

# Clean up previous images if in testing mode
"$VENV_PYTHON" << 'CLEANUP_EOF'
import json
from pathlib import Path
import shutil

# Load config to check testing mode
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

if config.get('testing', {}).get('auto_cleanup', False):
    print("   Limpiando imágenes y crops anteriores...")
    
    # Clean database (keep structure)
    db_path = Path('database/objects.json')
    if db_path.exists():
        with open(db_path, 'w') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print("   ✅ Base de datos limpiada")
    
    # Delete all raw images
    raw_dir = Path('images/raw')
    if raw_dir.exists():
        removed = 0
        for file in raw_dir.glob("*"):
            if file.is_file():
                file.unlink()
                removed += 1
        if removed > 0:
            print(f"   ✅ {removed} imágenes raw eliminadas")
    
    # Delete all crops
    crops_dir = Path('images/crops')
    if crops_dir.exists():
        removed_dirs = 0
        removed_files = 0
        for crop_dir in crops_dir.iterdir():
            if crop_dir.is_dir():
                crop_count = len(list(crop_dir.glob('*.jpg')))
                shutil.rmtree(crop_dir)
                removed_dirs += 1
                removed_files += crop_count
        if removed_dirs > 0:
            print(f"   ✅ {removed_dirs} directorios de crops eliminados ({removed_files} crops)")
    
    print("   ✅ Limpieza completa\n")
else:
    print("   ℹ️  Modo testing desactivado, no se limpia\n")
CLEANUP_EOF

# Enable MPS fallback for unsupported operations (e.g., _assert_async)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Enable MPS fallback for unsupported operations (e.g., _assert_async)
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Ejecutando live_detection.py..."
echo ""

# Use -u flag to disable buffering (show output immediately)
"$VENV_PYTHON" -u live_detection.py

