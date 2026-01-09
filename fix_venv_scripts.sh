#!/bin/bash
# Script para arreglar todos los scripts run_*.sh después de actualización de Mac
# Cambia de usar 'source venv/bin/activate' + 'python3' a usar 'venv/bin/python3' directamente

cd "$(dirname "$0")"

for script in run_*.sh; do
    if [ ! -f "$script" ]; then
        continue
    fi
    
    echo "📝 Actualizando $script..."
    
    # Backup
    cp "$script" "${script}.backup"
    
    # Replace source venv/bin/activate + python3 with VENV_PYTHON pattern
    perl -i -pe 's/source venv\/bin\/activate\n\n/# Use venv Python directly (more reliable after Mac update)\nVENV_PYTHON="venv\/bin\/python3"\nif [ ! -f "\$VENV_PYTHON" ]; then\n    echo "❌ Python del venv no encontrado"\n    echo "   Ejecuta primero: bash setup_venv.sh"\n    exit 1\nfi\n\n/g' "$script"
    
    # Replace standalone python3 with "$VENV_PYTHON"
    perl -i -pe 's/^python3 /"$VENV_PYTHON" /g' "$script"
    perl -i -pe 's/ python3 / "$VENV_PYTHON" /g' "$script"
    
    # Replace python << with "$VENV_PYTHON" <<
    perl -i -pe 's/^python <</"$VENV_PYTHON" <</g' "$script"
    
    # Replace python -c with "$VENV_PYTHON" -c
    perl -i -pe 's/ python -c/ "$VENV_PYTHON" -c/g' "$script"
    perl -i -pe 's/! python -c/! "$VENV_PYTHON" -c/g' "$script"
    
    echo "   ✅ $script actualizado"
done

echo ""
echo "✅ Todos los scripts actualizados"

