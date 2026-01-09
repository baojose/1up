#!/bin/bash
# Script auxiliar para ejecutar el experimento con imagen del portapapeles

echo "📋 Intentando leer imagen del portapapeles..."

# Guardar imagen del portapapeles en archivo temporal
TEMP_IMAGE=$(mktemp /tmp/claude_crop_experiment_XXXXXX.png)

# macOS: leer imagen del portapapeles usando osascript
osascript <<EOF
tell application "System Events"
    set clipboardImage to (the clipboard as «class PNGf»)
    if clipboardImage is not equal to "" then
        set fileRef to open for access file "$TEMP_IMAGE" with write permission
        write clipboardImage to fileRef
        close access fileRef
        return "success"
    else
        return "no_image"
    end if
end tell
EOF

if [ -f "$TEMP_IMAGE" ] && [ -s "$TEMP_IMAGE" ]; then
    echo "✅ Imagen cargada desde portapapeles"
    echo "🚀 Ejecutando script experimental..."
    echo ""
    
    # Ejecutar el script experimental con la imagen temporal
    venv/bin/python3 experimental_claude_crops.py "$TEMP_IMAGE"
    
    # Limpiar archivo temporal
    rm -f "$TEMP_IMAGE"
else
    echo "❌ No se encontró imagen en el portapapeles"
    echo ""
    echo "💡 Instrucciones:"
    echo "   1. Copia una imagen al portapapeles (Cmd+C o clic derecho → Copiar)"
    echo "   2. Ejecuta este script de nuevo: ./run_experiment_from_clipboard.sh"
    echo ""
    echo "O usa directamente:"
    echo "   venv/bin/python3 experimental_claude_crops.py /ruta/a/imagen.jpg"
    exit 1
fi

