#!/bin/bash
# Quick script to set HuggingFace token
# Usage: ./set_hf_token.sh hf_xxxxxxxxxxxxx

if [ -z "$1" ]; then
    echo "❌ Uso: ./set_hf_token.sh hf_xxxxxxxxxxxxx"
    echo ""
    echo "Obtén tu token en: https://huggingface.co/settings/tokens"
    exit 1
fi

TOKEN="$1"

# Validate token format
if [[ ! "$TOKEN" =~ ^hf_ ]]; then
    echo "⚠️  El token debería empezar con 'hf_'"
    read -p "¿Continuar de todas formas? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# Save to .hf_token file
echo "$TOKEN" > .hf_token
chmod 600 .hf_token  # Secure: only owner can read/write
echo "✅ Token guardado en .hf_token"

# Also try to set as environment variable for current session
export HF_TOKEN="$TOKEN"
echo "✅ Token también configurado como variable de entorno (HF_TOKEN)"

# Test authentication
echo ""
echo "🧪 Probando autenticación..."
cd /Users/jba7790/Desktop/1UP_2
venv/bin/python3 << 'EOF'
from huggingface_hub import whoami
try:
    user = whoami()
    print(f"✅ Autenticado como: {user}")
except Exception as e:
    print(f"❌ Error de autenticación: {e}")
    print("💡 Verifica que el token sea correcto y tengas acceso al repositorio facebook/sam3")
EOF
