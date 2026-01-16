# 🔐 Autenticación SAM 3 con HuggingFace

## Problema

SAM 3 requiere acceso a un repositorio **gated** (protegido) en HuggingFace. Sin autenticación, obtendrás el error:

```
401 Client Error
Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json
Access to model facebook/sam3 is restricted.
```

## Solución: Paso a Paso

### Paso 1: Solicitar Acceso al Repositorio

1. **Ve a la página del repositorio SAM 3:**
   https://huggingface.co/facebook/sam3

2. **Solicita acceso:**
   - Haz clic en el botón **"Request access"** o **"Request to access this model"**
   - Completa el formulario (explica tu uso del modelo)
   - Meta/Facebook revisará tu solicitud (puede tomar horas o días)

3. **Espera confirmación:**
   - Recibirás un email cuando tu solicitud sea aprobada
   - Asegúrate de aceptar cualquier invitación que llegue por email

### Paso 2: Crear Token de Acceso en HuggingFace

1. **Ve a tu perfil de HuggingFace:**
   https://huggingface.co/settings/tokens

2. **Crea un nuevo token:**
   - Haz clic en **"New token"**
   - **Nombre:** `sam3-1up` (o el que prefieras)
   - **Tipo:** `Read` (solo lectura, suficiente para descargar modelos)
   - Haz clic en **"Generate token"**

3. **Copia el token:**
   - ⚠️ **IMPORTANTE:** Copia el token inmediatamente (no se mostrará de nuevo)
   - El token tendrá el formato: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Paso 3: Autenticarse en el Sistema

Una vez tengas el token, autentica HuggingFace:

```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate

# Opción 1: Usando Python directamente (recomendado)
python3 -m huggingface_hub.cli login

# Te pedirá el token, pégalo y presiona Enter
```

**O usar la CLI de HuggingFace directamente si está instalada:**

```bash
hf auth login
```

### Paso 4: Verificar Autenticación

Verifica que estés autenticado correctamente:

```bash
python3 -m huggingface_hub.cli whoami
```

Deberías ver tu nombre de usuario de HuggingFace.

### Paso 5: Probar Descarga Manual (Opcional)

Puedes probar descargar el checkpoint manualmente:

```bash
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

# Verificar que el token esté configurado
token = os.environ.get('HF_TOKEN') or os.path.expanduser('~/.huggingface/token')
print(f"Token file: {token}")

try:
    # Intentar descargar el config (archivo pequeño)
    path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="config.json",
        token=None  # Usará el token de ~/.huggingface/token
    )
    print(f"✅ Config descargado exitosamente: {path}")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

## Alternativa: Token como Variable de Entorno

Si prefieres no usar la autenticación interactiva, puedes configurar el token como variable de entorno:

```bash
# En tu terminal o en ~/.zshrc / ~/.bashrc
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# O para esta sesión solamente:
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
./run_live_detection_with_claude.sh
```

**⚠️ SEGURIDAD:** No compartas tu token ni lo subas a Git. Ya está en `.gitignore`:

```
.env
.env.local
.claude_api_key
*.log
```

## Verificar que Todo Funciona

Después de autenticarte, ejecuta el sistema nuevamente:

```bash
./run_live_detection_with_claude.sh
```

Deberías ver:

```
✅ RTSP stream opened: 3840x2160
Loading SAM 3...
Downloading checkpoint from HuggingFace...  # Primera vez
✅ SAM 3 loaded on mps
```

## Troubleshooting

### Error: "401 Unauthorized"
- ✅ Verifica que hayas solicitado acceso al repositorio
- ✅ Verifica que hayas sido aceptado (revisa tu email)
- ✅ Verifica que estés autenticado: `python3 -m huggingface_hub.cli whoami`

### Error: "403 Forbidden"
- Tu solicitud de acceso puede no haber sido aprobada aún
- Contacta con el equipo de SAM 3 si ha pasado mucho tiempo

### Error: "Token not found"
- Ejecuta `python3 -m huggingface_hub.cli login` nuevamente
- Verifica que el token esté guardado en `~/.huggingface/token`

### Token Guardado Incorrectamente

Si necesitas eliminar el token anterior y empezar de nuevo:

```bash
rm ~/.huggingface/token
python3 -m huggingface_hub.cli login  # Login nuevamente
```

## Referencias

- Repositorio SAM 3: https://huggingface.co/facebook/sam3
- Documentación HuggingFace Hub: https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication
- Generar token: https://huggingface.co/settings/tokens
