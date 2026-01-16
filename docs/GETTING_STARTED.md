# 🚀 Inicio Rápido - 1UP

Guía completa para empezar con 1UP.

## 📋 Requisitos Previos

- **macOS** (M2 recomendado) o **Raspberry Pi**
- **Python 3.12** (NO usar 3.14 - ver [Troubleshooting](#troubleshooting))
- **Cámara externa** (opcional, puede usar la del laptop)

## ⚡ Setup Rápido

### Paso 1: Instalación

```bash
cd ~/Desktop/1UP_2
bash setup_venv.sh
```

**Tiempo estimado:** 10-15 minutos

**⚠️ IMPORTANTE**: SAM 3 requiere acceso a checkpoints en HuggingFace:
1. Ve a: https://huggingface.co/models?search=sam3
2. Solicita acceso a los checkpoints
3. Autentícate: `hf auth login --token <tu-token>`

### Paso 2: Verificar Cámaras

```bash
./run_list_cameras.sh
```

### Paso 3: Probar Detección

```bash
# Solo detección visual (rápido)
./run_test_detection.sh

# O con análisis Claude (completo)
./run_live_detection_with_claude.sh
```

## 🎮 Controles

### Detección en Vivo (`live_detection.py`)

- **SPACE** = Detectar objetos en frame actual
- **S** = Guardar y analizar con Claude
- **C** = Limpiar detecciones
- **Q** = Salir

### Test Simple (`test_detection.py`)

- **SPACE** = Capturar y detectar
- **D** = Toggle overlay
- **Q** = Salir

## 📝 Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `setup_venv.sh` | Setup inicial (solo una vez) |
| `run_list_cameras.sh` | Lista cámaras disponibles |
| `run_test_detection.sh` | Prueba rápida (sin Claude) |
| `run_live_detection.sh` | Detección en vivo (sin Claude) |
| `run_live_detection_with_claude.sh` | Detección + análisis Claude |
| `run_main.sh` | Pipeline completo |
| `run_web.sh` | Servidor web (localhost:5001) |

## 🔧 Troubleshooting

### Python 3.14 no compatible

**Problema:** Python 3.14 es muy nuevo y numpy/torch no compilan.

**Solución:** Usar Python 3.12

```bash
# Con Homebrew
brew install python@3.12
python3.12 -m venv venv

# O con pyenv
pyenv install 3.12.7
pyenv local 3.12.7
python -m venv venv
```

### "command not found: ./run_*.sh"

```bash
chmod +x run_*.sh setup_venv.sh
```

### "No module named 'cv2'"

Asegúrate de haber ejecutado `setup_venv.sh` y activado el venv:

```bash
source venv/bin/activate
```

### "No cameras found"

1. Verifica que la cámara esté conectada
2. Cierra otras apps que usen la cámara (Zoom, Teams, etc.)
3. Prueba diferentes índices en `config.yaml`

### "Failed to open camera"

Ejecuta `./run_list_cameras.sh` para ver cámaras disponibles y edita `config.yaml` con el índice correcto.

### SAM 3 no carga

1. Verifica acceso a HuggingFace: `hf auth whoami`
2. Si no estás autenticado: `hf auth login --token <token>`
3. Revisa logs para errores específicos

### Error MPS en macOS Intel

**⚠️ IMPORTANTE:** Mac Intel (pre-2020) **NO tiene MPS** (Metal Performance Shaders).  
Si ves errores de MPS, es porque tu Mac es Intel, no Apple Silicon.

**Solución:** Usa CPU en `config.yaml`:
```yaml
sam3:
  device: "cpu"  # Mac Intel no tiene MPS - usar CPU
```

**Detectar tipo de Mac:**
```bash
python3 -c "import platform; print(f'Processor: {platform.processor()}'); print(f'Machine: {platform.machine()}')"
```

- **Intel:** Processor: `i386`, Machine: `x86_64` → Usar CPU
- **Apple Silicon:** Processor: `arm`, Machine: `arm64` → Puede usar MPS

📖 Ver [docs/HARDWARE_CONFIG.md](HARDWARE_CONFIG.md) para detalles completos.

## 💡 Tips

- **Primera vez**: Usa `run_test_detection.sh` (más rápido, sin Claude)
- **Detección lenta**: SAM 3 tarda 5-15 segundos por imagen (normal)
- **Muchos objetos**: Ajusta filtros de SAM en `config.yaml` (aunque están deshabilitados por defecto)
- **Text prompts**: Usa `text_prompt` en `config.yaml` para conceptos específicos (ej: "bag", "shoes")

## 📚 Siguiente Paso

Una vez funcionando, lee:
- **[Uso del Sistema](USAGE.md)** - Cómo usar live detection y análisis
- **[Configuración SAM 3](SAM3_CONFIG.md)** - Ajustar parámetros de detección
- **[Filtrado](FILTERING.md)** - Entender el sistema de filtrado

