# 📤 Subir Código 1UP a RunPod - Paso a Paso

**Fecha:** 2026-01-10  
**Estado:** En progreso

---

## 🎯 Objetivo

Subir todos los archivos necesarios del proyecto 1UP al servidor RunPod para procesamiento GPU.

---

## ✅ PASO 1: Crear Estructura de Directorios

**En RunPod SSH terminal, ejecuta:**

```bash
# Crear directorios principales
mkdir -p ~/1UP_2/server
mkdir -p ~/1UP_2/client
mkdir -p ~/1UP_2/images/crops
mkdir -p ~/1UP_2/images/raw
mkdir -p ~/1UP_2/database

# Verificar
ls -la ~/1UP_2/
```

**Debe mostrar:**
```
server/
client/
images/
database/
```

---

## ✅ PASO 2: Crear Archivos de Configuración

**Copia y pega estos comandos UNO POR UNO en RunPod SSH:**

### 2.1: Crear `server/config_server.yaml`

```bash
cat > ~/1UP_2/server/config_server.yaml << 'EOF'
# 1UP Server Configuration - RunPod GPU
# Optimized for CUDA processing with maximum detection

sam3:
  device: "cuda"  # GPU CUDA para RunPod
  filtering:
    enabled: false  # Detect EVERYTHING, let Claude filter
  enhance_image: true  # CLAHE for dark objects
  text_prompt: ""  # Empty = automatic (uses "visual" prompt)

claude:
  api_key_env: "CLAUDE_API_KEY"  # Read from environment
  model: "claude-sonnet-4-20250514"
  max_tokens: 16000  # For large batches

storage:
  crops_dir: "images/crops"
  raw_dir: "images/raw"

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1  # Single worker for GPU (don't parallelize GPU work)
EOF
```

### 2.2: Crear `client/config_client.yaml`

```bash
cat > ~/1UP_2/client/config_client.yaml << 'EOF'
# 1UP Client Configuration - Local Capture
# Captures frames and sends to server for processing

camera:
  # Reolink RTSP stream (4K for maximum quality)
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"
  resolution: [3840, 2160]  # 4K for maximum crop quality
  fps: 2
  buffer_size: 1  # Low latency for RTSP
  
  # Image quality validation
  quality_check:
    enabled: true
    min_sharpness: 20.0  # Reject blurry frames
    warning_sharpness: 50.0

server:
  # RunPod server URL (update with your RunPod endpoint)
  url: "http://localhost:8000"  # Default local, update to RunPod IP:port
  timeout: 120  # 2 minutes timeout for processing
EOF
```

### 2.3: Crear `requirements.txt`

```bash
cat > ~/1UP_2/requirements.txt << 'EOF'
# 1UP - Dependencies

# Core ML/AI
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0

# SAM 3 (Segment Anything Model 3)
einops>=0.8.0
pycocotools>=2.0.0
psutil>=7.0.0
omegaconf>=2.3.0

# Claude API
anthropic>=0.18.0

# Configuration
PyYAML>=6.0

# Web Application
Flask>=3.0.0
EOF
```

### 2.4: Crear `server/requirements_server.txt`

```bash
cat > ~/1UP_2/server/requirements_server.txt << 'EOF'
# 1UP Server Dependencies - RunPod GPU
# Install with: pip install -r server/requirements_server.txt

# Core ML/AI
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0

# SAM 3 (Segment Anything Model 3)
einops>=0.8.0
pycocotools>=2.0.0
psutil>=7.0.0
omegaconf>=2.3.0

# Claude API
anthropic>=0.18.0

# Configuration
PyYAML>=6.0

# FastAPI
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# HTTP client (for health checks)
httpx>=0.25.0
EOF
```

**Verificar:**
```bash
ls -la ~/1UP_2/server/
ls -la ~/1UP_2/client/
```

---

## ✅ PASO 3: Crear Archivos Python (Script Base64)

**IMPORTANTE:** Este paso requiere copiar el contenido completo del archivo `runpod_python_files.txt` desde tu Mac local.

**Opción A: Copiar desde Mac local**

1. **En tu Mac local, abre el archivo:**
   ```bash
   cat /Users/jba7790/Desktop/1UP_2/runpod_python_files.txt
   ```

2. **Copia TODO el contenido** (desde `# ==========================================` hasta el final)

3. **En RunPod SSH terminal, pega TODO el contenido**

4. **Presiona ENTER** para ejecutar el script

**El script creará estos archivos:**
- ✅ `filters.py`
- ✅ `image_quality.py`
- ✅ `camera_utils.py`
- ✅ `analyzer.py`
- ✅ `storage_v2.py`
- ✅ `storage.py`
- ✅ `detector.py`
- ✅ `server/api.py`
- ✅ `client/capture_client.py`

**Verificar:**
```bash
ls -la ~/1UP_2/*.py
ls -la ~/1UP_2/server/*.py
ls -la ~/1UP_2/client/*.py
```

**Debe mostrar los 9 archivos Python.**

---

## ✅ PASO 4: Verificar Estructura Completa

**En RunPod, ejecuta:**

```bash
cd ~/1UP_2
tree -L 2  # Si tree está instalado
# O usar:
find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.txt" | sort
```

**Estructura esperada:**
```
1UP_2/
├── server/
│   ├── api.py
│   ├── config_server.yaml
│   └── requirements_server.txt
├── client/
│   ├── capture_client.py
│   └── config_client.yaml
├── filters.py
├── image_quality.py
├── camera_utils.py
├── analyzer.py
├── storage_v2.py
├── storage.py
├── detector.py
├── requirements.txt
└── images/
    ├── crops/
    └── raw/
```

---

## 🐛 Troubleshooting

### Error: "No such file or directory"
- Verifica que creaste los directorios en PASO 1
- Verifica que estás en `~/1UP_2` cuando ejecutas los comandos

### Error: "Permission denied"
- Verifica permisos: `chmod -R 755 ~/1UP_2`

### Script base64 no funciona
- Asegúrate de copiar TODO el contenido del archivo `runpod_python_files.txt`
- Verifica que no haya caracteres extraños al pegar
- Intenta ejecutar línea por línea si es necesario

---

## 📝 Siguiente Paso

Una vez completado este paso, continúa con:
- **PASO 4:** Crear entorno virtual
- **PASO 5:** Instalar SAM3
- Ver `docs/RUNPOD_SETUP.md` para los siguientes pasos

---

**Última actualización:** 2026-01-10
