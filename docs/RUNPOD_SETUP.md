# 🚀 Setup RunPod GPU - Servidor de Procesamiento 1UP

**Fecha:** 2026-01-10  
**Objetivo:** Configurar servidor RunPod para procesar frames 4K con SAM3 GPU + Claude

---

## 📋 Contexto

### Problema Actual
- Mac Intel 2018 no puede procesar 4K + SAM3 eficientemente
- Tarda 30-60 segundos por frame
- Errores HEVC masivos
- CPU muy lento para detección

### Solución: RunPod GPU Cloud
- **Pod:** RTX 4000 Ada, 20GB VRAM
- **Coste:** ~$0.26/hora
- **Ventaja:** GPU CUDA → Procesamiento rápido (5-15s vs 30-60s)

---

## 🏗️ Arquitectura

```
Mac Local (Cursor)          RunPod GPU Server
├─ Captura Reolink 4K  →    ├─ Recibe frame (POST /detect)
├─ Envía frame         →    ├─ SAM3 detecta (CUDA GPU)
├─ Espera resultado    ←    ├─ Claude analiza
└─ Muestra crops       ←    └─ Retorna results (JSON)
```

---

## ✅ Estado RunPod

**Pod activo:** `ytoissxrquxq5s`  
**SSH:** `ssh ytoissxrquxq5s-6441116d@ssh.runpod.io -i ~/.ssh/id_ed25519`

---

## 📝 Comandos Paso a Paso

### PASO 1: Conectar por SSH al RunPod

```bash
ssh ytoissxrquxq5s-6441116d@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### PASO 2: Instalar Dependencias Base (en RunPod)

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python 3.10+ y herramientas
sudo apt install -y python3.10 python3-pip python3-venv git

# Verificar Python
python3 --version  # Debe ser 3.10+
```

### PASO 3: Subir Código 1UP (desde Mac Local)

**Desde tu Mac local:**
```bash
cd /Users/jba7790/Desktop/1UP_2
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'images' --exclude 'models' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  ./ ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/
```

### PASO 4: Crear Entorno Virtual (en RunPod)

```bash
cd ~/1UP_2
python3 -m venv venv
source venv/bin/activate
```

### PASO 5: Instalar SAM3 (en RunPod)

```bash
# Clonar SAM3
cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Verificar instalación
python -c "from sam3.model_builder import build_sam3_image_model; print('✅ SAM3 installed')"
```

### PASO 6: Configurar Tokens (en RunPod)

```bash
cd ~/1UP_2

# HuggingFace token (para SAM3 checkpoints)
export HF_TOKEN="TU_HF_TOKEN_AQUI"
echo $HF_TOKEN > .hf_token

# Claude API key (copiar desde Mac local)
# Desde Mac local:
scp -i ~/.ssh/id_ed25519 .claude_api_key ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/.claude_api_key

# O exportar directamente en RunPod:
export CLAUDE_API_KEY="tu-claude-api-key-aqui"
echo $CLAUDE_API_KEY > .claude_api_key
```

### PASO 7: Instalar Dependencias Python (en RunPod)

```bash
cd ~/1UP_2
source venv/bin/activate

# Instalar dependencias servidor
pip install -r server/requirements_server.txt

# Instalar dependencias compartidas (si existen)
pip install -r requirements.txt
```

### PASO 8: Verificar CUDA (en RunPod)

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Debe mostrar:**
```
CUDA available: True
CUDA device: NVIDIA RTX 4000 Ada Generation
```

### PASO 9: Probar SAM3 en GPU (en RunPod)

```bash
cd ~/1UP_2
source venv/bin/activate
python3 -c "
from detector import SAM3Detector
import torch
print(f'CUDA: {torch.cuda.is_available()}')
detector = SAM3Detector(device='cuda')
print('✅ SAM3 initialized on CUDA')
"
```

### PASO 10: Iniciar Servidor API (en RunPod)

```bash
cd ~/1UP_2
source venv/bin/activate
cd server
python api.py
```

**O con uvicorn (mejor para producción):**
```bash
cd ~/1UP_2/server
source ../venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Verificar que el servidor está corriendo:**
```bash
curl http://localhost:8000/health
```

**Debe retornar:**
```json
{
  "status": "healthy",
  "detector_ready": true,
  "analyzer_ready": true
}
```

### PASO 11: Exponer Puerto RunPod

**En RunPod Web UI:**
1. Ir a tu pod (`ytoissxrquxq5s`)
2. Configurar "Port Mapping"
3. Añadir: `8000:8000` (host:container)
4. Copiar URL pública (ej: `https://abc123-8000.proxy.runpod.net`)

### PASO 12: Actualizar Cliente Local (en Mac)

Editar `client/config_client.yaml`:
```yaml
server:
  url: "https://abc123-8000.proxy.runpod.net"  # URL pública RunPod (ACTUALIZAR)
  timeout: 120
```

### PASO 13: Probar Cliente Local (en Mac)

```bash
# En tu Mac local:
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate
python client/capture_client.py
```

**Presiona SPACE** para capturar y enviar al servidor.

---

## 🔧 Configuración Máxima Detección

El servidor está configurado para máxima detección en GPU:

```yaml
# server/config_server.yaml
sam3:
  device: "cuda"  # GPU CUDA
  filtering:
    enabled: false  # Detecta TODO
  enhance_image: true  # CLAHE para objetos oscuros
```

---

## 🐛 Troubleshooting

### Error: "CUDA not available"
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Si no muestra GPU, el pod puede no tener GPU asignada
# Verificar en RunPod dashboard
```

### Error: "HF_TOKEN not found"
```bash
export HF_TOKEN="TU_HF_TOKEN_AQUI"
hf auth login  # O usar token directamente
```

### Error: "Claude API key not found"
```bash
export CLAUDE_API_KEY="tu-api-key"
# O copiar desde local:
scp -i ~/.ssh/id_ed25519 .claude_api_key user@host:~/1UP_2/.claude_api_key
```

### Error: "SAM3 model not found"
```bash
# SAM3 descarga checkpoints automáticamente si está autenticado
hf auth login
# O verificar token:
python -c "from huggingface_hub import whoami; print(whoami())"
```

### Servidor no responde
```bash
# Verificar que está corriendo
ps aux | grep uvicorn

# Ver logs
tail -f ~/1UP_2/server/logs/*.log  # Si hay logs
```

---

## 📊 Rendimiento Esperado

### Con GPU (RunPod CUDA)
- **SAM3 detección:** 5-15 segundos (vs 30-60s CPU)
- **Claude análisis:** 10-30 segundos (igual, API externa)
- **Total:** 15-45 segundos por frame (vs 40-90s CPU)

### Coste
- **RunPod:** ~$0.26/hora
- **Por frame (45s):** ~$0.003
- **100 frames:** ~$0.30

---

## 📝 Notas

- **Puerto:** El servidor corre en puerto 8000 por defecto
- **Timeout:** Cliente espera 120 segundos por respuesta
- **Crops:** Se generan en servidor, paths retornados en JSON
- **Storage:** Considera usar S3 para producción (no configurado aquí)

---

**Última actualización:** 2026-01-10  
**Mantenido por:** Jose (@jba7790)
