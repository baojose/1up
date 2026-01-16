# 🧪 Testing Local - Guía Paso a Paso

**Fecha:** 2026-01-10  
**Objetivo:** Probar servidor y cliente localmente antes de usar RunPod

---

## 📋 Plan de Testing Local

### PASO 1: Preparar Configuración

#### 1.1 Configurar Servidor para CPU (Local)

Editar `server/config_server.yaml`:
```yaml
sam3:
  device: "cpu"  # CPU para testing local (cambiar a "cuda" para RunPod)
```

#### 1.2 Configurar Cliente para Localhost

Editar `client/config_client.yaml`:
```yaml
server:
  url: "http://localhost:8000"  # Servidor local
  timeout: 120
```

---

### PASO 2: Iniciar Servidor

**Terminal 1:**
```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate
cd server
python api.py
```

**✅ Verificar que inicia correctamente:**
```
🚀 Starting 1UP Detection API...
Loading SAM 3...
✅ Using CPU (Intel Mac or no GPU available)
✅ SAM 3 loaded on cpu
✅ Claude analyzer initialized (claude-sonnet-4-20250514)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**❌ Si falla:**
- Verificar dependencias: `pip install -r server/requirements_server.txt`
- Verificar tokens: `.claude_api_key` y `.hf_token` deben existir
- Verificar SAM3 instalado: `python -c "from sam3.model_builder import build_sam3_image_model"`

---

### PASO 3: Probar Health Endpoint

**Terminal 2 (nueva):**
```bash
curl http://localhost:8000/health
```

**✅ Resultado esperado:**
```json
{
  "status": "healthy",
  "detector_ready": true,
  "analyzer_ready": true
}
```

---

### PASO 4: Probar con Script de Test

**Terminal 2:**
```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate
python test_server_local.py
```

**✅ Resultado esperado:**
```
🧪 Testing 1UP Server API (Local)
✅ Health check passed
✅ Detection successful!
   Objects detected: X
   Crops generated: Y
```

**❌ Si falla:**
- Revisar logs del servidor (Terminal 1)
- Verificar que Claude API key está configurada
- Verificar que SAM3 puede procesar imágenes

---

### PASO 5: Probar Cliente Completo

**Terminal 2:**
```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate
python client/capture_client.py
```

**✅ Verificar:**
1. Cliente inicia sin errores
2. Muestra preview de cámara Reolink
3. Presionar SPACE captura y envía al servidor
4. Recibe respuesta con detections y crops

**❌ Si falla:**
- Verificar cámara Reolink está accesible
- Verificar URL del servidor en `client/config_client.yaml`
- Revisar logs del servidor

---

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
# Instalar dependencias
pip install -r server/requirements_server.txt
```

### Error: "CLAUDE_API_KEY not found"
```bash
# Verificar token existe
test -f .claude_api_key && echo "✅ Exists" || echo "❌ Missing"

# Si falta, copiar desde donde lo tengas
```

### Error: "SAM3 not installed"
```bash
# Verificar SAM3
python -c "from sam3.model_builder import build_sam3_image_model; print('✅ OK')"

# Si falla, instalar:
cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Error: "Connection refused" (cliente)
- Verificar servidor está corriendo (Terminal 1)
- Verificar URL en `client/config_client.yaml` es `http://localhost:8000`

### Error: "Timeout" (cliente)
- Procesamiento en CPU es lento (30-60 segundos)
- Aumentar timeout en `client/config_client.yaml` si es necesario

---

## ✅ Checklist

- [ ] Servidor inicia sin errores
- [ ] Health endpoint responde correctamente
- [ ] Script de test pasa
- [ ] Cliente puede conectar al servidor
- [ ] Procesamiento completo funciona (captura → servidor → respuesta)
- [ ] Crops se generan correctamente

---

## 📝 Notas

- **CPU es lento:** Procesamiento puede tardar 30-60 segundos (vs 5-15s en GPU)
- **Testing local:** Solo valida que el código funciona, no el rendimiento
- **Si funciona localmente:** Listo para subir a RunPod y probar con GPU

---

**Última actualización:** 2026-01-10
