# 🧪 Testing RunPod Setup - Plan de Pruebas

**Fecha:** 2026-01-10  
**Objetivo:** Probar que el servidor RunPod funciona correctamente

---

## 📋 Plan de Pruebas Paso a Paso

### FASE 1: Verificar Servidor RunPod (Sin Procesamiento)

#### 1.1 Conectar por SSH

```bash
ssh nq11ijfcz10phy-6441116e@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**✅ Verificar:** Debe conectarse sin errores

---

#### 1.2 Verificar CUDA Disponible

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**✅ Resultado esperado:**
```
CUDA available: True
Device: NVIDIA RTX 4000 Ada Generation
```

**❌ Si falla:** GPU no disponible, revisar pod en RunPod dashboard

---

#### 1.3 Verificar SAM3 Instalado

```bash
python3 -c "from sam3.model_builder import build_sam3_image_model; print('✅ SAM3 installed')"
```

**✅ Resultado esperado:**
```
✅ SAM3 installed
```

**❌ Si falla:** Instalar SAM3 (ver RUNPOD_SETUP.md PASO 5)

---

#### 1.4 Verificar Tokens Configurados

```bash
cd ~/1UP_2
test -f .hf_token && echo "✅ HF_TOKEN exists" || echo "❌ HF_TOKEN missing"
test -f .claude_api_key && echo "✅ CLAUDE_API_KEY exists" || echo "❌ CLAUDE_API_KEY missing"
```

**✅ Resultado esperado:**
```
✅ HF_TOKEN exists
✅ CLAUDE_API_KEY exists
```

**❌ Si falla:** Configurar tokens (ver RUNPOD_SETUP.md PASO 6)

---

### FASE 2: Probar API del Servidor (Local en RunPod)

#### 2.1 Iniciar Servidor

```bash
cd ~/1UP_2
source venv/bin/activate
cd server
python api.py
```

**✅ Verificar:** Debe iniciar sin errores, mostrar:
```
🚀 Starting 1UP Detection API...
✅ SAM3 detector initialized
✅ Claude analyzer initialized
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**❌ Si falla:** Revisar logs, verificar dependencias instaladas

---

#### 2.2 Probar Health Endpoint (Desde RunPod)

**En otra terminal SSH:**
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

**❌ Si falla:** Servidor no está corriendo o hay error

---

#### 2.3 Probar con Imagen de Test (Opcional)

**Crear script de test simple:**
```python
# test_api_local.py
import requests
import base64
import cv2
import numpy as np

# Crear imagen de test
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), -1)
_, buffer = cv2.imencode('.jpg', img)
img_base64 = base64.b64encode(buffer).decode('utf-8')

# Enviar a API
response = requests.post(
    'http://localhost:8000/detect',
    json={
        'image_base64': img_base64,
        'timestamp': 'test_20260110'
    },
    timeout=120
)

print(response.json())
```

```bash
python test_api_local.py
```

**✅ Verificar:** Debe retornar JSON con detections y crops

---

### FASE 3: Probar Cliente Local (Desde Mac)

#### 3.1 Verificar Cliente Puede Conectar

**En Mac local:**
```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate

# Verificar URL servidor
cat client/config_client.yaml | grep url
```

**✅ Verificar:** URL debe ser la de RunPod (no localhost)

---

#### 3.2 Probar Health Check desde Cliente

**Crear script de test:**
```python
# test_server_connection.py
import requests
from client.config_client import config  # O leer YAML directamente

import yaml
with open('client/config_client.yaml', 'r') as f:
    config = yaml.safe_load(f)

url = config['server']['url']
print(f"Testing server: {url}")

try:
    response = requests.get(f"{url}/health", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")
```

```bash
python test_server_connection.py
```

**✅ Resultado esperado:**
```
Status: 200
Response: {'status': 'healthy', 'detector_ready': True, 'analyzer_ready': True}
```

**❌ Si falla:**
- Servidor no está corriendo
- Puerto no está expuesto en RunPod
- URL incorrecta

---

#### 3.3 Probar Cliente Completo

```bash
python client/capture_client.py
```

**✅ Verificar:**
- Cliente inicia sin errores
- Muestra preview de cámara
- Presionar SPACE envía frame al servidor
- Recibe respuesta con detections y crops

**❌ Si falla:** Revisar logs, verificar conexión servidor

---

### FASE 4: Probar Procesamiento Completo (End-to-End)

#### 4.1 Capturar Frame Real

**En cliente:**
1. Presionar SPACE para capturar
2. Esperar respuesta del servidor (15-60 segundos)
3. Verificar resultados

**✅ Resultado esperado:**
- Logs muestran "✅ Server response: X objects, Y crops"
- No hay errores
- Detections tiene objetos identificados

---

#### 4.2 Verificar Crops Generados

**En RunPod SSH:**
```bash
cd ~/1UP_2
ls -la images/crops/
```

**✅ Verificar:** Debe haber directorios con crops generados

---

## 🐛 Troubleshooting Común

### Error: "CUDA not available"
- **Causa:** GPU no disponible
- **Solución:** Verificar pod en RunPod dashboard, reiniciar pod

### Error: "HF_TOKEN not found"
- **Causa:** Token no configurado
- **Solución:** `export HF_TOKEN="..."` y `echo $HF_TOKEN > .hf_token`

### Error: "Connection refused" (cliente)
- **Causa:** Servidor no corriendo o puerto no expuesto
- **Solución:** 
  - Verificar servidor está corriendo en RunPod
  - Verificar puerto 8000 está mapeado en RunPod dashboard

### Error: "Timeout" (cliente)
- **Causa:** Procesamiento muy lento o servidor no responde
- **Solución:** Aumentar timeout en `client/config_client.yaml`

### Error: "save_crops_for_useful_objects" falla
- **Causa:** Estructura de datos incorrecta
- **Solución:** Revisar API del servidor (ver código)

---

## ✅ Checklist Final

- [ ] Servidor inicia sin errores
- [ ] Health endpoint responde correctamente
- [ ] CUDA disponible y funcionando
- [ ] SAM3 puede inicializarse en GPU
- [ ] Claude API funciona
- [ ] Cliente puede conectar al servidor
- [ ] Procesamiento completo funciona (captura → servidor → respuesta)
- [ ] Crops se generan correctamente

---

**Última actualización:** 2026-01-10
