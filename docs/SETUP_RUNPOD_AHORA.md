# 🚀 SETUP RUNPOD - MÉTODO RÁPIDO Y DIRECTO

**Fecha:** 2026-01-10  
**Objetivo:** Hacer funcionar 1UP en RunPod AHORA

---

## ⚡ MÉTODO RÁPIDO (TODO AUTOMÁTICO)

### Ejecuta este comando en tu Mac:

```bash
cd /Users/jba7790/Desktop/1UP_2
./SETUP_RUNPOD_COMPLETE.sh
```

**Este script:**
- ✅ Crea estructura de directorios
- ✅ Sube todos los archivos Python esenciales
- ✅ Crea archivos de configuración
- ✅ Instala dependencias base
- ✅ Verifica CUDA

**Tiempo estimado:** 5-10 minutos

---

## 📋 PASOS MANUALES (si prefieres hacerlo tú)

### PASO 1: Subir archivos

```bash
cd /Users/jba7790/Desktop/1UP_2
./upload_to_runpod_now.sh
```

### PASO 2: Conectar a RunPod

```bash
ssh -i ~/.ssh/id_ed25519 ytoissxrquxq5s-6441116d@ssh.runpod.io
```

### PASO 3: Crear entorno virtual

```bash
cd ~/1UP_2
python3 -m venv venv
source venv/bin/activate
```

### PASO 4: Instalar SAM3

```bash
cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### PASO 5: Configurar tokens

```bash
cd ~/1UP_2
export HF_TOKEN="TU_HF_TOKEN_AQUI"
export CLAUDE_API_KEY="tu-claude-api-key-aqui"

# Guardar en archivos
echo $HF_TOKEN > .hf_token
echo $CLAUDE_API_KEY > .claude_api_key
```

### PASO 6: Instalar dependencias

```bash
cd ~/1UP_2
source venv/bin/activate
pip install -r server/requirements_server.txt
```

### PASO 7: Verificar CUDA

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Debe mostrar:**
```
CUDA: True
GPU: NVIDIA RTX 4000 Ada Generation
```

### PASO 8: Iniciar servidor

```bash
cd ~/1UP_2/server
source ../venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Verificar que funciona:**
```bash
# En otra terminal SSH
curl http://localhost:8000/health
```

**Debe retornar:**
```json
{"status":"healthy","detector_ready":true,"analyzer_ready":true}
```

---

## 🔧 CONFIGURAR PUERTO PÚBLICO

1. Ve a RunPod Dashboard
2. Selecciona tu pod (`ytoissxrquxq5s`)
3. Ve a "Port Mapping"
4. Añade: `8000:8000`
5. Copia la URL pública (ej: `https://abc123-8000.proxy.runpod.net`)

---

## 🧪 PROBAR DESDE MAC LOCAL

### Actualizar URL del servidor

Edita `client/config_client.yaml`:
```yaml
server:
  url: "https://abc123-8000.proxy.runpod.net"  # URL de RunPod
  timeout: 120
```

### Ejecutar cliente

```bash
cd /Users/jba7790/Desktop/1UP_2
source venv/bin/activate
python client/capture_client.py
```

**Presiona SPACE** para capturar y enviar al servidor.

---

## 🐛 PROBLEMAS COMUNES

### Error: "CUDA not available"
```bash
# Verificar drivers
nvidia-smi

# Si no muestra GPU, verifica en RunPod dashboard que el pod tenga GPU
```

### Error: "HF_TOKEN not found"
```bash
export HF_TOKEN="TU_HF_TOKEN_AQUI"
hf auth login
```

### Error: "Claude API key not found"
```bash
export CLAUDE_API_KEY="tu-api-key"
# Verificar que está en el entorno
echo $CLAUDE_API_KEY
```

### Servidor no responde
```bash
# Verificar que está corriendo
ps aux | grep uvicorn

# Ver logs
tail -f ~/1UP_2/server/logs/*.log
```

### Error: "Module not found"
```bash
# Asegúrate de estar en el venv
source venv/bin/activate

# Reinstalar dependencias
pip install -r server/requirements_server.txt
```

---

## ✅ CHECKLIST FINAL

- [ ] Archivos subidos a RunPod
- [ ] Entorno virtual creado
- [ ] SAM3 instalado
- [ ] Tokens configurados (HF_TOKEN, CLAUDE_API_KEY)
- [ ] Dependencias instaladas
- [ ] CUDA verificado
- [ ] Servidor iniciado
- [ ] Puerto público configurado
- [ ] Cliente local configurado con URL RunPod
- [ ] Prueba exitosa de detección

---

**Última actualización:** 2026-01-10
