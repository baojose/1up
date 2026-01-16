# 📤 Subir Archivos Python a RunPod - Método Simple

Como `scp` está fallando, vamos a crear los archivos directamente en RunPod usando Python.

## Método: Usar Python para crear archivos desde base64

**En RunPod, ejecuta este comando para cada archivo:**

### 1. detector.py

```bash
python3 << 'ENDPYTHON'
import base64
import sys

# Base64 del archivo detector.py
data = '''[AQUÍ VA EL BASE64 DEL ARCHIVO]'''

with open('/root/1UP_2/detector.py', 'wb') as f:
    f.write(base64.b64decode(data))
    
print('✅ detector.py creado')
ENDPYTHON
```

**PERO** este método es muy largo. Mejor solución:

## Solución Alternativa: Usar `rsync` en lugar de `scp`

**En tu Mac, ejecuta:**

```bash
cd /Users/jba7790/Desktop/1UP_2

# Probar rsync (más robusto que scp)
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  detector.py \
  ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/
```

Si `rsync` funciona, sube todos los archivos:

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  detector.py analyzer.py filters.py image_quality.py camera_utils.py storage_v2.py storage.py \
  ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/

rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  server/api.py \
  ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/server/

rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  client/capture_client.py \
  ytoissxrquxq5s-6441116d@ssh.runpod.io:~/1UP_2/client/
```

---

## Si rsync también falla: Crear archivos manualmente

Si nada funciona, podemos crear los archivos directamente en RunPod copiando el contenido. Pero esto es muy tedioso.

**Mejor opción:** Usar el método de `rsync` primero.
