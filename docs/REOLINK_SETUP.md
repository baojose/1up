# 📹 Configuración Cámara Reolink RLC-810A

## ⚠️ Configuración Actual (Pruebas)

**Hardware:** Mac Intel (2018)  
**Stream:** Sub 1080p H.264 (más estable que 4K HEVC)  
**Razón:** Mac Intel tiene problemas decodificando HEVC 4K

**Test completado:**
```python
# Stream sub (1080p H.264) - RECOMENDADO para Mac Intel
url = "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"
cap = cv2.VideoCapture(url)
# ✅ Frame capturado: 1920x1080 (1080p H.264)

# Stream main (4K HEVC) - Solo para Apple Silicon
# url = "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"
# ⚠️ Problemas: Errores HEVC, frames corruptos, muy lento en Mac Intel
```

## 🔧 Configuración en config.yaml (Actual)

**Configuración para Mac Intel (Pruebas):**
```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub 1080p
  resolution: [1920, 1080]  # 1080p H.264 (más estable)
  fps: 3  # Reducido para ordenador más lento
  buffer_size: 1  # Baja latencia

sam3:
  device: "cpu"  # Mac Intel no tiene MPS
```

**Para volver a 4K (solo Mac Apple Silicon):**
```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"  # Stream main 4K
  resolution: [3840, 2160]  # 4K HEVC
  fps: 3  # Reducir FPS para estabilidad
  buffer_size: 1

sam3:
  device: "mps"  # Apple Silicon tiene MPS (más rápido)
```

## 📋 Especificaciones de la Cámara

- **Modelo**: Reolink RLC-810A
- **Tipo**: Bullet, exterior, PoE, IP66
- **Protocolo**: RTSP
- **Puerto**: 8554 (NO usar 554, bloqueado por firewall)
- **Red**: WiFi 5GHz

**Streams disponibles:**
- **Stream main (4K HEVC):** `rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_main`
  - Resolución: 3840x2160 (4K)
  - Codec: HEVC (H.265)
  - Uso: Producción (requiere Mac Apple Silicon)
  - Problemas en Mac Intel: Errores de decodificación HEVC
  
- **Stream sub (1080p H.264):** `rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_sub` ✅ (ACTUAL)
  - Resolución: 1920x1080 (1080p)
  - Codec: H.264 (AAC)
  - Uso: Pruebas/desarrollo (más estable)
  - Ventaja: Mejor compatibilidad, menos errores

## 🚀 Uso

Una vez configurado `config.yaml`, ejecuta:

```bash
export CLAUDE_API_KEY='sk-ant-api03-...'
./run_live_detection_with_claude.sh
```

El sistema:
1. Se conectará automáticamente a la cámara Reolink vía RTSP (stream sub 1080p)
2. Capturará frames en 1080p H.264 (más estable que 4K HEVC)
3. SAM 3 procesará a 720p (previene OOM), luego escalará bboxes/máscaras a 1080p
4. Detectar objetos con SAM 3 (CPU en Mac Intel)
5. Analizar con Claude (1 imagen + bboxes, ~$0.003 por captura)
6. Generar crops estandarizados (512x512, objeto centrado) y guardar en base de datos

## 🔄 Volver a Webcam USB

Si quieres usar una webcam USB en lugar de Reolink:

```yaml
camera:
  source: 1  # Índice de cámara USB (0, 1, 2...)
  resolution: [1920, 1080]
  fps: 5
```

## ⚠️ Notas Importantes

- **Puerto 8554**: Usa siempre el puerto 8554, no 554 (bloqueado por firewall)
- **Buffer size 1**: Importante para baja latencia en detección en vivo
- **Resolución 4K**: Asegúrate de que tu red WiFi soporte el ancho de banda necesario
- **Credenciales**: Las credenciales están en `config.yaml` - NO las subas a Git

## 🐛 Troubleshooting

### Preview pixelado o corrupto

**Causa:** Stream 4K HEVC tiene errores de decodificación en Mac Intel.  
**Solución:** Usar stream sub (1080p H.264):
```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub
  resolution: [1920, 1080]
```

### Frames muy lentos (5-30 segundos)

**Causa:** Stream 4K HEVC o CPU es más lento que MPS.  
**Solución:**
1. Usar stream sub (1080p H.264) en lugar de main (4K HEVC)
2. Reducir FPS: `fps: 2` o `fps: 1`
3. Aceptar que CPU es más lento (normal en Mac Intel)

### Errores HEVC (cu_qp_delta, Could not find ref)

**Causa:** Mac Intel tiene problemas decodificando HEVC 4K.  
**Solución:** Usar stream sub (1080p H.264) que usa H.264 en lugar de HEVC.

### No se conecta a la cámara

1. Verifica que la IP sea correcta: `192.168.1.188`
2. Verifica que el puerto sea 8554 (no 554)
3. Verifica credenciales (usuario: `admin`, contraseña: `Polic!ia1`)
4. Verifica que la cámara esté encendida y conectada a la red
5. Prueba ambos streams (main y sub) para ver cuál funciona mejor

### Error "Failed to open RTSP stream"

1. Prueba la URL directamente con VLC o ffplay:
   ```bash
   # Stream sub (1080p H.264) - RECOMENDADO
   ffplay rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub
   
   # Stream main (4K HEVC) - Solo si funciona bien en tu Mac
   ffplay rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main
   ```
2. Si funciona en VLC pero no en el código, puede ser un problema de codec
3. Verifica que OpenCV tenga soporte RTSP compilado
4. **Recomendación:** Usa stream sub (H.264) si stream main (HEVC) tiene problemas