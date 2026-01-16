# 🔧 Troubleshooting Reolink RTSP

## ⚠️ Configuración Actual (Pruebas)

**Hardware:** Mac Intel (2018)  
**Stream:** Sub 1080p H.264 (más estable que 4K HEVC)  
**Razón:** Mac Intel tiene problemas decodificando HEVC 4K

## Problema: Stream Timeout / Cuelgues / Errores HEVC

### Síntomas
- El sistema se queda colgado esperando frames
- Warnings: `Stream timeout triggered after 30091 ms`
- Errores HEVC: `cu_qp_delta outside valid range`, `Could not find ref with POC`
- Preview pixelado/corrupto
- `cap.read()` bloquea 5-30 segundos

### Causa (Mac Intel)
1. **Errores HEVC:** Mac Intel tiene problemas decodificando HEVC 4K (stream main)
2. **Stream 4K muy pesado:** Cada frame 4K requiere ~8-12MB sin comprimir
3. **Red WiFi lenta:** RTSP sobre WiFi puede ser inestable con 4K
4. **Buffer acumulado:** OpenCV acumula demasiados frames antiguos

### Solución 1: Usar Stream Sub (1080p H.264) ⬅️ **RECOMENDADO**

**Configuración actual (Mac Intel):**
```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub 1080p
  resolution: [1920, 1080]  # ⬇️ Stream sub es 1080p H.264 (más estable)
  fps: 3  # Reducido para ordenador más lento
  buffer_size: 1
```

**Ventajas:**
- ✅ Más rápido y estable
- ✅ Menor latencia
- ✅ Menos problemas de red
- ✅ Suficiente para detección SAM 3

**Desventajas:**
- ⚠️ Menos detalle en objetos pequeños

### Solución 2: Volver a 4K (Solo Mac Apple Silicon)

Si tienes **Mac Apple Silicon (M1/M2/M3)**, puedes usar stream main (4K HEVC):

**Configuración para Apple Silicon:**
```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"  # Stream main 4K
  resolution: [3840, 2160]  # 4K HEVC
  fps: 3  # Reducir FPS para estabilidad
  buffer_size: 1

sam3:
  device: "mps"  # Apple Silicon tiene MPS (más rápido)
```

**⚠️ IMPORTANTE:** Mac Intel NO puede usar esta configuración (errores HEVC).

📖 Ver [docs/HARDWARE_CONFIG.md](HARDWARE_CONFIG.md) para detalles completos.

### Solución 3: Ajustar Timeout y Buffer

Si necesitas mantener 4K, puedes:

1. **Aumentar buffer_size** (pero aumenta latencia):
```yaml
buffer_size: 3  # Más buffer, menos bloqueos, más latencia
```

2. **Reducir FPS**:
```yaml
fps: 2  # Menos frames por segundo, más tiempo entre frames
```

### Solución 4: Verificar Conexión de Red

El problema puede ser de red, no de software:

```bash
# Verificar que la cámara responde
ping 192.168.1.188

# Verificar puerto RTSP
telnet 192.168.1.188 8554

# Probar con VLC u otro reproductor
# Si VLC también se cuelga, es problema de red/cámara
```

### Recomendación Inicial (Mac Intel)

**Para Mac Intel, usa siempre stream sub (1080p H.264):**

```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub
  resolution: [1920, 1080]  # 1080p H.264 - más estable
  fps: 3  # Reducido para ordenador más lento
  buffer_size: 1

sam3:
  device: "cpu"  # Mac Intel no tiene MPS
```

**Ventajas:**
- ✅ Sin errores HEVC (usa H.264)
- ✅ Frames más rápidos (1-3 segundos vs 5-30 segundos)
- ✅ Preview claro (sin pixelación)
- ✅ Suficiente para SAM 3 y crops de e-commerce

### Volver a 4K (Solo Apple Silicon)

Si cambias a **Mac Apple Silicon (M1/M2/M3)**, puedes volver a 4K:

```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"  # Stream main
  resolution: [3840, 2160]  # 4K HEVC - mayor detalle
  fps: 3  # Reducir FPS para estabilidad
  buffer_size: 1

sam3:
  device: "mps"  # Apple Silicon tiene MPS (más rápido)
```

📖 Ver [docs/HARDWARE_CONFIG.md](HARDWARE_CONFIG.md) para guía completa.

## Verificación de Estado

El sistema mostrará:
```
✅ RTSP stream opened: 1920x1080  # O 3840x2160
✅ Reading frames: 1920x1080
```

Si ves warnings de timeout repetidos, reduce la resolución o FPS.
