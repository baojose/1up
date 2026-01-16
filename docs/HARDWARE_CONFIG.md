# 🖥️ Configuración de Hardware - Mac Intel vs Apple Silicon

## ⚠️ IMPORTANTE: Arquitectura Cliente/Servidor

**Hardware local:** MacBook Pro 15" 2018 (Intel Core i9)  
**Hardware servidor:** RunPod GPU (RTX 4000 Ada, 20GB VRAM)  
**Configuración:** Cliente captura local → Servidor procesa con GPU

**Solución:** Separar captura (local) de procesamiento (servidor GPU)
- ✅ Cliente local: Captura frames 4K (sin procesamiento pesado)
- ✅ Servidor GPU: Procesa con SAM3 CUDA (rápido, 5-15s)
- ✅ Diseño 4K mantenido (crops de máxima calidad)

**Para testing local (monolítico):**
- Mac Intel CPU + Stream 1080p H.264
- Más lento (30-60s por frame) pero funciona
- Ver [docs/TESTING_LOCAL.md](TESTING_LOCAL.md)

---

## 📋 Configuración Actual (Mac Intel)

### config.yaml (Pruebas)

```yaml
camera:
  # Stream sub 1080p H.264 (más estable que 4K HEVC en Mac Intel)
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_sub"
  resolution: [1920, 1080]  # Stream sub es 1080p
  fps: 3

sam3:
  device: "cpu"  # Mac Intel NO tiene MPS - usar CPU
```

**Por qué:**
- ✅ Mac Intel (2018) no tiene MPS (solo Apple Silicon lo tiene)
- ✅ Stream sub (1080p H.264) es más estable que main (4K HEVC)
- ✅ CPU funciona bien aunque sea más lento
- ✅ Crops 1080p son suficientes para e-commerce (se estandarizan a 512x512)

---

## 🔄 Volver a Configuración Anterior (4K + MPS)

Si cambias a un **Mac Apple Silicon** (M1/M2/M3) o quieres usar **4K**:

### Opción 1: Mac Apple Silicon (M1/M2/M3)

```yaml
camera:
  # Stream main 4K HEVC (requiere Mac Apple Silicon con MPS)
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"
  resolution: [3840, 2160]  # 4K
  fps: 3  # Reducir FPS para estabilidad en 4K
  buffer_size: 1

sam3:
  device: "mps"  # Apple Silicon tiene MPS (más rápido que CPU)
```

**Ventajas:**
- ✅ MPS es más rápido que CPU
- ✅ 4K para crops de mayor calidad
- ✅ Mejor validación de objetos

**Desventajas:**
- ⚠️ Requiere Mac Apple Silicon (M1/M2/M3)
- ⚠️ MPS tiene límite de memoria (~6.8 GB)
- ⚠️ 4K HEVC puede causar problemas de decodificación

### Opción 2: Mac Intel con 4K (NO RECOMENDADO)

```yaml
camera:
  source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"
  resolution: [3840, 2160]  # 4K HEVC
  fps: 2  # Reducir FPS aún más
  buffer_size: 1

sam3:
  device: "cpu"  # Mac Intel siempre usa CPU
```

**Problemas:**
- ❌ Errores de decodificación HEVC en Mac Intel
- ❌ Frames muy lentos (5-30 segundos)
- ❌ Preview pixelado/corrupto
- ❌ SAM 3 con CPU en 4K es muy lento

**No recomendado** - Mejor usar stream sub (1080p).

---

## 🔍 Detección Automática de Hardware

El sistema detecta automáticamente el tipo de Mac:

- **Apple Silicon (M1/M2/M3):** `platform.processor() == 'arm'` o `'arm64' in platform.machine()`
- **Intel:** `platform.processor() == 'i386'` o `'x86_64' in platform.machine()`

**Código:**
```python
import platform
is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.machine().lower()
```

**En detector.py:**
- Si `device="mps"` pero es Mac Intel → Automáticamente usa CPU
- Si `device="mps"` y es Apple Silicon → Usa MPS
- Si `device="cpu"` → Siempre usa CPU (funciona en ambos)

---

## 📊 Comparación de Configuraciones

| Configuración | Mac Intel | Apple Silicon | Estabilidad | Velocidad | Calidad Crops |
|--------------|-----------|---------------|-------------|-----------|---------------|
| **CPU + 1080p** (Actual) | ✅ Funciona | ✅ Funciona | ✅ Muy estable | ⚠️ Lenta | ✅ Suficiente |
| **MPS + 1080p** | ❌ No disponible | ✅ Funciona | ✅ Estable | ✅ Rápida | ✅ Suficiente |
| **CPU + 4K** | ⚠️ Errores HEVC | ✅ Funciona | ❌ Inestable | ❌ Muy lenta | ✅ Máxima |
| **MPS + 4K** | ❌ No disponible | ⚠️ OOM | ⚠️ Inestable | ✅ Rápida | ✅ Máxima |

**Recomendación actual (Mac Intel):** CPU + 1080p

---

## 🎯 Streams Reolink Disponibles

La Reolink RLC-810A tiene múltiples streams:

### Stream Main (4K HEVC)
```
rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_main
```
- **Resolución:** 3840x2160 (4K)
- **Codec:** HEVC (H.265)
- **Uso:** Producción (requiere hardware potente)
- **Problemas en Mac Intel:** Errores de decodificación HEVC

### Stream Sub (1080p H.264) ⬅️ **RECOMENDADO para pruebas**
```
rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_sub
```
- **Resolución:** 1920x1080 (1080p)
- **Codec:** H.264 (AAC)
- **Uso:** Pruebas/desarrollo (más estable)
- **Ventajas:** Mejor compatibilidad, menos errores

---

## 🚀 Cómo Cambiar de Configuración

### Desde 1080p (Pruebas) → 4K (Producción)

1. **Verificar hardware:**
   ```bash
   python3 -c "import platform; print(f'Processor: {platform.processor()}'); print(f'Machine: {platform.machine()}')"
   ```

2. **Si es Apple Silicon**, edita `config.yaml`:
   ```yaml
   camera:
     source: "rtsp://admin:Polic!ia1@192.168.1.188:8554/h264Preview_01_main"
     resolution: [3840, 2160]  # 4K
     fps: 3
   
   sam3:
     device: "mps"  # Cambiar de "cpu" a "mps"
   ```

3. **Si es Mac Intel**, mantener 1080p (4K no funciona bien)

### Desde CPU → MPS (Solo Apple Silicon)

1. Edita `config.yaml`:
   ```yaml
   sam3:
     device: "mps"  # Cambiar de "cpu" a "mps"
   ```

2. El sistema detectará automáticamente si MPS está disponible

---

## 💡 Optimizaciones Aplicadas para Mac Intel

### 1. Reducción de Resolución para SAM 3

**Problema:** Procesar 4K directamente causa OOM (out of memory).  
**Solución:** Reducir a 720p antes de SAM 3, luego escalar bboxes/máscaras de vuelta a 1080p.

```python
# detector.py - Línea ~129
max_sam3_dimension = 720  # Reducido de 1008 a 720 para CPU
# Procesa a 720p, luego escala bboxes/máscaras a resolución original
```

### 2. Preview Escalado

**Problema:** Mostrar 1080p completo causa pixelación.  
**Solución:** Canvas fijo 1280x720 con frame centrado.

```python
# live_detection.py
target_preview_width = 1280
target_preview_height = 720
# Frame centrado en canvas negro (previene pixelación)
```

### 3. Crops Estandarizados

**Todos los crops:** 512x512 píxeles, objeto centrado.  
**Ventaja:** Tamaño estándar para e-commerce, sin importar resolución original.

---

## 🐛 Troubleshooting

### "MPS out of memory" en Mac Intel

**Causa:** Mac Intel no tiene MPS, pero el código intenta usarlo.  
**Solución:** Cambiar `sam3.device: "cpu"` en `config.yaml`.

### Preview pixelado

**Causa:** Stream 4K HEVC tiene errores de decodificación.  
**Solución:** Usar stream sub (1080p H.264).

### Frames muy lentos

**Causa:** CPU es más lento que MPS, o stream 4K es muy pesado.  
**Solución:** 
- Reducir FPS: `fps: 2` o `fps: 1`
- Usar stream sub (1080p)
- Aceptar que CPU es más lento (normal)

### Errores HEVC

**Causa:** Mac Intel tiene problemas decodificando HEVC 4K.  
**Solución:** Usar stream sub (1080p H.264) en lugar de main (4K HEVC).

---

## 📚 Documentación Relacionada

- **[Estado Actual](ESTADO_ACTUAL.md)** - Estado completo del sistema
- **[Reolink Setup](REOLINK_SETUP.md)** - Configuración de cámara
- **[Reolink Troubleshooting](REOLINK_TROUBLESHOOTING.md)** - Solución de problemas
- **[SAM 3 Config](SAM3_CONFIG.md)** - Configuración de SAM 3

---

**Última actualización:** 2026-01-10  
**Configuración actual:** Mac Intel + CPU + Stream 1080p (Pruebas)
