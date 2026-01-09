# ⚠️ TEMPORAL: Autofocus Inteligente para Cámara Externa USB

## 📋 Estado

**⚠️ ESTO ES TEMPORAL - DEBE ELIMINARSE**

Este módulo es una solución **TEMPORAL** para el dispositivo externo (cámara USB).
Una vez que el proyecto avance (app móvil, punto limpio automático), este módulo debe ser **ELIMINADO**.

---

## 🎯 Propósito

Proporciona autofocus inteligente para cámaras USB externas que tienen problemas de enfoque.

**Cuándo usar:**
- ✅ Desarrollo con cámara USB externa (C270, etc.)
- ✅ Prototipado en Mac/Desktop

**Cuándo NO usar:**
- ❌ App móvil (tiene su propio sistema de cámara)
- ❌ Sistema automático punto limpio (no necesita autofocus manual)
- ❌ Producción en Raspberry Pi (puede tener cámara fija)

---

## 🔧 Cómo Funciona

### Flujo de Autofocus

```
Usuario presiona 'S' para guardar
    ↓
1. Trigger autofocus (toggle on/off)
    ↓
2. Espera 2 segundos (autofocus_delay)
    ↓
3. Captura 5 frames consecutivos
    ↓
4. Calcula nitidez de cada uno (Laplacian Variance)
    ↓
5. Elige el más nítido
    ↓
6. ¿Es > 20.0?
   Sí → Procesa
   No → Reintenta (máx 3 veces)
    ↓
7. Si tras 3 intentos sigue borrosa:
   - Registra error
   - Devuelve mejor intento con warning
```

---

## 📝 Configuración

En `config.yaml`:

```yaml
camera:
  # ⚠️ TEMPORAL: Smart autofocus for external USB camera
  autofocus:
    enabled: false  # Enable intelligent autofocus (TEMPORAL)
    autofocus_delay: 2.0  # Seconds to wait after triggering autofocus
    focus_attempts: 5  # Number of frames to capture to pick the sharpest
    max_autofocus_attempts: 3  # Maximum autofocus attempts if image is blurry
```

---

## 📂 Archivos Relacionados

- `smart_camera.py` - Clase SmartCamera (TEMPORAL)
- `live_detection.py` - Integración del autofocus (línea ~549)
- `config.yaml` - Configuración de autofocus

---

## ⚠️ CUÁNDO ELIMINAR

**Eliminar este módulo cuando:**
1. ✅ Se implemente la app móvil (iOS/Android)
2. ✅ Se implemente el sistema automático de punto limpio
3. ✅ Ya no se use cámara USB externa para desarrollo

**Cómo eliminar:**
1. Eliminar archivo `smart_camera.py`
2. Eliminar código de autofocus en `live_detection.py` (línea ~560-605)
3. Eliminar configuración en `config.yaml` (sección `camera.autofocus`)
4. Eliminar esta documentación

---

## 🎯 Alternativa (cuando se elimine)

En lugar de autofocus inteligente:
- App móvil: Usar API nativa de cámara (auto-focus automático)
- Raspberry Pi: Usar cámara fija con enfoque manual ajustado
- Producción: Cámara profesional con autofocus hardware

---

**Fecha de creación:** 2025-01-02  
**Estado:** TEMPORAL - Marcar para eliminación en futuro

