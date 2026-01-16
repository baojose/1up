# 📊 Estado Actual del Sistema 1UP

**Fecha:** 2026-01-11  
**Versión:** MVP - Arquitectura Cliente/Servidor  
**Hardware Local:** MacBook Pro 15" 2018 (Intel Core i9)  
**Hardware Servidor:** RunPod GPU (RTX 4000 Ada, 20GB VRAM)  
**Configuración:** Cliente local (captura) + Servidor RunPod (procesamiento GPU)

---

## 🎯 Resumen Ejecutivo

**1UP** es un sistema automático de reconocimiento de objetos para puntos limpios (centros de reciclaje) en Madrid que promueve economía circular. Utiliza **SAM 3** (Segment Anything Model 3) para detectar objetos y **Claude Sonnet 4** para identificarlos y analizarlos.

**🎯 Objetivo:** Objetos funcionales NO van a basura → Segunda vida (1UP 🍄)

**Filosofía actual:** "Detectar TODO, filtrar después con Claude"
- SAM 3 detecta TODO (hipersensible, múltiples prompts)
- Claude decide qué objetos son útiles (filtrado inteligente)
- Solo se generan crops para objetos útiles

**✅ Integración Reolink:** Sistema soporta cámara IP Reolink RLC-810A vía RTSP  
**✅ Arquitectura Cliente/Servidor:** Cliente local captura frames, servidor RunPod procesa con GPU  
**📖 Ver [docs/RUNPOD_SETUP.md](RUNPOD_SETUP.md)** para setup del servidor  
**📖 Ver [docs/TESTING_LOCAL.md](TESTING_LOCAL.md)** para testing local

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **`detector.py`** - SAM 3 Object Detector
   - Detecta objetos usando SAM 3 (Segment Anything Model 3)
   - Usa múltiples prompts para máxima cobertura: `"visual"`, `"container"`, `"object"`, `"furniture"`, `"tool"`
   - `confidence_threshold: 0.001` (MÁXIMA DETECCIÓN)
   - Devuelve: máscaras, bboxes, scores
   - **NO identifica objetos** (solo detecta dónde están)

2. **`analyzer.py`** - Claude Vision Analyzer
   - Analiza objetos usando Claude Sonnet 4
   - Recibe: 1 imagen completa + lista de bboxes (texto)
   - Identifica: nombres, categorías, condiciones, descripciones, precios
   - Filtra objetos útiles (`useful="yes"` o `useful="no"`)
   - Agrupa objetos similares (ej: "Especiero con 7 frascos")

3. **`live_detection.py`** - Pipeline Principal
   - Orquesta SAM 3 y Claude
   - Maneja cámara (Reolink RTSP o USB), captura, validación de calidad
   - Genera crops solo para objetos útiles (después de Claude)
   - Guarda en base de datos JSON

4. **`camera_utils.py`** - Utilidades de Cámara
   - Función `open_camera()` soporta índices USB y URLs RTSP
   - Detección automática de dispositivo
   - Configuración de buffer para RTSP (baja latencia)

5. **`storage_v2.py`** - Gestión de Crops
   - Genera crops/thumbnails para objetos útiles
   - Validación de contenido de crops
   - Estandariza aspect ratio (1:1, cuadrado)

6. **`filters.py`** - Filtros Post-Claude
   - Filtra nombres genéricos
   - Filtra objetos muy grandes (fondo)
   - Centralizado y documentado

7. **`web_app.py`** - Servidor Web Marketplace
   - Flask app en `http://localhost:5001`
   - Muestra objetos detectados en formato marketplace

---

## 🔄 Flujo Completo del Sistema

```
1. Cámara Reolink captura foto automática (4K) O usuario presiona SPACE → Captura foto
   ↓
2. ✅ Validación de calidad (blur detection) → Rechaza imágenes borrosas
   ↓
3. SAM 3 detecta objetos (MÚLTIPLES PROMPTS) → Genera bboxes y máscaras
   - Prompts: "visual", "container", "object", "furniture", "tool"
   - confidence_threshold: 0.001 (MÁXIMA DETECCIÓN)
   - Resultado: 50-150 detecciones RAW
   ↓
4. Claude valida y analiza TODO → 1 imagen + lista de TODOS los bboxes (texto)
   - Input: 1 imagen completa + bboxes en texto
   - Output: JSON con análisis de cada objeto
   - Claude decide: useful="yes" o useful="no"
   - Claude agrupa objetos similares
   ↓
5. Post-filtrado (filters.py) → Filtra nombres genéricos, objetos muy grandes
   ↓
6. Genera crops DESPUÉS de Claude → Thumbnails SOLO para objetos útiles
   - n=1 → obj_001.jpg
   - n=2 → obj_002.jpg
   - Validación de contenido de crops
   ↓
7. Merge objetos similares → Agrupa duplicados (ej: frascos de especias)
   ↓
8. Guarda en base de datos → JSON con metadata
   ↓
9. Web muestra → Flask sirve objetos desde JSON
```

---

## ⚙️ Configuración Actual

### Modo Cliente/Servidor (Recomendado)

**Cliente Local (Mac Intel):**
- Captura frames 4K de Reolink
- Envía al servidor RunPod
- Muestra resultados

**Servidor RunPod (GPU):**
- Procesa con SAM3 CUDA (rápido)
- Analiza con Claude API
- Genera crops 4K
- Retorna resultados JSON

**Configuración:**
- `client/config_client.yaml` - URL servidor RunPod
- `server/config_server.yaml` - Device CUDA, configuración GPU

### Modo Monolítico (Local - Testing)

**⚠️ Para testing local solamente:**
- Todo en Mac Intel (CPU)
- Más lento (30-60s por frame)
- Ver [docs/TESTING_LOCAL.md](TESTING_LOCAL.md)

**📖 Ver [docs/HARDWARE_CONFIG.md](HARDWARE_CONFIG.md)** para:
- Configuración hardware específica
- Troubleshooting

### SAM 3

```yaml
sam3:
  device: "cpu"  # ⚠️ Mac Intel (2018) NO tiene MPS - usar CPU
  # 🔄 Para Apple Silicon (M1/M2/M3): cambiar a "mps" (más rápido)
  text_prompt: ""  # Vacío = detección automática (múltiples prompts)
  enhance_image: true  # CLAHE para objetos oscuros
  confidence_threshold: 0.001  # MÁXIMA DETECCIÓN (en detector.py)
  
  filtering:
    enabled: false  # DESHABILITADO - SAM detecta TODO
    min_area: 50
    max_area_ratio: 0.9
    nms_iou_threshold: 0.9  # Solo duplicados exactos
```

**Optimización para Mac Intel:**
- Procesa imágenes a 720p antes de SAM 3 (previene OOM)
- Escala bboxes/máscaras de vuelta a resolución original (1080p) para crops

### Claude

```yaml
claude:
  model: "claude-sonnet-4-20250514"
  max_tokens: 16000
  temperature: 0  # Determinístico
```

### Cámara

```yaml
camera:
  # ⚠️ Configuración actual (Pruebas - Mac Intel):
  source: "rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub 1080p H.264
  resolution: [1920, 1080]  # Stream sub es 1080p (más estable que 4K HEVC)
  fps: 3  # Reducido para ordenador más lento
  buffer_size: 1  # Buffer para RTSP (baja latencia)
  
  # 🔄 Para volver a 4K (solo Mac Apple Silicon):
  # source: "rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_main"
  # resolution: [3840, 2160]
  
  allow_iphone: true  # Permite iPhone/Continuity Camera
  quality_check:
    enabled: true
    min_sharpness: 20.0  # Rechaza imágenes borrosas
```

**Cámara Reolink RLC-810A:**
- Tipo: Bullet, exterior, PoE, IP66
- **Stream main:** 3840x2160 (4K HEVC) - Producción
- **Stream sub:** 1920x1080 (1080p H.264) - Pruebas ✅ (ACTUAL)
- Protocolo: RTSP (puerto 8554)
- Uso: Producción (24/7, automático)

**Streams disponibles:**
- `h264Preview_01_main` - 4K HEVC (requiere hardware potente, problemas en Mac Intel)
- `h264Preview_01_sub` - 1080p H.264 (más estable, recomendado para pruebas)

---

## 📊 Métricas de Rendimiento

### Detección

- **Detecciones SAM:** 50-150 objetos RAW (depende de escena)
- **Objetos enviados a Claude:** TODOS (sin pre-filtrado)
- **Objetos útiles (Claude):** 10-30 objetos (depende de escena)
- **Objetos finales guardados:** 8-25 objetos (después de merge y filtros)

### Tiempo

- **SAM 3 detección:** 5-15 segundos
- **Claude análisis:** 10-30 segundos (depende de número de objetos)
- **Generación de crops:** 1-3 segundos
- **Total por captura:** 20-50 segundos

### Coste

- **Claude API:** ~$0.003-0.005 por captura
  - 1 imagen (~5,000 tokens input)
  - Análisis de objetos (~8,000-15,000 tokens output)

---

## ✅ Características Implementadas

### Detección

- ✅ SAM 3 con múltiples prompts (máxima cobertura)
- ✅ Detección hipersensible (confidence_threshold: 0.001)
- ✅ Image enhancement (CLAHE) para objetos oscuros
- ✅ Validación de calidad de imagen (blur detection)

### Análisis

- ✅ Claude Sonnet 4 para identificación y análisis
- ✅ 1 imagen + bboxes en texto (eficiente, ~$0.003 por captura)
- ✅ Filtrado inteligente (Claude decide qué es útil)
- ✅ Agrupación de objetos similares (ej: "Especiero con 7 frascos")

### Post-procesamiento

- ✅ Generación de crops solo para objetos útiles
- ✅ **Crops estandarizados: 512x512 píxeles, objeto centrado**
- ✅ Validación de contenido de crops
- ✅ Merge de objetos similares (evita duplicados)
- ✅ Aspect ratio preservado (sin distorsión)

### Visualización

- ✅ **Preview mejorado: auras/máscaras visibles en ventana de detección**
- ✅ **Canvas fijo 1280x720 para evitar pixelación**
- ✅ Labels con fondo semi-transparente para mejor legibilidad
- ✅ Hasta 50 objetos visibles en preview

### Visualización

- ✅ Servidor web Flask (localhost:5001)
- ✅ Formato e-commerce con thumbnails
- ✅ Metadata completa (nombre, categoría, condición, precio)

---

## 🔧 Optimizaciones Aplicadas

1. ✅ **SAM se ejecuta solo una vez** (no dos veces)
2. ✅ **Crops se generan después de Claude** (solo objetos útiles)
3. ✅ **Mapeo simplificado** (n directo: n=1 → obj_001.jpg)
4. ✅ **Filtros centralizados** (módulo `filters.py`)
5. ✅ **Prompt de Claude simplificado** (~50 líneas vs ~600)
6. ✅ **Validación de calidad de imagen** (blur detection)

**Ahorro total:** 6.5-18 segundos por imagen procesada

---

## 🚧 Limitaciones Actuales

### SAM 3

- ❌ **NO hay video tracking** - Modo "image per frame"
  - Cada frame se procesa independientemente
  - No hay `object_id` persistente entre frames
  - No podemos extraer el mejor crop del mismo objeto a lo largo de frames

- ❌ **NO identifica objetos** - Solo detecta dónde están
  - La identificación la hace Claude

### Claude

- ⚠️ **Depende de API externa** - Requiere conexión a internet
- ⚠️ **Coste por captura** - ~$0.003-0.005 (aceptable para MVP)

### Cámara

- ⚠️ **Autofocus temporal** - Solo para cámaras USB externas (eliminar en futuro)
- ⚠️ **Calidad de imagen** - Depende de iluminación y enfoque

---

## 📁 Estructura de Archivos

```
1UP_2/
├── detector.py          # SAM 3 detector
├── analyzer.py          # Claude analyzer
├── live_detection.py    # Pipeline principal
├── storage_v2.py        # Gestión de crops
├── filters.py           # Filtros post-Claude
├── web_app.py          # Servidor web Flask
├── config.yaml          # Configuración central
├── camera_utils.py      # Utilidades de cámara
├── image_quality.py     # Validación de calidad
│
├── database/
│   └── objects.json     # Base de datos JSON
│
├── images/
│   ├── raw/             # Escenas completas
│   └── crops/            # Objetos individuales
│
├── docs/                # Documentación
│   ├── ESTADO_ACTUAL.md  # Este archivo
│   ├── PROCESO_COMPLETO.md
│   ├── SAM3_CURRENT_USAGE.md
│   └── ...
│
└── sam3/                # SAM 3 source code
```

---

## 🎯 Próximos Pasos

### Corto Plazo

1. ✅ **Integración con cámara Reolink** (COMPLETADO)
2. **Captura automática 24/7** (trigger desde Reolink)
3. **Mejora de agrupación** (reducir duplicados)
4. **Validación de crops** (mejorar detección de crops vacíos)

### Medio Plazo

1. **Video tracking con SAM 3** (habilitar object_ids persistentes)
2. **Integración con ecommerce** (Shopify, WooCommerce)
3. **API REST** para subir productos

### Largo Plazo

1. **App móvil** (usuario toma foto → auto-upload)
2. **Sistema automático punto limpio** (cámara → detección → publicación)
3. **Base de datos PostgreSQL** (migrar de JSON)

---

## 📚 Documentación Relacionada

- **[Proceso Completo](PROCESO_COMPLETO.md)** - Flujo end-to-end detallado
- **[SAM 3 Current Usage](SAM3_CURRENT_USAGE.md)** - Detalles técnicos de SAM 3
- **[Getting Started](GETTING_STARTED.md)** - Guía de inicio rápido
- **[Live Detection](LIVE_DETECTION.md)** - Uso de detección en vivo
- **[SAM 3 Config](SAM3_CONFIG.md)** - Configuración de SAM 3
- **[Filtering](FILTERING.md)** - Sistema de filtrado

---

**Última actualización:** 2026-01-10  
**Mantenido por:** Jose (@jba7790)
**Hardware actual:** MacBook Pro 15" 2018 (Intel Core i9)  
**Configuración:** CPU + Stream 1080p H.264 (Pruebas)

