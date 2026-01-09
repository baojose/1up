# 🎯 Configuración SAM 3 - 1UP

Guía completa para configurar SAM 3 (Segment Anything Model 3) para detección de objetos.

## ⚠️ IMPORTANTE: SAM 3 NO identifica objetos

**SAM 3 es un modelo de SEGMENTACIÓN, NO de reconocimiento:**
- ✅ Detecta **dónde** están los objetos (máscaras y bounding boxes)
- ❌ **NO** identifica **qué** son los objetos (no da nombres, categorías, descripciones)
- La **identificación** (nombres, categorías) la hace **Claude** en el paso siguiente

**Flujo:**
1. SAM 3 detecta objetos → bboxes y máscaras (sin nombres)
2. Claude analiza cada región → nombres, categorías, descripciones

## 🚀 Concept-Based Detection

SAM 3 soporta **text prompts** para detección basada en conceptos. Puedes buscar objetos específicos usando descripciones en texto.

### Text Prompts

```yaml
sam3:
  text_prompt: ""  # Vacío = detecta todos los objetos
```

**Ejemplos:**
- `text_prompt: "bag"` - Solo detecta bolsas
- `text_prompt: "shoes, boots"` - Detecta zapatos y botas
- `text_prompt: "electronics"` - Detecta dispositivos electrónicos
- `text_prompt: ""` - Detecta todos los objetos (modo automático)

### Cuándo Usar Text Prompts

**✅ Usar cuando:**
- Buscas objetos específicos (ej: solo "shoes" en un punto limpio de ropa)
- Quieres reducir falsos positivos
- Tienes un catálogo específico

**❌ No usar cuando:**
- Quieres detectar todos los objetos posibles
- No sabes qué objetos habrá en la escena
- Estás probando el sistema por primera vez

## 🎯 Confidence Threshold

El `confidence_threshold` controla qué tan sensible es SAM 3 para detectar objetos.

### Configuración Actual

```yaml
sam3:
  # confidence_threshold se configura en detector.py
  # Valor actual: 0.001 (MÁXIMA DETECCIÓN)
```

**Valores:**
- `0.5` (default): Solo objetos muy claros y visibles
- `0.10`: Detecta objetos pequeños, oscuros y parcialmente ocluidos
- `0.05`: Detecta objetos superpuestos, muy pequeños y oscuros
- `0.001` (actual): MÁXIMA DETECCIÓN - detecta TODO (filtrado por Claude después)

**Impacto:**
- **Más bajo (0.05)**: Detecta más objetos, incluyendo superpuestos, pero puede generar más falsos positivos
- **Más alto (0.5)**: Solo objetos muy claros, menos falsos positivos pero puede perder objetos importantes

### Cuándo Ajustar

**Bajar (0.05 → 0.03):**
- Si faltan objetos superpuestos
- Si faltan objetos muy pequeños
- Si faltan objetos oscuros

**Subir (0.05 → 0.10):**
- Si hay demasiados falsos positivos
- Si detecta demasiados fragmentos

## 🖼️ Image Enhancement (CLAHE)

CLAHE (Contrast Limited Adaptive Histogram Equalization) mejora la detección de objetos oscuros.

### Configuración

```yaml
sam3:
  enhance_image: false  # true = activa CLAHE
```

**Impacto esperado:** +20-30% detección de objetos oscuros

**Cuándo activar:**
- Escenas con poca iluminación
- Objetos oscuros sobre fondo oscuro
- Mejora general de contraste

**Cuándo desactivar:**
- Escenas bien iluminadas
- Objetos claros sobre fondo claro
- Si causa falsos positivos

## 🔧 Parámetros de Detección

### ⚠️ Pre-filtering DESHABILITADO por defecto

**Configuración actual:**
```yaml
sam3:
  filtering:
    enabled: false  # DESHABILITADO - SAM detecta TODO
```

**Filosofía actual:** "Detectar TODO, filtrar después con Claude"

**Razón:** Claude es más inteligente para filtrar que reglas matemáticas simples. SAM 3 detecta TODO (hipersensible, múltiples prompts), y Claude decide qué objetos son útiles usando `useful="yes"` o `useful="no"`.

**Múltiples prompts:** SAM 3 usa múltiples prompts (`"visual"`, `"container"`, `"object"`, `"furniture"`, `"tool"`) para máxima cobertura.

## 📊 Parámetros de Filtrado

Ver **[Sistema de Filtrado](FILTERING.md)** para detalles completos.

### Resumen Rápido

```yaml
sam3:
  filtering:
    enabled: false  # DESHABILITADO por defecto - SAM detecta TODO
    min_area: 50    # Mínimo muy bajo (solo elimina ruido extremo)
    max_area_ratio: 0.9    # Máx 90% de imagen (permite objetos grandes)
    min_aspect_ratio: 0.01  # Permite objetos muy alargados
    max_aspect_ratio: 50.0  # Permite objetos muy anchos
    nms_iou_threshold: 0.9  # Solo elimina duplicados exactos
```

**⚠️ NOTA:** Con `enabled: false`, estos parámetros NO se aplican. SAM envía TODO a Claude.

## 🎯 Ejemplos de Configuración

### Configuración Conservadora (menos objetos, más precisión)

```yaml
sam3:
  text_prompt: ""
  enhance_image: false
  # min_area_for_analysis eliminado - SAM envía TODO a Claude
  filtering:
    enabled: true
    min_area: 3000
    max_area_ratio: 0.3
    nms_iou_threshold: 0.4
```

**Resultado:** ~10-15 objetos, muy precisos, sin fragmentos

### Configuración Agresiva (más objetos, puede tener fragmentos)

```yaml
sam3:
  text_prompt: ""
  enhance_image: true
  # min_area_for_analysis eliminado - SAM envía TODO a Claude
  filtering:
    enabled: true
    min_area: 1000
    max_area_ratio: 0.5
    nms_iou_threshold: 0.3
```

**Resultado:** ~25-35 objetos, puede incluir fragmentos pequeños

### Configuración Específica (solo ciertos objetos)

```yaml
sam3:
  text_prompt: "bag, backpack, suitcase"
  enhance_image: false
  # min_area_for_analysis eliminado - SAM envía TODO a Claude
  filtering:
    enabled: true
    min_area: 2000
    max_area_ratio: 0.4
    nms_iou_threshold: 0.3
```

**Resultado:** Solo detecta bolsas, mochilas y maletas

## 🔍 Troubleshooting

### No detecta objetos

1. **Verifica iluminación** - Objetos deben ser claramente visibles
2. **Reduce `min_area`** - Puede estar filtrando objetos pequeños
3. **Usa `text_prompt`** - Especifica conceptos si buscas algo concreto
4. **Activa `enhance_image`** - Mejora detección en escenas oscuras

### Detecta demasiados objetos

1. **Aumenta `min_area`** - Filtra objetos más pequeños
2. **Aumenta `nms_iou_threshold`** - Más agresivo contra duplicados
3. **Ajusta filtros de SAM** (aunque están deshabilitados por defecto)
4. **Usa `text_prompt`** - Limita a conceptos específicos

### Detecta fragmentos

1. **Reduce `nms_iou_threshold`** - Más agresivo (ej: 0.4)
2. **Aumenta `min_area`** - Solo objetos más grandes
3. **Verifica `filtering.enabled: true`** - Asegúrate que está activo

### Detección muy lenta

1. **Usa MPS** (Mac) o **CUDA** (NVIDIA) en lugar de CPU
2. **Reduce `text_prompt`** - Detección automática es más rápida
3. **Ajusta `filtering.enabled`** - Activa filtros de SAM si necesitas reducir detecciones

## 📚 Más Información

- **[Sistema de Filtrado](FILTERING.md)** - Detalles del pipeline de filtrado
- **[Proceso Completo](PROCESO_COMPLETO.md)** - Flujo end-to-end
- **[Uso del Sistema](USAGE.md)** - Cómo usar las funcionalidades

