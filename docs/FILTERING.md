# 🔄 Sistema de Filtrado - 1UP

Documentación completa del sistema de filtrado multi-etapa que elimina fragmentos, duplicados y objetos no útiles.

**⚡ ACTUALIZACIÓN:** Los filtros post-Claude están centralizados en el módulo `filters.py`

**⚠️ IMPORTANTE:** Los filtros PRE-Claude están **DESHABILITADOS por defecto** en `config.yaml`:
```yaml
sam3:
  filtering:
    enabled: false  # SAM detecta TODO, Claude decide
```

**Filosofía actual:** "Detectar TODO, filtrar después con Claude"
- SAM 3 detecta TODO (hipersensible, múltiples prompts)
- Claude decide qué objetos son útiles (filtrado inteligente)
- Solo se generan crops para objetos útiles

## 📊 Visión General

**Filtros PRE-Claude:** En `detector.py` (pipeline de filtrado de SAM) - **DESHABILITADOS por defecto**
**Filtros POST-Claude:** En `filters.py` (módulo centralizado) - **ACTIVOS**

Cuando `filtering.enabled: false`, SAM 3 envía TODAS las detecciones a Claude sin pre-filtrado.

```
SAM Detection (raw)
    ↓
1. Smart Filtering (área, aspect ratio, visibilidad)
    ↓
2. Sort by Area (grandes primero)
    ↓
3. Filter Contained Boxes (elimina bboxes dentro de otros)
    ↓
4. Keep Largest in Group (overlap >80%, solo el más grande)
    ↓
5. Sort by Confidence (mayor confianza primero)
    ↓
6. NMS (overlap >30%, solo el de mayor confianza) - Ajustado para permitir más objetos superpuestos
    ↓
7. Final Sort by Area (grandes primero)
    ↓
Final Detections (solo objetos completos)
```

## 🎯 Paso 1: Smart Filtering

**¿Qué hace?**
- Filtra por **área mínima/máxima** (elimina ruido y escena completa)
- Filtra por **aspect ratio** (elimina objetos muy altos/anchos)
- Filtra por **visibilidad** (elimina objetos fuera de imagen)

**Configuración:**
```yaml
sam3:
  filtering:
    enabled: false  # Por defecto DESHABILITADO - SAM detecta TODO
    min_area: 50    # Mínimo muy bajo (solo elimina ruido extremo)
    max_area_ratio: 0.9   # Máx 90% de imagen (permite objetos grandes)
    min_aspect_ratio: 0.01 # Permite objetos muy alargados
    max_aspect_ratio: 50.0 # Permite objetos muy anchos
    nms_iou_threshold: 0.9  # Solo elimina duplicados exactos
```

**⚠️ NOTA:** Con `enabled: false`, estos parámetros NO se aplican. SAM envía TODO a Claude.

**Impacto:** 54 detecciones → 36 detecciones (elimina basura obvia)

## 🎯 Paso 2: Sort by Area

Ordena las detecciones de mayor a menor área. Necesario para que `filter_contained_boxes` sea eficiente (solo compara con bboxes más grandes).

## 🎯 Paso 3: Filter Contained Boxes

**¿Qué hace?**
- Elimina bboxes **completamente dentro** de otros más grandes
- Compara cada bbox con todos los más grandes (ya ordenados)

**Ejemplo:**
```
Bbox grande: Cómic completo (135,331 px²)
Bbox pequeño: Título del cómic (42,473 px²) ← DENTRO del grande
→ Elimina el pequeño
```

**Impacto:** 36 detecciones → 30 detecciones (elimina fragmentos internos)

## 🎯 Paso 4: Keep Largest in Group

**¿Qué hace?**
- Agrupa bboxes con **overlap muy alto (>80%)**
- De cada grupo, mantiene **solo el más grande**
- Elimina fragmentos internos de objetos grandes

**Ejemplo:**
```
Grupo superpuesto:
- Bbox #1: Cómic completo (IoU=0.85 con #2)
- Bbox #2: Fragmento esquina (IoU=0.85 con #1)
→ Mantiene solo #1 (más grande)
```

**Impacto:** 30 detecciones → 25 detecciones (elimina fragmentos superpuestos)

## 🎯 Paso 5: Sort by Confidence

Ordena las detecciones de mayor a menor confianza. Necesario para NMS (siguiente paso).

## 🎯 Paso 6: NMS (Non-Maximum Suppression)

**¿Qué hace?**
- Elimina duplicados con **overlap >30%** (ajustado para permitir más objetos superpuestos)
- Mantiene la detección con **mayor confianza**

**Configuración:**
```yaml
sam3:
  filtering:
    nms_iou_threshold: 0.3  # Menos agresivo (era 0.5, ajustado para permitir más objetos superpuestos)
```

**Impacto:** 25 detecciones → 22 detecciones (permite más overlap, mejor para objetos superpuestos)

## 🎯 Paso 7: Final Sort by Area

Ordena las detecciones finales de mayor a menor área para consistencia.

## 📈 Impacto Total

**ANTES (sin filtrado):**
- 54 detecciones
- 1 gigante (toda la imagen)
- 16 microscópicas (< 2,000 px²)
- Duplicados (mismo objeto 2-3 veces)
- Fragmentos (título dentro de cómic)

**DESPUÉS (con filtrado):**
- ~18-23 detecciones
- Solo objetos útiles
- Sin duplicados
- Sin ruido
- Sin fragmentos

**Ahorro en costes:** ~60-70% 💰

## 🔧 Configuración Avanzada

### Ajuste Fino

**Si filtra demasiado** (pierde objetos útiles):
```yaml
sam3:
  filtering:
    min_area: 1500           # Menos estricto
    max_area_ratio: 0.5      # Permite objetos más grandes
    nms_iou_threshold: 0.6   # Menos agresivo
```

**Si filtra poco** (sigue detectando basura):
```yaml
sam3:
  filtering:
    min_area: 3000           # Más estricto
    max_area_ratio: 0.3      # Más estricto
    nms_iou_threshold: 0.4   # Más agresivo
```

## 🧪 Testing

### Verificar Filtrado

```bash
# Ejecutar detección
python3 main.py

# Logs esperados:
# 📊 Filtering: 54 → 36 (removed 18)
# 📦 Filtered contained boxes: 36 → 30 (removed 6)
# 🎯 Kept largest in groups: 30 → 25 (removed 5)
# 🔍 NMS filtering: 25 → 20 (removed 5)
# ✅ Final result: 20 quality objects
```

### Desactivar Temporalmente

```yaml
sam3:
  filtering:
    enabled: false  # Vuelve al comportamiento original
```

## 📝 Notas Técnicas

- **Orden crítico**: Cada paso depende del anterior
- **Eficiencia**: Ordenar por área antes de `filter_contained` reduce complejidad
- **Thresholds**: 80% para Keep Largest (solo fragmentos internos), 50% para NMS (elimina duplicados)
- **Compatibilidad**: Retrocompatible, puede desactivarse sin romper código

## 🔄 Filtros Post-Claude (Centralizados)

**Archivo:** `filters.py`

Los filtros que se aplican DESPUÉS de Claude están centralizados en el módulo `filters.py`:

1. **`filter_generic_names()`**: Filtra nombres genéricos ("superficie", "fragmento", etc.)
2. **`filter_by_size()`**: Filtra objetos muy grandes (fondo)
3. **`filter_useful_objects()`**: Filtro completo que aplica todos los filtros post-Claude

**Ventajas:**
- Código centralizado y documentado
- Fácil de mantener y modificar
- Reutilizable en diferentes partes del pipeline

Ver **[Optimizaciones](OPTIMIZACIONES.md)** para más detalles.

## 🎯 Resultado Final

De **54 detecciones con fragmentos y duplicados** a **~18-23 objetos completos y únicos**, con ahorro de ~60-70% en costes de API.

