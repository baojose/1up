# 📖 Uso del Sistema - 1UP

Guía de uso de las funcionalidades principales.

## 🎥 Detección en Vivo

### Iniciar

```bash
./run_live_detection_with_claude.sh
```

O sin Claude:

```bash
./run_live_detection.sh
```

### Controles

- **SPACE** = Detectar objetos en frame actual
  - Congela la imagen
  - Muestra bounding boxes y máscaras
  - Los objetos se muestran con colores diferentes

- **S** = Guardar y analizar con Claude
  - Guarda la escena actual
  - Envía a Claude para análisis (1 imagen + bboxes en texto)
  - Guarda solo objetos "útiles" en la base de datos
  - Genera crops/thumbnails (solo para objetos útiles, después de Claude)

- **C** = Limpiar detecciones
  - Vuelve al feed en vivo
  - Limpia las detecciones actuales

- **Q** o **ESC** = Salir

### Qué Esperar

1. **Primera ejecución**: 1-2 minutos cargando SAM 3 (descarga checkpoints si necesario)
2. **Cámara**: Usa la cámara externa automáticamente
3. **Detección**: Al presionar SPACE, SAM 3 detecta objetos (5-15 segundos)
4. **Análisis**: Al presionar S, Claude analiza los objetos detectados

## 🌐 Servidor Web

### Iniciar

```bash
./run_web.sh
```

Abre: http://localhost:5001

### Funcionalidad

- Muestra objetos detectados en formato e-commerce
- Incluye thumbnails, descripciones, categorías, condiciones
- Se actualiza automáticamente cuando guardas nuevas escenas

## 🧪 Testing

### Test de Detección Simple

```bash
./run_test_detection.sh
```

Prueba solo la detección visual sin Claude.

### Test con Imagen Estática

```bash
./run_test_image.sh images/raw/scene_camCAM0_TIMESTAMP.jpg
```

Prueba SAM 3 sobre una imagen específica (útil cuando no tienes cámara).

### Test de Integración Claude

```bash
./run_test_batch.sh
```

Verifica que la integración con Claude funciona correctamente.

## 📊 Flujo Completo

1. **Captura**: Presiona SPACE para detectar objetos
2. **Revisión**: Verifica que los objetos detectados sean correctos
3. **Análisis**: Presiona S para guardar y analizar con Claude
4. **Resultado**: Objetos útiles se guardan en `database/objects.json`
5. **Visualización**: Abre `./run_web.sh` para ver los resultados

## ⚙️ Configuración

### Ajustar Detección

Edita `config.yaml`:

```yaml
sam3:
  text_prompt: ""  # Concepto específico (ej: "bag", "shoes")
  filtering:
    enabled: true
    min_area: 2000  # Tamaño mínimo de objetos
    nms_iou_threshold: 0.5  # Agresividad de NMS
  # NOTA: min_area_for_analysis fue eliminado - SAM envía TODO a Claude
```

### Ajustar Análisis Claude

```yaml
claude:
  max_tokens: 16000  # Tamaño de respuesta
  temperature: 0.7  # Creatividad (0-1)
```

Ver **[Configuración SAM 3](SAM3_CONFIG.md)** para más detalles.

## 💾 Datos Generados

### Estructura

```
images/
  raw/
    scene_camCAM0_TIMESTAMP.jpg          # Imagen original
    scene_camCAM0_TIMESTAMP_viz.jpg      # Con visualización
  crops/
    TIMESTAMP/
      obj_000.jpg                         # Crop objeto 1
      obj_001.jpg                         # Crop objeto 2
      ...

database/
  objects.json                            # Base de datos JSON
```

### Formato de Base de Datos

```json
[
  {
    "id": "obj_2025-12-02_11-30-45_001",
    "timestamp": "2025-12-02_11-30-45",
    "name": "Botas marrones",
    "category": "clothing",
    "condition": "good",
    "description": "Botas de cuero marrón...",
    "estimated_value": "€20-30",
    "thumbnail": "images/crops/2025-12-02_11-30-45/obj_000.jpg",
    "bbox": [100, 200, 300, 400],
    "confidence": 0.95
  }
]
```

## 🔍 Troubleshooting

### No se detectan objetos

1. Verifica iluminación (objetos deben ser visibles)
2. Ajusta `min_area` en `config.yaml` (reduce para detectar objetos más pequeños)
3. Usa `text_prompt` para conceptos específicos

### Demasiados objetos detectados

1. Aumenta `min_area` en `config.yaml`
2. Aumenta `nms_iou_threshold` (más agresivo)
3. Ajusta filtros de SAM en `config.yaml` (aunque están deshabilitados por defecto)

### Claude no analiza correctamente

1. Verifica API key: `echo $CLAUDE_API_KEY`
2. Revisa logs para errores de API
3. Aumenta `max_tokens` si la respuesta se trunca

### Cámara no funciona

1. Ejecuta `./run_list_cameras.sh` para ver cámaras disponibles
2. Edita `config.yaml` con el índice correcto
3. Cierra otras apps que usen la cámara

Ver **[Inicio Rápido](GETTING_STARTED.md)** para más troubleshooting.

