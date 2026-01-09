# 🎥 Detección en Vivo - Identificación Visual de Objetos

## ¿Qué hace?

Muestra la cámara en tiempo real y detecta objetos visualmente cuando presionas una tecla.

**Perfecto para:**
- Ver qué objetos detecta la cámara
- Probar diferentes configuraciones
- Identificar objetos en tiempo real

## Cómo usar

### Opción 1: Script automático con Claude (Recomendado)

```bash
# Configura API key primero
export CLAUDE_API_KEY='sk-ant-api03-...'

# Ejecuta
./run_live_detection_with_claude.sh
```

### Opción 2: Script básico (solo detección visual)

```bash
./run_live_detection.sh
```

### Opción 3: Manualmente

```bash
source venv/bin/activate
export CLAUDE_API_KEY='sk-ant-api03-...'  # Opcional, para análisis
python3 live_detection.py
```

## Controles

| Tecla | Acción |
|-------|--------|
| **SPACE** | Detectar objetos en el frame actual |
| **S** | Guardar y analizar con Claude (requiere CLAUDE_API_KEY) |
| **L** | Listar todos los objetos detectados en consola |
| **C** | Limpiar detecciones (ocultar bounding boxes) |
| **Q** | Salir |

## Flujo de uso

1. **Abre la cámara**: Se muestra la vista en vivo
2. **Apunta a objetos**: Coloca objetos claramente visibles
3. **Presiona SPACE**: Ejecuta detección (tarda 5-15 segundos)
4. **Ve los resultados**: Aparecen bounding boxes con los objetos detectados (frozen sobre la foto)
5. **Presiona A** (opcional): Analiza objetos con Claude para identificarlos (requiere CLAUDE_API_KEY)
6. **Presiona L**: Ver lista completa de objetos en consola
7. **Presiona C**: Limpia detecciones y vuelve al video en vivo

## Ejemplo de uso

```
1. Ejecuta: ./run_live_detection.sh
2. Apunta la cámara a una mesa con varios objetos
3. Presiona SPACE
4. Espera 5-15 segundos
5. Verás bounding boxes verdes/amarillos/naranjas alrededor de cada objeto (frozen)
6. En la consola verás: "Detected X objects"
7. Presiona A para analizar con Claude (opcional, requiere API key)
8. Presiona L para ver lista completa con nombres, categorías, condiciones
9. Presiona C para limpiar y volver al video en vivo
```

## Reconocimiento de objetos

Si tienes `CLAUDE_API_KEY` configurada:

1. **Presiona SPACE** para detectar objetos
2. **Presiona A** para analizar con Claude
3. Cada objeto será analizado y recibirá:
   - **Nombre específico** (ej: "Silla de oficina roja")
   - **Categoría** (furniture, electronics, etc.)
   - **Condición** (excellent/good/fair/poor)
   - **Descripción detallada**
   - **Valor estimado** (opcional)
4. **Presiona L** para ver la lista completa en consola
5. Los nombres aparecerán sobre los bounding boxes en la imagen

## Guardado de Escenas (Tecla S)

Cuando presionas **S**, se guarda automáticamente:

1. **Imagen completa** (`images/raw/scene_YYYYMMDD_HHMMSS.jpg`)
2. **Visualización con contornos** (`images/raw/scene_YYYYMMDD_HHMMSS_viz.jpg`)
3. **Crops individuales** (`images/crops/YYYYMMDD_HHMMSS/obj_XXX.jpg`)
4. **Metadata JSON** (`images/raw/scene_YYYYMMDD_HHMMSS_meta.json`)

La metadata incluye:
- IDs de cámara, usuario y sistema (preparado para futuro multi-cámara/usuario)
- Rutas a todas las imágenes
- Información de cada detección (bbox, confidence, área)
- Relación entre objetos y sus crops

**Nota**: En el futuro, Claude eliminará crops e imágenes no reconocibles automáticamente.

## Colores de los contornos

- **Cian/Verde/Amarillo**: Diferentes objetos detectados
- **Contornos brillantes**: Con blending tipo "screen" para ver la imagen original

## Tips

- **Buena iluminación**: Mejora la detección
- **Objetos claros**: Coloca objetos con buen contraste
- **Varios objetos**: Prueba con 3-5 objetos diferentes
- **Paciencia**: La detección tarda 5-15 segundos (normal)

## Problemas comunes

### "No objects detected"

- Mejora la iluminación
- Acerca más los objetos a la cámara
- Asegúrate de que los objetos sean claramente visibles
- Ajusta `min_mask_region_area` en `config.yaml` (reduce el valor)

### "Detection muy lenta"

Es normal. SAM 3 tarda 5-15 segundos por detección. Si quieres más velocidad:
- Usa `device: "cpu"` en `config.yaml` (más lento pero más compatible)
- Usa `text_prompt` en `config.yaml` para buscar conceptos específicos (más rápido que detección automática)

### "Cámara no funciona"

Ver [Inicio Rápido](GETTING_STARTED.md) para detectar cámaras.

