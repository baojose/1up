# 📋 Uso Actual de SAM 3 en 1UP

## 1️⃣ Imports

```python
# detector.py líneas 15-22
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.error("SAM 3 not installed...")
```

## 2️⃣ Inicialización

```python
# detector.py líneas 71-95
# Build SAM 3 model (IMAGE MODE, NO VIDEO)
self.model = build_sam3_image_model()

# Move to device
device_obj = torch.device(actual_device)
self.model = self.model.to(device_obj)

# Initialize processor
self.processor = Sam3Processor(
    self.model, 
    device=actual_device,
    confidence_threshold=0.001  # MÁXIMA DETECCIÓN
)
```

**⚠️ IMPORTANTE**: Usamos `build_sam3_image_model()` → **IMAGE MODE**
- ❌ NO usamos `build_sam3_video_predictor()` → VIDEO MODE (con tracking)

## 3️⃣ Llamada de Detección

```python
# detector.py líneas 135-160
# Step 1: Set image
inference_state = self.processor.set_image(pil_image)

# Step 2: Detect with text prompt(s)
if text_prompt:
    output = self.processor.set_text_prompt(
        state=inference_state, 
        prompt=text_prompt
    )
else:
    # Multi-prompt detection
    prompts = ["visual", "container", "object", "furniture", "tool"]
    for prompt in prompts:
        output = self.processor.set_text_prompt(
            state=inference_state, 
            prompt=prompt
        )
        all_detections.append(output)

# Step 3: Extract results
masks = output.get("masks")      # Tensor [N, 1, H, W]
boxes = output.get("boxes")      # Tensor [N, 4] (x0, y0, x1, y1)
scores = output.get("scores")    # Tensor [N] (confidence scores)
```

## 4️⃣ Estructura de Datos que Devuelve SAM 3

### Output de `processor.set_text_prompt()`

```python
output = {
    "masks": torch.Tensor,    # Shape: [N, 1, H, W] - Máscaras booleanas/float
    "boxes": torch.Tensor,    # Shape: [N, 4] - Bboxes [x0, y0, x1, y1]
    "scores": torch.Tensor,   # Shape: [N] - Confidence scores [0.0-1.0]
    # ❌ NO HAY "object_id" o "instance_id"
    # ❌ NO HAY tracking entre frames
}
```

### Ejemplo Concreto

Si detectamos 5 objetos en una imagen:

```python
output = {
    "masks": tensor([
        [[[True, False, True, ...], ...]],   # Objeto 0: máscara [1, H, W]
        [[[False, True, False, ...], ...]],  # Objeto 1
        [[[True, True, False, ...], ...]],    # Objeto 2
        [[[False, False, True, ...], ...]],   # Objeto 3
        [[[True, True, True, ...], ...]]      # Objeto 4
    ]),  # Shape: [5, 1, H, W]
    
    "boxes": tensor([
        [100, 200, 300, 400],  # Objeto 0: [x0, y0, x1, y1]
        [500, 100, 700, 300],  # Objeto 1
        [50, 50, 150, 150],    # Objeto 2
        [800, 400, 1000, 600], # Objeto 3
        [200, 500, 400, 700]   # Objeto 4
    ]),  # Shape: [5, 4]
    
    "scores": tensor([0.95, 0.87, 0.72, 0.65, 0.58])  # Shape: [5]
}
```

**❌ NO hay `object_id` persistente** - Solo índices locales (0, 1, 2, 3, 4)

## 5️⃣ Conversión a Nuestro Formato

```python
# detector.py líneas 207-256
detections = []

for i in range(num_objects):
    # Extract mask
    mask = masks[i, 0]  # Shape: [H, W]
    mask_np = mask.cpu().numpy()
    if mask_np.dtype != bool:
        mask_np = (mask_np > 0.5).astype(bool)
    
    # Extract box [x0, y0, x1, y1] → [x, y, w, h]
    box = boxes[i].cpu().numpy()
    x0, y0, x1, y1 = box
    x, y = int(x0), int(y0)
    w, h = int(x1 - x0), int(y1 - y0)
    bbox = [x, y, w, h]
    
    # Extract score
    score = float(scores[i].item())
    
    # Calculate area
    area = int(np.sum(mask_np))
    
    detection = {
        'id': i,                    # ❌ ID LOCAL (0, 1, 2...) - NO es tracking ID
        'original_index': i,        # Índice antes de sorting/filtering
        'bbox': bbox,               # [x, y, w, h]
        'mask': mask_np,            # Boolean numpy array [H, W]
        'confidence': float(score), # Score de SAM 3
        'area': area,               # Píxeles en máscara
        'coverage_ratio': float     # area / bbox_area
    }
    detections.append(detection)
```

## 6️⃣ ¿Video Tracking Funciona?

### ❌ NO - Estamos en Modo "IMAGE PER FRAME"

**Evidencia en el código:**

1. **Inicialización** (línea 72):
   ```python
   self.model = build_sam3_image_model()  # ← IMAGE MODE
   ```
   ❌ NO usamos: `build_sam3_video_predictor()`  # ← VIDEO MODE (con tracking)

2. **Procesamiento** (líneas 135-160):
   - Cada frame se procesa **INDEPENDIENTEMENTE**
   - `processor.set_image(pil_image)` → procesa 1 imagen
   - No hay secuencia de frames
   - No hay estado de tracking entre frames

3. **Output**:
   - ❌ NO hay `object_id` persistente entre frames
   - ❌ NO hay tracking automático
   - ❌ NO podemos saber que "objeto 0 en frame 1" = "objeto 2 en frame 2"

### Ejemplo del Problema

```
Frame 1 (t=0):
  Objeto físico "pesa" → ID local: 0
  Objeto físico "plancha" → ID local: 1

Frame 2 (t=1):
  Objeto físico "pesa" → ID local: 2  ❌ DIFERENTE ID
  Objeto físico "plancha" → ID local: 0  ❌ DIFERENTE ID

Frame 3 (t=2):
  Objeto físico "pesa" → ID local: 1  ❌ DIFERENTE ID
  Objeto físico "plancha" → ID local: 3  ❌ DIFERENTE ID
```

**Resultado**: No podemos extraer el mejor crop del mismo objeto a lo largo de frames.

## 7️⃣ ¿Qué Falta para Habilitar Tracking?

Para habilitar video tracking necesitaríamos:

### 1. Cambiar a VIDEO MODE

```python
# ACTUAL (IMAGE MODE):
from sam3.model_builder import build_sam3_image_model
self.model = build_sam3_image_model()

# NECESARIO (VIDEO MODE):
from sam3.model_builder import build_sam3_video_predictor
video_predictor = build_sam3_video_predictor()
```

### 2. Procesar Secuencia de Frames

```python
# ACTUAL: Frames independientes
for frame in frames:
    output = processor.set_text_prompt(state, prompt)
    # Cada frame = IDs nuevos (0, 1, 2...)

# NECESARIO: Secuencia de frames
response = video_predictor.handle_request(
    prompt="visual",
    resource_path=video_path,  # o lista de frames
    frame_index=0
)
# SAM 3 devuelve object_ids consistentes entre frames
```

### 3. Output con Object IDs

Con video mode, SAM 3 devuelve:
- `object_ids`: IDs persistentes entre frames
- Tracking automático del mismo objeto físico
- Podemos extraer el mejor crop del mismo objeto a lo largo de frames

## 8️⃣ Resumen

| Aspecto | Estado Actual | Con Video Tracking |
|---------|---------------|-------------------|
| **Modo** | IMAGE MODE | VIDEO MODE |
| **Inicialización** | `build_sam3_image_model()` | `build_sam3_video_predictor()` |
| **Procesamiento** | Frames independientes | Secuencia de frames |
| **Object IDs** | ❌ Solo IDs locales (0,1,2...) | ✅ IDs persistentes entre frames |
| **Tracking** | ❌ No | ✅ Sí |
| **Mejor Crop** | ❌ No podemos extraer | ✅ Sí, del mismo objeto |

## 9️⃣ Conclusión

**Actualmente:**
- ✅ SAM 3 funciona correctamente en IMAGE MODE
- ✅ Detecta objetos con máscaras, bboxes y scores
- ❌ NO hay tracking entre frames
- ❌ NO hay object_ids persistentes
- ❌ NO podemos extraer el mejor crop del mismo objeto

**Para habilitar tracking:**
- Cambiar a VIDEO MODE (`build_sam3_video_predictor()`)
- Procesar secuencia de frames (no frames independientes)
- Usar object_ids persistentes para extraer mejor crop


