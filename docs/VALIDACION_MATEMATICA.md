# 🔬 Validación Matemática: Calidad de Imagen y Correspondencia Thumbnail-Contenido

## 📋 Problema

Cuando la cámara está desenfocada, el sistema:
1. Detecta objetos incorrectamente (SAM genera bboxes imprecisos)
2. Claude identifica incorrectamente (no puede ver detalles)
3. Los thumbnails no coinciden con el contenido identificado

## ✅ Soluciones Matemáticas Objetivas Implementadas

### 1. Detección de Blur (Laplacian Variance)

**Métrica matemática:** `Var(Laplacian(I))`

**Cómo funciona:**
- Aplica filtro Laplacian a la imagen (detecta bordes)
- Calcula la varianza de los valores del Laplacian
- Imágenes nítidas → alta varianza (muchos bordes definidos)
- Imágenes borrosas → baja varianza (bordes suaves)

**Umbrales:**
- `>100`: Buena nitidez ✅
- `50-100`: Nitidez aceptable ⚠️
- `<50`: Borrosa (rechazar) ❌

**Implementación:**
```python
def calculate_sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)
```

**Uso:**
- Se valida ANTES de procesar con SAM
- Si nitidez < umbral → imagen rechazada automáticamente
- Usuario recibe mensaje: "Imagen demasiado borrosa, enfoca la cámara"

---

### 2. Validación Thumbnail ↔ Contenido

**Problema:** Asegurar que el thumbnail generado corresponde al objeto que Claude identificó.

**Métricas matemáticas combinadas:**

#### Métrica 1: Correlación Espacial (matchTemplate)
- Compara thumbnail con región original usando bbox
- Rango: 0-1 (1 = idéntico, 0 = diferente)
- Peso: 50%

#### Métrica 2: Matching de Características (ORB)
- Detecta puntos clave en thumbnail y región original
- Cuenta matches entre características
- Ratio: matches / características detectadas
- Peso: 30%

#### Métrica 3: Correlación de Histogramas
- Compara distribución de intensidades
- Rango: 0-1 (1 = misma distribución, 0 = diferente)
- Peso: 20%

**Score combinado:**
```
score = 0.5 * correlación_espacial + 0.3 * match_ratio + 0.2 * hist_correlation
```

**Umbral:** Score > 0.3 = válido ✅

**Si score < 0.3:** Se registra warning (thumbnail puede no corresponder)

---

### 3. Validación de Calidad de Crop

**Métricas:**
1. **Nitidez del crop:** `calculate_sharpness_score(crop)`
   - Umbral mínimo: 20.0 (más bajo que imagen completa)
   - Warning si < 20.0

2. **Ratio de contenido:**
   - Porcentaje de píxeles que NO son fondo blanco
   - Umbral mínimo: 20% del área
   - Asegura que hay contenido real, no solo fondo

---

## 🔧 Configuración

En `config.yaml`:

```yaml
camera:
  quality_check:
    enabled: true  # Activar validación de calidad
    min_sharpness: 50.0  # Umbral mínimo de nitidez
```

---

## 📊 Flujo de Validación

```
1. Usuario captura imagen (SPACE)
   ↓
2. ✅ Validación de nitidez (Laplacian Variance)
   - Si nitidez < 50 → RECHAZAR, mensaje al usuario
   - Si nitidez ≥ 50 → CONTINUAR
   ↓
3. SAM detecta objetos
   ↓
4. Claude identifica objetos
   ↓
5. Generación de crops
   ↓
6. ✅ Validación de cada crop:
   - Nitidez del crop
   - Contenido vs fondo blanco
   - Correspondencia thumbnail-bbox (histogramas)
   ↓
7. Si validación falla → Warning (pero se guarda si es útil)
```

---

## 🎯 Ventajas

1. **Objetivo:** Métricas matemáticas, no subjetivas
2. **Rápido:** Validación en <10ms por imagen
3. **Efectivo:** Rechaza imágenes borrosas antes de procesamiento costoso
4. **Trazable:** Logs muestran scores exactos para debugging

---

## 📝 Logs de Ejemplo

```
🔍 Validating image quality (blur detection)...
✅ Calidad de imagen aceptable: nitidez=124.5

⚠️  n=3: Crop muy borroso (nitidez=15.2 < 20.0)
⚠️  n=5: Posible discrepancia thumbnail-bbox (correlación histograma=0.42)
```

---

## 🔗 Archivos Relacionados

- `image_quality.py`: Módulo de validaciones matemáticas
- `live_detection.py`: Integración de validación antes de SAM
- `storage_v2.py`: Validación de crops generados
- `config.yaml`: Configuración de umbrales

