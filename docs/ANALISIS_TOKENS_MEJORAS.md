# 📊 Análisis de Tokens: Mejoras Propuestas

## 🎯 Objetivo
Identificar qué mejoras mantienen el número de tokens casi igual con Claude.

## 📋 Análisis por Mejora

### ✅ Mejora 1: Aplicación Precisa de Máscara
**Impacto en tokens: 0 tokens adicionales**
- ✅ **Ya implementado** en `storage.py`
- Solo procesamiento de imagen (no afecta prompt)
- No añade texto al prompt de Claude
- **Conclusión**: Mantiene tokens iguales

---

### 📝 Mejora 2: Explicitar Uso de Máscara en Prompt
**Impacto en tokens: ~50-80 tokens adicionales**

**Texto propuesto:**
```
🧠 IMPORTANTE:
- El objeto en esta imagen fue segmentado con una máscara precisa.
- Solo debes analizar lo que se ve dentro de este crop (fondo blanco = fuera del objeto).
- Si el objeto está parcialmente visible, describe lo que sea claramente identificable.
- No asumas contexto adicional fuera de la imagen.
```

**Análisis:**
- Añade ~50-80 tokens de texto al prompt
- Se añade UNA VEZ por llamada API (no por objeto)
- **Impacto total**: +50-80 tokens por captura (insignificante)
- **Conclusión**: Aumenta tokens mínimamente (~1-2% del prompt actual)

---

### 📐 Mejora 3: Enviar Datos Auxiliares
**Impacto en tokens: ~30-50 tokens por objeto**

**Datos propuestos:**
```
📐 Datos del objeto:
- Tamaño del objeto: X píxeles (~2.3% del total de la imagen original)
- Posición aproximada en la imagen original: [x, y, ancho, alto]
- Parte visible: máscara aplicada, objeto parcialmente oculto
```

**Análisis:**
- Ya incluimos tamaño y posición en el prompt actual
- Solo añadiría "Parte visible: máscara aplicada" (~10 tokens por objeto)
- Si hay 10 objetos: +100 tokens total
- **Conclusión**: Aumenta tokens moderadamente (~2-3% del prompt actual)

---

### 🔍 Mejora 4: Validar Cobertura de Máscara
**Impacto en tokens: 0-20 tokens adicionales (condicional)**

**Lógica:**
```python
coverage_ratio = mask_area / bbox_area
if coverage_ratio < 0.5:
    # Añadir nota al prompt: "objeto parcialmente visible"
    # +20 tokens solo si hay problema
```

**Análisis:**
- Solo añade tokens si hay problema de cobertura
- En la mayoría de casos: 0 tokens adicionales
- Si hay problema: +20 tokens por objeto afectado
- **Conclusión**: Mantiene tokens iguales en casos normales, mínimo aumento si hay problemas

---

### 🧠 Mejora 5: Usar CLIP Embeddings
**Impacto en tokens: ~20-30 tokens adicionales (opcional)**

**Texto propuesto:**
```
"Este objeto fue segmentado con SAM3 y su embedding visual sugiere que puede ser un 'kettlebell'. Confírmalo con lo que ves visualmente."
```

**Análisis:**
- Requiere procesamiento adicional (CLIP)
- Añade ~20-30 tokens por objeto como "hint"
- Si hay 10 objetos: +200-300 tokens total
- **Conclusión**: Aumenta tokens moderadamente (~3-5% del prompt actual)

---

### 🔄 Mejora 6: Ciclo de Feedback Interactivo
**Impacto en tokens: +100% tokens (múltiples llamadas API)**

**Análisis:**
- Requiere múltiples llamadas API
- Primera llamada: tokens normales
- Segunda llamada (zoom/hint): tokens adicionales
- **Conclusión**: Duplica o triplica tokens (NO recomendado para mantener tokens iguales)

---

## 📊 Resumen: Mejoras que Mantienen Tokens Casi Iguales

### ✅ **Mantienen tokens iguales (0 tokens adicionales)**
1. **Mejora 1**: Aplicación de máscara (ya implementado)
2. **Mejora 4**: Validación de cobertura (0 tokens en casos normales)

### 📈 **Aumento mínimo (<100 tokens total, <2% del prompt)**
3. **Mejora 2**: Explicitar uso de máscara (+50-80 tokens, una vez por captura)

### 📊 **Aumento moderado (100-300 tokens, 2-5% del prompt)**
4. **Mejora 3**: Datos auxiliares (+30-50 tokens por objeto)
5. **Mejora 5**: CLIP embeddings (+20-30 tokens por objeto)

### ❌ **Aumento significativo (NO recomendado)**
6. **Mejora 6**: Ciclo de feedback (duplica/triplica tokens)

---

## 🎯 Recomendación: Implementación por Fases

### **Fase 1: Sin aumento de tokens (implementar YA)**
- ✅ Mejora 1: Ya implementado
- ✅ Mejora 4: Validar cobertura (solo lógica, sin tokens adicionales en casos normales)

### **Fase 2: Aumento mínimo (<2%)**
- 📝 Mejora 2: Explicitar uso de máscara (+50-80 tokens, impacto insignificante)

### **Fase 3: Aumento moderado (si es necesario)**
- 📐 Mejora 3: Datos auxiliares (solo si mejora 2 no es suficiente)
- 🧠 Mejora 5: CLIP embeddings (solo para casos problemáticos)

### **Fase 4: NO implementar**
- ❌ Mejora 6: Ciclo de feedback (demasiado costoso en tokens)

---

## 💡 Conclusión

**Para mantener tokens casi iguales:**
- ✅ **Mejora 1**: Ya implementado (0 tokens)
- ✅ **Mejora 4**: Validación de cobertura (0 tokens en casos normales)
- ✅ **Mejora 2**: Explicitar máscara (+50-80 tokens, <1% del prompt actual)

**Total aumento recomendado: <100 tokens (<2% del prompt actual)**

Esto es **insignificante** comparado con el costo de la imagen (~$0.003 por captura).

