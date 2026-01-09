# 📚 Documentación - 1UP

Documentación completa del proyecto 1UP.

## 📊 Estado Actual

- **[Estado Actual del Sistema](ESTADO_ACTUAL.md)** - Resumen ejecutivo completo ⭐⭐⭐

## 🚀 Empezar

- **[Inicio Rápido](GETTING_STARTED.md)** - Setup y primeros pasos ⭐
- **[Uso del Sistema](USAGE.md)** - Cómo usar live detection y análisis ⭐

## 🎯 Guías Principales

- **[Detección en Vivo](LIVE_DETECTION.md)** - Identificación visual con cámara
- **[Proceso Completo](PROCESO_COMPLETO.md)** - Flujo end-to-end desde cámara hasta web ⭐

## ⚙️ Configuración

- **[Configuración SAM 3](SAM3_CONFIG.md)** - Text prompts, enhancement, parámetros ⭐
- **[Sistema de Filtrado](FILTERING.md)** - Pipeline completo de filtrado multi-etapa ⭐

## ⚡ Optimizaciones

- **[Optimizaciones Aplicadas](OPTIMIZACIONES.md)** - Mejoras de rendimiento y eficiencia ⭐

## 🔧 Troubleshooting

- **[Problema Técnico: Crops Ocluidos](PROBLEMA_TECNICO.md)** - Análisis del problema de objetos parcialmente ocultos
- **[Validación Matemática](VALIDACION_MATEMATICA.md)** - Detección de blur y validación thumbnail-contenido ⭐
- **[⚠️ TEMPORAL: Autofocus Inteligente](AUTOFOCUS_TEMPORAL.md)** - Autofocus para cámara USB externa (eliminar en futuro) ⚠️

## 🎯 Objetivo del Proyecto

**MVP Actual**: Detectar múltiples objetos en una foto y generar datos listos para ecommerce.

**Filosofía:** "Detectar TODO, filtrar después con Claude"
- SAM 3 detecta TODO (hipersensible, múltiples prompts)
- Claude decide qué objetos son útiles (filtrado inteligente)
- Solo se generan crops para objetos útiles

**Flujo (OPTIMIZADO):**
1. 📸 Toma una foto
2. 🔍 SAM 3 detecta todos los objetos (múltiples prompts, máxima cobertura)
3. 🤖 Claude analiza cada objeto (1 imagen + bboxes en texto)
4. ✂️ Genera crops/thumbnails (solo para objetos útiles, después de Claude)
5. 📦 Genera datos para ecommerce (JSON + thumbnails)

**Roadmap:**
- ✅ Fase 1: MVP (actual)
- 🔜 Fase 2: Integración con ecommerce
- 🚀 Fase 3: App móvil y sistema automático punto limpio
