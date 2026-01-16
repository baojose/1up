# 📚 Documentación - 1UP

Documentación completa del proyecto 1UP.

## 📊 Estado Actual

- **[Estado Actual del Sistema](ESTADO_ACTUAL.md)** - Resumen ejecutivo completo ⭐⭐⭐

## 🚀 Empezar

- **[Inicio Rápido](GETTING_STARTED.md)** - Setup y primeros pasos ⭐
- **[Uso del Sistema](USAGE.md)** - Cómo usar live detection y análisis ⭐
- **[Setup RunPod](RUNPOD_SETUP.md)** - Configurar servidor GPU para procesamiento ⭐
- **[Testing Local](TESTING_LOCAL.md)** - Probar sistema localmente antes de RunPod ⭐

## 🎯 Guías Principales

- **[Detección en Vivo](LIVE_DETECTION.md)** - Identificación visual con cámara
- **[Proceso Completo](PROCESO_COMPLETO.md)** - Flujo end-to-end desde cámara hasta web ⭐

## ⚙️ Configuración

- **[Configuración SAM 3](SAM3_CONFIG.md)** - Text prompts, enhancement, parámetros ⭐
- **[Sistema de Filtrado](FILTERING.md)** - Pipeline completo de filtrado multi-etapa ⭐

## 🔧 Troubleshooting y Configuración

- **[Configuración de Hardware](HARDWARE_CONFIG.md)** - Arquitectura cliente/servidor, hardware ⭐
- **[Setup Reolink](REOLINK_SETUP.md)** - Configuración de cámara Reolink RLC-810A ⭐
- **[Troubleshooting Reolink](REOLINK_TROUBLESHOOTING.md)** - Solución de problemas RTSP/HEVC ⭐
- **[Testing RunPod](TESTING_RUNPOD.md)** - Plan de pruebas para servidor GPU ⭐
- **[Problema Técnico: Crops Ocluidos](PROBLEMA_TECNICO.md)** - Análisis del problema de objetos parcialmente ocultos
- **[Validación Matemática](VALIDACION_MATEMATICA.md)** - Detección de blur y validación thumbnail-contenido ⭐
- **[⚠️ TEMPORAL: Autofocus Inteligente](AUTOFOCUS_TEMPORAL.md)** - Autofocus para cámara USB externa (eliminar en futuro) ⚠️

## 🎯 Objetivo del Proyecto

**MVP Actual**: Detectar múltiples objetos en una foto y generar datos listos para ecommerce.

**Filosofía:** "Detectar TODO, filtrar después con Claude"
- SAM 3 detecta TODO (hipersensible, múltiples prompts)
- Claude decide qué objetos son útiles (filtrado inteligente)
- Solo se generan crops para objetos útiles

**Flujo (Cliente/Servidor):**
1. 📸 Cliente captura frame 4K de Reolink
2. 📤 Cliente envía frame al servidor RunPod
3. 🔍 Servidor detecta objetos con SAM3 GPU (múltiples prompts, máxima cobertura)
4. 🤖 Servidor analiza con Claude (1 imagen + bboxes en texto)
5. ✂️ Servidor genera crops 4K (solo para objetos útiles)
6. 📥 Cliente recibe resultados (JSON + paths de crops)
7. 📦 Datos listos para ecommerce (JSON + thumbnails)

**Roadmap:**
- ✅ Fase 1: MVP (actual)
- 🔜 Fase 2: Integración con ecommerce
- 🚀 Fase 3: App móvil y sistema automático punto limpio
