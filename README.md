# 🍄 1UP - Multi-Object Detection for Ecommerce

Sistema de detección automática de múltiples objetos en una foto para generar datos listos para ecommerce.

## 📖 ¿Qué es 1UP?

1UP detecta múltiples objetos en una foto, los analiza con IA, y genera datos listos para subir a plataformas de ecommerce.

**Flujo MVP (OPTIMIZADO):**
1. 📸 Toma una foto (manual por ahora)
2. 🔍 **SAM 3** detecta TODOS los objetos en la imagen (máscaras y bboxes, **sin nombres**) - **Una sola vez**
3. 🤖 **Claude Sonnet 4** valida y analiza objetos (**identifica qué son**: descripción, categoría, condición, precio)
4. ✂️ Genera crops/thumbnails **SOLO para objetos útiles** (después de Claude)
5. 📦 Genera datos listos para ecommerce (JSON/CSV + thumbnails)

**⚠️ Nota:** SAM 3 detecta **dónde** están los objetos, pero **Claude identifica QUÉ son** (nombres, categorías, etc.)

## 🎯 Objetivo Actual: MVP para Ecommerce

**Input**: Una foto con múltiples objetos  
**Output**: 
- Thumbnails de cada objeto detectado
- Descripciones en texto de cada objeto
- Metadata (categoría, condición, precio estimado)
- Formato listo para subir a ecommerce

## 🚀 Inicio Rápido

### Setup (requiere Python 3.12)

```bash
# Instalar Python 3.12 si no lo tienes
brew install python@3.12

# Crear entorno virtual
python3.12 -m venv venv
source venv/bin/activate

# Instalar dependencias
bash setup_venv.sh
```

### Uso Básico

```bash
# Detección en vivo con cámara + análisis Claude (recomendado)
export CLAUDE_API_KEY='sk-ant-api03-...'
./run_live_detection_with_claude.sh

# Detección en vivo sin Claude (solo visual)
./run_live_detection.sh

# Procesar una imagen
python3 main.py --image foto.jpg

# Modo interactivo completo (con Claude)
export CLAUDE_API_KEY='sk-ant-api03-...'
python3 main.py

# Ver objetos en web e-commerce (nuevo!)
./run_web.sh
# Luego abre: http://localhost:5001
# (Puerto 5000 suele estar ocupado por AirPlay en macOS)
```

📖 **Ver [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) para guía completa**  
📖 **Ver [docs/LIVE_DETECTION.md](docs/LIVE_DETECTION.md) para detección en vivo**

## 📁 Estructura del Proyecto

```
1UP_2/
├── README.md                 # Este archivo
├── .cursorrules              # Reglas de desarrollo
├── config.yaml               # Configuración central
│
├── detector.py               # SAM 3 detection (<350 líneas)
├── analyzer.py               # Claude analysis (<350 líneas)
├── main.py                   # Pipeline principal (<350 líneas)
│
├── models/                   # Modelos AI (SAM 3 checkpoints se descargan automáticamente)
│
├── images/                   # Almacenamiento
│   ├── raw/                  # Escenas completas
│   └── crops/                # Objetos individuales (thumbnails)
│
├── database/                 # Base de datos simple
│   └── objects.json          # JSON con metadata
│
├── web_app.py                # Aplicación web Flask (e-commerce)
├── templates/                # Templates HTML
│   └── index.html           # Página principal
├── static/                   # Archivos estáticos
│   └── css/
│       └── style.css        # Estilos e-commerce
│
└── docs/                     # Documentación
    ├── INICIO_RAPIDO.md
    ├── SETUP_VENV.md
    ├── QUICK_TEST.md
    └── ...
```

## 🎯 Roadmap

### ✅ Fase 1: MVP (Actual)
- [x] Detección múltiple de objetos (SAM 3)
- [x] Análisis con Claude
- [x] Generación de crops/thumbnails
- [x] Formato de salida para ecommerce (JSON)
- [x] **Web app e-commerce local** 🎉

### 🔜 Fase 2: Integración
- [ ] Integración con plataformas de ecommerce (Shopify, WooCommerce, etc.)
- [ ] API REST para subir productos
- [ ] Batch processing de múltiples fotos

### 🚀 Fase 3: Escalado
- [ ] App móvil (usuario toma foto → auto-upload)
- [ ] Sistema automático punto limpio (cámara → detección → publicación)

## ⚙️ Stack Tecnológico

- **Detección**: SAM 3 (Segment Anything Model 3) - Real SAM 3 from Meta
  - Text prompts para concept-based detection
  - Open-vocabulary segmentation
  - Install: `git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .`
  - Requires access to checkpoints on HuggingFace
- **Análisis**: Claude Sonnet 4
- **Cámara**: OpenCV (cv2) - Opcional para MVP
- **Base de datos**: JSON files
- **Web**: Flask (aplicación e-commerce local)
- **Config**: YAML

## 📚 Documentación

Toda la documentación está en `docs/`:

- **[Inicio Rápido](docs/GETTING_STARTED.md)** - Setup y primeros pasos
- **[Uso del Sistema](docs/USAGE.md)** - Cómo usar live detection y análisis
- **[Configuración SAM 3](docs/SAM3_CONFIG.md)** - Text prompts, enhancement, parámetros
- **[Sistema de Filtrado](docs/FILTERING.md)** - Pipeline completo de filtrado
- **[Proceso Completo](docs/PROCESO_COMPLETO.md)** - Flujo end-to-end
- **[Detección en Vivo](docs/LIVE_DETECTION.md)** - Guía de uso

Ver `docs/README.md` para índice completo.

## 🐛 Problemas Comunes

### Python 3.14 no compatible

Usa Python 3.12:
```bash
brew install python@3.12
python3.12 -m venv venv
```

### "CLAUDE_API_KEY not set"

```bash
export CLAUDE_API_KEY="sk-ant-api03-xxxxx"
```

O usa el script que lo carga automáticamente:
```bash
./run_live_detection_with_claude.sh
```

### Web app no muestra imágenes

Asegúrate de que:
1. Los objetos están guardados en `database/objects.json`
2. Las imágenes existen en `images/crops/`
3. El servidor puede acceder a los archivos (permisos correctos)

## 📝 Reglas de Desarrollo

- **Máximo 350 líneas por archivo** (sin excepciones)
- **Type hints obligatorios**
- **Error handling obligatorio**
- **Logging (no print())**
- **Configuración en YAML (nunca hardcode)**
- **Multi-plataforma desde día 1**

Ver `.cursorrules` para más detalles.

---

**Developer**: Jose (@jba7790)  
**Location**: Tres Cantos, Madrid  
**Project**: 1UP - Multi-Object Detection for Ecommerce 🍄
