# 🍄 1UP - Sistema Automático de Reconocimiento de Objetos para Puntos Limpios

Sistema automático de reconocimiento de objetos para puntos limpios (centros de reciclaje) en Madrid que promueve economía circular.

## 📖 ¿Qué es 1UP?

1UP es un sistema que detecta automáticamente objetos funcionales depositados en puntos limpios, los analiza con IA, y los publica en un marketplace para darles segunda vida.

**🎯 Objetivo:** Objetos funcionales NO van a basura → Segunda vida (1UP 🍄)

**Flujo Completo del Sistema:**
1. 👤 Usuario deposita objeto funcional en zona "AÚN FUNCIONA" del punto limpio
2. 📹 **Cámara Reolink RLC-810A** (exterior, 24/7) captura foto automática (1080p para pruebas, 4K para producción)
3. 🔍 **SAM 3** detecta TODOS los objetos en la imagen (máscaras y bboxes) - **Una sola vez**
4. ✂️ Sistema genera crops individuales estandarizados (512x512, objeto centrado)
5. 🤖 **Claude Sonnet 4** analiza 1 imagen completa + lista de bboxes (eficiente, ~$0.003 por captura):
   - Identifica objeto (nombre específico)
   - Evalúa condición (excellent/good/fair/poor)
   - Estima precio
   - Decide si es útil (useful="yes/no")
6. 📦 Crops útiles se suben a website/marketplace
7. 👥 Personas reservan y recogen objetos gratis

**⚠️ Nota:** SAM 3 detecta **dónde** están los objetos, pero **Claude identifica QUÉ son** (nombres, categorías, etc.)

## 🎯 Estado Actual: MVP Funcional (Pruebas)

**Hardware actual:** Mac Intel (2018) + CPU + Stream 1080p  
**Configuración:** Optimizada para pruebas en ordenador más lento  
**Ver:** [docs/HARDWARE_CONFIG.md](docs/HARDWARE_CONFIG.md) para volver a 4K/MPS si es necesario

**Input**: Foto automática de cámara Reolink (1080p para pruebas, 4K para producción) o manual  
**Output**: 
- Thumbnails estandarizados (512x512, objeto centrado) de cada objeto útil detectado
- Descripciones en texto de cada objeto
- Metadata (categoría, condición, precio estimado)
- Formato listo para marketplace/web

## 🚀 Inicio Rápido

### Setup (requiere Python 3.12.10)

**⚠️ IMPORTANTE:** PyTorch no soporta Python 3.14. El proyecto requiere **Python 3.12.10**.

```bash
# Instalar Python 3.12.10 desde python.org (recomendado)
# Descarga: https://www.python.org/downloads/release/python-31210/
# Mac Intel: python-3.12.10-macos11.pkg
# Mac M1/M2: python-3.12.10-macos11-arm64.pkg

# Verificar instalación
python3.12 --version  # Debería mostrar: Python 3.12.10

# Crear entorno virtual
python3.12 -m venv venv
source venv/bin/activate

# Instalar dependencias
bash setup_venv.sh
```

📖 **Ver [docs/PYTHON_SETUP.md](docs/PYTHON_SETUP.md) para guía detallada de instalación**

### Uso Básico

```bash
# Detección en vivo con cámara Reolink + análisis Claude (recomendado)
export CLAUDE_API_KEY='sk-ant-api03-...'
./run_live_detection_with_claude.sh

# Detección en vivo sin Claude (solo visual)
./run_live_detection.sh

# Procesar una imagen estática
python3 main.py --image foto.jpg

# Modo interactivo completo (con Claude)
export CLAUDE_API_KEY='sk-ant-api03-...'
python3 main.py

# Ver objetos en web marketplace
./run_web.sh
# Luego abre: http://localhost:5001
# (Puerto 5000 suele estar ocupado por AirPlay en macOS)
```

### Configuración Cámara Reolink

**Configuración actual (Pruebas - Mac Intel):**
```yaml
camera:
  source: "rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_sub"  # Stream sub 1080p H.264
  resolution: [1920, 1080]  # 1080p (más estable que 4K HEVC en Mac Intel)
  fps: 3
  buffer_size: 1

sam3:
  device: "cpu"  # Mac Intel no tiene MPS
```

**Para volver a 4K (solo Mac Apple Silicon):**
```yaml
camera:
  source: "rtsp://admin:PASSWORD@192.168.1.188:8554/h264Preview_01_main"  # Stream main 4K HEVC
  resolution: [3840, 2160]  # 4K
  fps: 3
  buffer_size: 1

sam3:
  device: "mps"  # Apple Silicon tiene MPS (más rápido)
```

**Nota:** El sistema también funciona con webcams USB (índice numérico) para desarrollo.  
📖 Ver [docs/HARDWARE_CONFIG.md](docs/HARDWARE_CONFIG.md) para detalles completos.

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
- [x] Análisis con Claude Sonnet 4
- [x] Generación de crops/thumbnails
- [x] Formato de salida para marketplace (JSON)
- [x] **Web app marketplace local** 🎉
- [x] **Integración cámara Reolink RTSP** ✅

### 🔜 Fase 2: Producción
- [ ] Captura automática 24/7 desde Reolink
- [ ] Integración con plataformas de ecommerce (Shopify, WooCommerce, etc.)
- [ ] API REST para subir productos
- [ ] Sistema de reservas y recogida
- [ ] Migración a PostgreSQL

### 🚀 Fase 3: Escalado
- [ ] Múltiples puntos limpios (federación)
- [ ] App móvil (usuario reserva → recoge)
- [ ] Sistema automático completo (cámara → detección → publicación → notificaciones)

## ⚙️ Stack Tecnológico

- **Detección**: SAM 3 (Segment Anything Model 3) - Real SAM 3 from Meta
  - Text prompts para concept-based detection
  - Open-vocabulary segmentation
  - Install: `git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .`
  - Requires access to checkpoints on HuggingFace
  - **Device:** CPU (Mac Intel) o MPS (Apple Silicon) o CUDA (NVIDIA)
- **Análisis**: Claude Sonnet 4 (1 imagen + bboxes, ~$0.003 por captura)
- **Cámara**: 
  - **Producción**: Reolink RLC-810A (RTSP, 4K HEVC para Apple Silicon, 1080p H.264 para Intel)
  - **Pruebas actual**: Stream sub 1080p H.264 (más estable en Mac Intel)
  - **Desarrollo**: OpenCV (cv2) con webcams USB
- **Base de datos**: JSON files (fácil migración a PostgreSQL)
- **Web**: Flask (marketplace local, futuro: producción)
- **Config**: YAML
- **Crops**: Estandarizados 512x512 píxeles, objeto centrado

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

### Python 3.14 no compatible con PyTorch

Usa Python 3.12.10:
```bash
# Descarga desde python.org: https://www.python.org/downloads/release/python-31210/
python3.12 --version  # Debe mostrar: Python 3.12.10
python3.12 -m venv venv
```

📖 Ver [docs/PYTHON_SETUP.md](docs/PYTHON_SETUP.md) para instrucciones detalladas.

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
