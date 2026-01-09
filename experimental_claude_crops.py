"""
⚠️ EXPERIMENTAL: Claude-only crop generation

Este script es TEMPORAL - será eliminado después del experimento.

Genera crops usando SOLO Claude (sin SAM):
- Toma una imagen
- Envía a Claude para que identifique objetos y genere bboxes
- Genera crops basados en los bboxes de Claude
- NO hace reconocimiento/descripción, solo crops
- NO guarda en database

Uso:
    python experimental_claude_crops.py <imagen.jpg>
"""
import cv2
import json
import logging
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def ask_claude_for_bboxes(image_path: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Pide a Claude que identifique objetos y genere bboxes.
    NO pide descripciones, solo bboxes.
    """
    from anthropic import Anthropic
    
    client = Anthropic(api_key=api_key)
    
    # Leer imagen y convertir a base64
    import base64
    
    # Detectar tipo MIME basado en extensión
    image_ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    media_type = mime_types.get(image_ext, 'image/jpeg')
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prompt simple: solo queremos bboxes, no descripciones
    prompt = """Analiza esta imagen y identifica TODOS los objetos visibles.

Para cada objeto, proporciona SOLO un bounding box en formato [x, y, width, height] donde:
- x, y = esquina superior izquierda
- width, height = ancho y alto del bbox

Responde con un JSON array de objetos, cada uno con:
{
  "bbox": [x, y, width, height],
  "name": "nombre breve del objeto"
}

IMPORTANTE:
- Identifica TODOS los objetos visibles
- Incluye objetos pequeños, parcialmente visibles, etc.
- No necesitas describir, solo dar bbox y nombre breve
- El bbox debe estar en píxeles de la imagen original

Responde SOLO con el JSON array, sin texto adicional."""

    logger.info("🤖 Enviando imagen a Claude para identificación de objetos...")
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        response_text = message.content[0].text
        
        # Extraer JSON de la respuesta
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            objects = json.loads(json_text)
            logger.info(f"✅ Claude identificó {len(objects)} objetos")
            return objects
        else:
            logger.error("❌ No se encontró JSON en la respuesta de Claude")
            logger.error(f"Respuesta: {response_text[:500]}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error comunicándose con Claude: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def create_centered_square_crop(
    image: np.ndarray,
    bbox: List[int],
    target_size: int = 300,
    context_factor: float = 1.5
) -> np.ndarray:
    """
    Crea crop cuadrado 1:1 centrado en el objeto con contexto.
    
    El objeto queda físicamente centrado en el crop, con contexto alrededor.
    
    Args:
        image: Imagen original
        bbox: [x, y, w, h]
        target_size: Tamaño final del crop (300x300px)
        context_factor: Cuánto contexto añadir (1.5 = 50% extra alrededor)
    
    Returns:
        Crop cuadrado target_size x target_size con objeto centrado
    """
    x, y, w, h = bbox
    img_height, img_width = image.shape[:2]
    
    # 1. Calcular centro físico del objeto
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 2. Calcular lado del cuadrado (el mayor entre w y h + contexto)
    max_side = max(w, h)
    square_side = int(max_side * context_factor)
    
    # 3. Calcular coordenadas del cuadrado centrado en el objeto
    half_side = square_side // 2
    x1 = center_x - half_side
    y1 = center_y - half_side
    x2 = center_x + half_side
    y2 = center_y + half_side
    
    # 4. Ajustar si se sale de la imagen (mover el cuadrado pero mantener centro del objeto)
    if x1 < 0:
        offset = -x1
        x1 = 0
        x2 = min(img_width, x2 + offset)
    if y1 < 0:
        offset = -y1
        y1 = 0
        y2 = min(img_height, y2 + offset)
    if x2 > img_width:
        offset = x2 - img_width
        x1 = max(0, x1 - offset)
        x2 = img_width
    if y2 > img_height:
        offset = y2 - img_height
        y1 = max(0, y1 - offset)
        y2 = img_height
    
    # 5. Extraer crop de la imagen original
    crop = image[y1:y2, x1:x2].copy()
    
    if crop.size == 0:
        # Si el crop está vacío, crear uno negro del tamaño objetivo
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 6. Si el crop no es cuadrado (porque tocó los bordes), añadir padding
    crop_h, crop_w = crop.shape[:2]
    
    if crop_w != crop_h:
        # Crear cuadrado con color de fondo (promedio del borde del crop)
        border_color = _get_border_color(crop)
        max_dim = max(crop_w, crop_h)
        square_crop = np.full((max_dim, max_dim, 3), border_color, dtype=np.uint8)
        
        # Centrar el crop en el cuadrado
        y_offset = (max_dim - crop_h) // 2
        x_offset = (max_dim - crop_w) // 2
        square_crop[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = crop
        crop = square_crop
    
    # 7. Redimensionar a tamaño objetivo manteniendo aspecto cuadrado
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    return crop_resized


def generate_crops(image: np.ndarray, bboxes: List[Dict[str, Any]], output_dir: Path) -> List[str]:
    """
    Genera crops cuadrados 1:1 centrados en los objetos con contexto.
    
    Todos los crops tienen el mismo tamaño y el objeto está centrado físicamente.
    """
    img_height, img_width = image.shape[:2]
    crop_paths = []
    
    logger.info(f"\n✂️  Generando {len(bboxes)} crops cuadrados centrados...")
    logger.info(f"   ⚙️  Tamaño objetivo: 300x300px (1:1)")
    logger.info(f"   ⚙️  Factor de contexto: 1.5x (añade 50% contexto)")
    
    for i, obj in enumerate(bboxes):
        bbox = obj.get('bbox')
        name = obj.get('name', f'object_{i+1}')
        
        if not bbox or len(bbox) != 4:
            logger.warning(f"  ⚠️  Objeto {i+1} tiene bbox inválido, saltando")
            continue
        
        x, y, w, h = bbox
        
        # Validar bbox básico
        if w <= 0 or h <= 0:
            logger.warning(f"  ⚠️  Objeto {i+1} ({name}): dimensiones inválidas, saltando")
            continue
        
        # Ajustar bbox a límites de la imagen si es necesario
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Crear crop cuadrado centrado
        try:
            square_crop = create_centered_square_crop(
                image=image,
                bbox=[x, y, w, h],
                target_size=300,
                context_factor=1.5
            )
        except Exception as e:
            logger.warning(f"  ⚠️  Objeto {i+1} ({name}): error creando crop: {e}")
            continue
        
        # Guardar crop
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        crop_filename = f"crop_{i+1:03d}_{safe_name.replace(' ', '_')}.jpg"
        crop_path = output_dir / crop_filename
        cv2.imwrite(str(crop_path), square_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        crop_paths.append(str(crop_path))
        logger.info(f"  ✅ {i+1}/{len(bboxes)}: {name} → {crop_filename}")
    
    return crop_paths


def _get_border_color(crop: np.ndarray) -> tuple:
    """Obtiene el color promedio del borde del crop para padding"""
    h, w = crop.shape[:2]
    border_pixels = np.concatenate([
        crop[0, :].reshape(-1, 3),  # Top
        crop[-1, :].reshape(-1, 3),  # Bottom
        crop[:, 0].reshape(-1, 3),  # Left
        crop[:, -1].reshape(-1, 3)  # Right
    ])
    return tuple(map(int, border_pixels.mean(axis=0)))


def load_image_from_clipboard() -> Optional[Path]:
    """Intenta cargar imagen desde el portapapeles (macOS)"""
    try:
        import subprocess
        import tempfile
        
        # macOS: leer imagen del portapapeles
        result = subprocess.run(
            ['osascript', '-e', 'the clipboard as «class PNGf»'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Intentar con otro método
            result = subprocess.run(
                ['osascript', '-e', 'the clipboard as «class JPEG»'],
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0 and result.stdout.strip():
            # Guardar en archivo temporal
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_path = Path(temp_file.name)
            
            # Leer datos binarios del portapapeles
            png_data = subprocess.run(
                ['osascript', '-e', 'the clipboard as «class PNGf»'],
                capture_output=True
            ).stdout
            
            if png_data:
                temp_file.write(png_data)
                temp_file.close()
                logger.info(f"✅ Imagen cargada desde portapapeles: {temp_path}")
                return temp_path
        
        return None
    except Exception as e:
        logger.debug(f"No se pudo cargar del portapapeles: {e}")
        return None


def main():
    """Main function"""
    image_path = None
    
    # Si no hay argumentos, intentar leer del portapapeles
    if len(sys.argv) < 2:
        logger.info("⚠️  No se proporcionó imagen, intentando leer del portapapeles...")
        image_path = load_image_from_clipboard()
        
        if not image_path:
            print("Uso: python experimental_claude_crops.py <imagen.jpg>")
            print("\nEjemplos:")
            print("  python experimental_claude_crops.py imagen.jpg")
            print("  python experimental_claude_crops.py /ruta/completa/imagen.png")
            print("  python experimental_claude_crops.py images/raw/scene_*.jpg")
            print("\nO copia una imagen al portapapeles y ejecuta sin argumentos")
            sys.exit(1)
    else:
        image_path_str = sys.argv[1]
        
        # Expandir wildcards si es necesario
        if '*' in image_path_str or '?' in image_path_str:
            import glob
            matches = glob.glob(image_path_str)
            if not matches:
                logger.error(f"❌ No se encontraron imágenes que coincidan con: {image_path_str}")
                sys.exit(1)
            if len(matches) > 1:
                logger.warning(f"⚠️  Múltiples coincidencias, usando la primera: {matches[0]}")
            image_path_str = matches[0]
        
        image_path = Path(image_path_str)
    
    if not image_path or not image_path.exists():
        logger.error(f"❌ Imagen no encontrada: {image_path}")
        if image_path:
            logger.error(f"   Ruta absoluta intentada: {image_path.absolute()}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("⚠️  EXPERIMENTO: Claude-only crop generation")
    logger.info("="*60)
    logger.info(f"📸 Imagen: {image_path}")
    
    # Load config
    config = load_config()
    
    # Try to load API key from file first (same as run_live_detection_with_claude.sh)
    api_key_file = Path('.claude_api_key')
    api_key = None
    
    if api_key_file.exists():
        try:
            api_key = api_key_file.read_text().strip()
            logger.info(f"✅ API key cargada desde {api_key_file}")
        except Exception as e:
            logger.warning(f"⚠️  No se pudo leer {api_key_file}: {e}")
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.environ.get(config['claude']['api_key_env'])
        if api_key:
            logger.info(f"✅ API key cargada desde variable de entorno")
    
    if not api_key:
        logger.error(f"❌ API key no encontrada.")
        logger.error(f"   Configura {config['claude']['api_key_env']} o crea el archivo .claude_api_key")
        logger.error(f"   Ejemplo: echo 'sk-ant-api03-...' > .claude_api_key")
        sys.exit(1)
    
    # Leer imagen
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"❌ No se pudo leer la imagen: {image_path}")
        sys.exit(1)
    
    logger.info(f"✅ Imagen cargada: {image.shape[1]}x{image.shape[0]}")
    
    # Crear directorio temporal para crops
    # Usar un directorio fijo en images/test/crops_experiment para facilitar acceso
    experiment_dir = Path("images/test/crops_experiment")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorio con timestamp para esta ejecución
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = experiment_dir / f"crops_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Directorio de crops: {output_dir}")
    logger.info(f"   (Ruta completa: {output_dir.absolute()})")
    
    try:
        # Paso 1: Pedir a Claude que identifique objetos y genere bboxes
        objects = ask_claude_for_bboxes(str(image_path), api_key)
        
        if not objects:
            logger.error("❌ Claude no identificó ningún objeto")
            return
        
        logger.info(f"\n✅ Claude identificó {len(objects)} objetos:")
        for i, obj in enumerate(objects, 1):
            bbox = obj.get('bbox', [])
            name = obj.get('name', 'Unknown')
            logger.info(f"  {i}. {name}: bbox={bbox}")
        
        # Paso 2: Generar crops
        crop_paths = generate_crops(image, objects, output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ EXPERIMENTO COMPLETADO")
        logger.info(f"{'='*60}")
        logger.info(f"📁 Directorio de crops: {output_dir}")
        logger.info(f"   Ruta completa: {output_dir.absolute()}")
        logger.info(f"📊 Total de crops generados: {len(crop_paths)}")
        logger.info(f"\n📋 Lista de crops:")
        for i, path in enumerate(crop_paths, 1):
            logger.info(f"   {i:3d}. {Path(path).name}")
        
        logger.info(f"\n💡 Los crops están guardados permanentemente en:")
        logger.info(f"   {output_dir.absolute()}")
        logger.info(f"\n⚠️  RECUERDA: Este es un experimento temporal")
        logger.info(f"   Puedes eliminar el directorio cuando termines:")
        logger.info(f"   rm -rf {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrumpido por usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # NO eliminamos automáticamente - el usuario puede revisar los crops primero
        logger.info(f"\n💡 Para eliminar el directorio temporal:")
        logger.info(f"   rm -rf {output_dir}")


if __name__ == "__main__":
    main()

