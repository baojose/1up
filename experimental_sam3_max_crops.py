"""
⚠️ EXPERIMENTAL: SAM3 maximum detection crop generation

Este script es TEMPORAL - será eliminado después del experimento.

Genera crops usando SOLO SAM3 con máxima detección:
- Toma una imagen
- Ejecuta SAM3 con umbral de confianza MÍNIMO (máxima detección)
- Genera crops para TODAS las detecciones de SAM3
- NO usa Claude, solo SAM3
- NO guarda en database

Uso:
    python experimental_sam3_max_crops.py <imagen.jpg>
"""
import cv2
import json
import logging
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
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
        bbox: [x, y, w, h] de SAM3
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


def generate_crops_from_detections(
    image: np.ndarray, 
    detections: List[Dict[str, Any]], 
    output_dir: Path,
    target_size: int = 300,
    context_factor: float = 1.5
) -> List[str]:
    """
    Genera crops cuadrados 1:1 centrados en los objetos con contexto.
    
    Todos los crops tienen el mismo tamaño y el objeto está centrado físicamente.
    """
    img_height, img_width = image.shape[:2]
    crop_paths = []
    
    logger.info(f"\n✂️  Generando {len(detections)} crops cuadrados centrados...")
    logger.info(f"   ⚙️  Tamaño objetivo: {target_size}x{target_size}px (1:1)")
    logger.info(f"   ⚙️  Factor de contexto: {context_factor}x (añade {int((context_factor-1)*100)}% contexto)")
    
    for i, det in enumerate(detections):
        bbox = det.get('bbox')
        confidence = det.get('confidence', 0.0)
        area = det.get('area', 0)
        
        if not bbox or len(bbox) != 4:
            logger.warning(f"  ⚠️  Detección {i+1} tiene bbox inválido, saltando")
            continue
        
        x, y, w, h = bbox
        
        # Validar bbox básico
        if w <= 0 or h <= 0:
            logger.warning(f"  ⚠️  Detección {i+1}: dimensiones inválidas, saltando")
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
                target_size=target_size,
                context_factor=context_factor
            )
        except Exception as e:
            logger.warning(f"  ⚠️  Detección {i+1}: error creando crop: {e}")
            continue
        
        # Guardar crop con información de detección
        crop_filename = f"crop_{i+1:04d}_conf{confidence:.4f}_area{area}.jpg"
        crop_path = output_dir / crop_filename
        cv2.imwrite(str(crop_path), square_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        crop_paths.append(str(crop_path))
        logger.info(f"  ✅ {i+1}/{len(detections)}: conf={confidence:.4f}, area={area}px² → {crop_filename}")
    
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


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Uso: python experimental_sam3_max_crops.py <imagen.jpg>")
        print("\nEjemplos:")
        print("  python experimental_sam3_max_crops.py imagen.jpg")
        print("  python experimental_sam3_max_crops.py /ruta/completa/imagen.png")
        print("  python experimental_sam3_max_crops.py images/test/image.png")
        sys.exit(1)
    
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
    if not image_path.exists():
        logger.error(f"❌ Imagen no encontrada: {image_path}")
        logger.error(f"   Ruta absoluta intentada: {image_path.absolute()}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("⚠️  EXPERIMENTO: SAM3 maximum detection crop generation")
    logger.info("="*60)
    logger.info(f"📸 Imagen: {image_path}")
    
    # Load config
    config = load_config()
    
    # Leer imagen
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"❌ No se pudo leer la imagen: {image_path}")
        sys.exit(1)
    
    logger.info(f"✅ Imagen cargada: {image.shape[1]}x{image.shape[0]}")
    
    # Crear directorio para crops
    experiment_dir = Path("images/test/sam3_crops_experiment")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorio con timestamp para esta ejecución
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = experiment_dir / f"crops_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Directorio de crops: {output_dir}")
    logger.info(f"   (Ruta completa: {output_dir.absolute()})")
    
    try:
        # Inicializar SAM3 con MÁXIMA DETECCIÓN
        logger.info("\n🔍 Inicializando SAM3 con máxima detección...")
        logger.info("   ⚙️  Configuración: confidence_threshold=0.0001 (máxima sensibilidad)")
        logger.info("   ⚙️  Sin filtros: detecta TODO (objetos, sombras, reflejos, fragmentos)")
        
        from detector import SAM3Detector
        
        # Crear detector con configuración de máxima detección
        # Usar confidence_threshold aún más bajo que el default (0.001)
        detector = SAM3Detector(device=config.get('sam3', {}).get('device', 'mps'))
        
        # Override processor con threshold aún más bajo para máxima detección
        from sam3.model.sam3_image_processor import Sam3Processor
        detector.processor = Sam3Processor(
            detector.model,
            device=detector.device,
            confidence_threshold=0.0001  # MÁXIMA DETECCIÓN EXPERIMENTAL
        )
        
        logger.info("✅ SAM3 inicializado con máxima detección")
        
        # Ejecutar detección
        logger.info("\n🔍 Ejecutando detección SAM3 (esto puede tardar 10-30 segundos)...")
        logger.info("   ⚙️  Sin filtros aplicados - detectará TODO")
        
        detections = detector.detect_objects(
            image=image,
            apply_filtering=False,  # Sin filtros
            enhance_image=config.get('sam3', {}).get('enhance_image', True),
            text_prompt=config.get('sam3', {}).get('text_prompt') or None
        )
        
        logger.info(f"\n✅ SAM3 detectó {len(detections)} objetos")
        
        if not detections:
            logger.warning("⚠️  No se detectaron objetos. Intenta:")
            logger.warning("   - Verificar que la imagen tenga objetos visibles")
            logger.warning("   - Ajustar la iluminación de la imagen")
            return
        
        # Mostrar estadísticas de detecciones
        confidences = [d.get('confidence', 0.0) for d in detections]
        areas = [d.get('area', 0) for d in detections]
        
        logger.info(f"\n📊 Estadísticas de detecciones:")
        logger.info(f"   Confianza: min={min(confidences):.4f}, max={max(confidences):.4f}, avg={np.mean(confidences):.4f}")
        logger.info(f"   Área: min={min(areas)}px², max={max(areas)}px², avg={int(np.mean(areas))}px²")
        
        # Generar crops cuadrados centrados
        crop_paths = generate_crops_from_detections(
            image=image,
            detections=detections,
            output_dir=output_dir,
            target_size=300,  # Todos los crops serán 300x300px
            context_factor=1.5  # Añade 50% contexto alrededor del objeto
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ EXPERIMENTO COMPLETADO")
        logger.info(f"{'='*60}")
        logger.info(f"📁 Directorio de crops: {output_dir}")
        logger.info(f"   Ruta completa: {output_dir.absolute()}")
        logger.info(f"📊 Total de crops generados: {len(crop_paths)}")
        logger.info(f"\n📋 Lista de crops (primeros 10):")
        for i, path in enumerate(crop_paths[:10], 1):
            logger.info(f"   {i:3d}. {Path(path).name}")
        if len(crop_paths) > 10:
            logger.info(f"   ... y {len(crop_paths) - 10} más")
        
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


if __name__ == "__main__":
    main()

