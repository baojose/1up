#!/usr/bin/env python3
"""
Test SAM 3 with a static image (no camera required).
Downloads a test image if needed and runs detection.
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_test_image(output_path: Path) -> bool:
    """Download a test image from a public URL."""
    import urllib.request
    
    # Using a public test image (objects on a table)
    test_urls = [
        "https://images.unsplash.com/photo-1556912172-45b7abe8b7e4?w=800",  # Kitchen objects
        "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800",  # Office desk
    ]
    
    for url in test_urls:
        try:
            logger.info(f"📥 Descargando imagen de prueba desde {url}...")
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"✅ Imagen descargada: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"⚠️  No se pudo descargar desde {url}: {e}")
            continue
    
    return False

def create_test_image(output_path: Path) -> bool:
    """Create a simple test image with geometric shapes."""
    logger.info("🎨 Creando imagen de prueba con formas geométricas...")
    
    # Create a 1280x960 image (same as camera)
    img = np.zeros((960, 1280, 3), dtype=np.uint8)
    img.fill(240)  # Light gray background
    
    # Draw some colored rectangles (simulating objects)
    cv2.rectangle(img, (100, 100), (300, 400), (0, 100, 255), -1)  # Blue rectangle
    cv2.rectangle(img, (400, 200), (600, 500), (0, 255, 100), -1)  # Green rectangle
    cv2.rectangle(img, (700, 150), (900, 450), (255, 100, 0), -1)  # Orange rectangle
    cv2.rectangle(img, (1000, 300), (1200, 600), (255, 0, 100), -1)  # Pink rectangle
    
    # Add some circles
    cv2.circle(img, (350, 600), 80, (255, 200, 0), -1)  # Yellow circle
    cv2.circle(img, (950, 700), 100, (200, 0, 255), -1)  # Purple circle
    
    # Save
    cv2.imwrite(str(output_path), img)
    logger.info(f"✅ Imagen de prueba creada: {output_path}")
    return True

def main():
    """Test SAM 3 with a static image."""
    logger.info("=" * 60)
    logger.info("🧪 TEST SAM 3 - Imagen Estática")
    logger.info("=" * 60)
    
    # Create test image directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    test_image_path = test_dir / "test_image.jpg"
    
    # Get or create test image
    if not test_image_path.exists():
        logger.info("📸 No hay imagen de prueba, creando una...")
        if not download_test_image(test_image_path):
            logger.info("⚠️  No se pudo descargar, creando imagen sintética...")
            create_test_image(test_image_path)
    else:
        logger.info(f"✅ Usando imagen existente: {test_image_path}")
    
    # Load image
    logger.info("📖 Cargando imagen...")
    image = cv2.imread(str(test_image_path))
    if image is None:
        logger.error(f"❌ No se pudo cargar la imagen: {test_image_path}")
        return 1
    
    logger.info(f"✅ Imagen cargada: {image.shape[1]}x{image.shape[0]} píxeles")
    
    # Load config
    try:
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Error cargando config.yaml: {e}")
        return 1
    
    # Initialize detector
    logger.info("🤖 Inicializando SAM 3...")
    try:
        from detector import SAM3Detector
        import torch
        
        # Use CPU for static test (more stable than MPS for testing)
        # MPS has some known issues with certain operations
        if torch.backends.mps.is_available():
            logger.info("ℹ️  MPS disponible, pero usando CPU para este test (más estable)")
        device = "cpu"
        detector = SAM3Detector(device=device)
        logger.info("✅ SAM 3 inicializado en CPU")
    except Exception as e:
        logger.error(f"❌ Error inicializando detector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Detect objects
    logger.info("🔍 Detectando objetos...")
    logger.info("   (Esto puede tardar 30-60 segundos la primera vez)")
    logger.info("")
    
    try:
        text_prompt = config.get("sam3", {}).get("text_prompt", "object")
        enhance_image = config.get("sam3", {}).get("enhance_image", False)
        
        detections = detector.detect_objects(
            image,
            text_prompt=text_prompt,
            enhance_image=enhance_image
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✅ Detección completada: {len(detections)} objetos encontrados")
        logger.info("=" * 60)
        
        if len(detections) == 0:
            logger.warning("⚠️  No se detectaron objetos. Esto puede ser normal si:")
            logger.warning("   - La imagen es muy simple")
            logger.warning("   - Los objetos no son claramente visibles")
            logger.warning("   - Los parámetros de SAM son muy estrictos")
            return 0
        
        # Show results
        logger.info("")
        logger.info("📊 RESULTADOS:")
        logger.info("-" * 60)
        for i, det in enumerate(detections[:10]):  # Show first 10
            bbox = det.get('bbox', [])
            area = det.get('area', 0)
            conf = det.get('confidence', 0.0)
            logger.info(f"  Objeto #{i+1}:")
            logger.info(f"    Bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            logger.info(f"    Área: {area} px²")
            logger.info(f"    Confianza: {conf:.3f}")
        
        if len(detections) > 10:
            logger.info(f"  ... y {len(detections) - 10} objetos más")
        
        # Save visualization
        logger.info("")
        logger.info("💾 Guardando visualización...")
        viz_path = test_dir / "test_detection_viz.jpg"
        viz_image = detector.visualize_detections(image.copy(), detections)
        cv2.imwrite(str(viz_path), viz_image)
        logger.info(f"✅ Visualización guardada: {viz_path}")
        
        # Save crops
        logger.info("")
        logger.info("✂️  Guardando crops...")
        crops_dir = test_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        crop_paths = detector.save_crops(image, detections, str(crops_dir), prefix="obj")
        logger.info(f"✅ {len(crop_paths)} crops guardados en: {crops_dir}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 ¡Test completado exitosamente!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📁 Archivos generados:")
        logger.info(f"   - Imagen original: {test_image_path}")
        logger.info(f"   - Visualización: {viz_path}")
        logger.info(f"   - Crops: {crops_dir}/")
        logger.info("")
        logger.info("💡 Puedes abrir la visualización para ver los resultados:")
        logger.info(f"   open {viz_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error durante la detección: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

