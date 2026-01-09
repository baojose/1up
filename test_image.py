#!/usr/bin/env python3
"""
Test SAM 3 on a specific image file.
Usage: python3 test_image.py <image_path>
"""

import sys
import logging
from pathlib import Path
import cv2
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test SAM 3 on a specific image."""
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        # Default to the image the user mentioned
        image_path = Path("images/raw/scene_camCAM0_2025-12-01_20-36-47.jpg")
    
    if not image_path.exists():
        logger.error(f"❌ Imagen no encontrada: {image_path}")
        logger.info("💡 Uso: python3 test_image.py <ruta_imagen>")
        return 1
    
    logger.info("=" * 60)
    logger.info("🔍 TEST SAM 3 - Imagen Específica")
    logger.info("=" * 60)
    logger.info(f"📸 Imagen: {image_path}")
    
    # Load image
    logger.info("📖 Cargando imagen...")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"❌ No se pudo cargar la imagen: {image_path}")
        return 1
    
    logger.info(f"✅ Imagen cargada: {image.shape[1]}x{image.shape[0]} píxeles")
    
    # Load config
    try:
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
        
        # Force CPU for testing (MPS has known bugs with some operations)
        # In production with camera, MPS should work better
        device = "cpu"
        logger.info("ℹ️  Usando CPU para este test (MPS tiene bugs conocidos)")
        logger.info("   Para uso con cámara, MPS debería funcionar mejor")
        
        detector = SAM3Detector(device=device)
        logger.info(f"✅ SAM 3 inicializado en {device}")
    except Exception as e:
        logger.error(f"❌ Error inicializando detector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Detect objects
    logger.info("")
    logger.info("🔍 Detectando objetos con SAM 3...")
    logger.info("   (Esto puede tardar 30-90 segundos)")
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
            logger.warning("⚠️  No se detectaron objetos")
            logger.info("💡 Intenta:")
            logger.info("   - Ajustar parámetros en config.yaml")
            logger.info("   - Usar un text_prompt más específico")
            return 0
        
        # Show results
        logger.info("")
        logger.info("📊 RESULTADOS (primeros 15 objetos):")
        logger.info("-" * 60)
        for i, det in enumerate(detections[:15]):
            bbox = det.get('bbox', [])
            area = det.get('area', 0)
            conf = det.get('confidence', 0.0)
            logger.info(f"  Objeto #{i+1}:")
            logger.info(f"    Bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            logger.info(f"    Área: {area:.0f} px²")
            logger.info(f"    Confianza: {conf:.3f}")
        
        if len(detections) > 15:
            logger.info(f"  ... y {len(detections) - 15} objetos más")
        
        # Save visualization
        logger.info("")
        logger.info("💾 Guardando visualización...")
        output_dir = Path("test_images")
        output_dir.mkdir(exist_ok=True)
        
        viz_path = output_dir / f"{image_path.stem}_detection.jpg"
        viz_image = detector.visualize_detections(image.copy(), detections)
        cv2.imwrite(str(viz_path), viz_image)
        logger.info(f"✅ Visualización guardada: {viz_path}")
        
        # Save crops
        logger.info("")
        logger.info("✂️  Guardando crops...")
        crops_dir = output_dir / f"{image_path.stem}_crops"
        crops_dir.mkdir(exist_ok=True)
        crop_paths = detector.save_crops(image, detections, str(crops_dir), prefix="obj")
        logger.info(f"✅ {len(crop_paths)} crops guardados en: {crops_dir}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 ¡Test completado exitosamente!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📁 Archivos generados:")
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

