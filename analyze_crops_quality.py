"""
Script temporal para analizar la calidad de los crops generados
comparándolos con la imagen original.
"""
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

def analyze_crop_quality(original_image: np.ndarray, crop_path: Path, bbox: List[int]) -> Dict[str, Any]:
    """
    Analiza la calidad de un crop comparándolo con la región correspondiente en la imagen original.
    """
    # Leer crop
    crop = cv2.imread(str(crop_path))
    if crop is None:
        return {'error': 'No se pudo leer el crop'}
    
    # Extraer región de la imagen original usando el bbox
    x, y, w, h = bbox
    img_h, img_w = original_image.shape[:2]
    
    # Ajustar bbox a límites de la imagen
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return {'error': 'Bbox inválido'}
    
    original_region = original_image[y1:y2, x1:x2].copy()
    
    # Métricas básicas
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    orig_gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY) if len(original_region.shape) == 3 else original_region
    
    # Redimensionar para comparar
    min_size = min(crop_gray.shape[0], crop_gray.shape[1], orig_gray.shape[0], orig_gray.shape[1])
    if min_size < 50:
        return {'error': 'Regiones demasiado pequeñas para comparar'}
    
    crop_resized = cv2.resize(crop_gray, (min_size, min_size))
    orig_resized = cv2.resize(orig_gray, (min_size, min_size))
    
    # Calcular correlación de histogramas
    hist_crop = cv2.calcHist([crop_resized], [0], None, [256], [0, 256])
    hist_orig = cv2.calcHist([orig_resized], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist_crop, hist_orig, cv2.HISTCMP_CORREL)
    
    # Calcular nitidez (Laplacian variance)
    def sharpness(img):
        return cv2.Laplacian(img, cv2.CV_64F).var()
    
    crop_sharpness = sharpness(crop_gray)
    orig_sharpness = sharpness(orig_gray)
    
    # Calcular contenido (píxeles no blancos/negros)
    crop_content = np.sum((crop_gray > 10) & (crop_gray < 245)) / crop_gray.size
    orig_content = np.sum((orig_gray > 10) & (orig_gray < 245)) / orig_gray.size
    
    return {
        'histogram_correlation': float(hist_corr),
        'crop_sharpness': float(crop_sharpness),
        'original_sharpness': float(orig_sharpness),
        'crop_content_ratio': float(crop_content),
        'original_content_ratio': float(orig_content),
        'sharpness_ratio': float(crop_sharpness / orig_sharpness) if orig_sharpness > 0 else 0,
        'content_match': hist_corr > 0.5,  # Buen match si correlación > 0.5
        'quality_score': float((hist_corr * 0.5 + min(crop_sharpness / 100, 1.0) * 0.3 + crop_content * 0.2))
    }

def parse_crop_filename(filename: str) -> Dict[str, Any]:
    """Extrae información del nombre del archivo crop"""
    # Formato: crop_0001_conf0.2508_area16890.jpg
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) >= 4:
        try:
            index = int(parts[1])
            conf = float(parts[2].replace('conf', ''))
            area = int(parts[3].replace('area', ''))
            return {'index': index, 'confidence': conf, 'area': area}
        except:
            pass
    return {}

def main():
    original_image_path = Path("images/test/image.png")
    crops_dir = Path("images/test/sam3_crops_experiment/crops_20251204_110748")
    
    if not original_image_path.exists():
        print(f"❌ Imagen original no encontrada: {original_image_path}")
        return
    
    if not crops_dir.exists():
        print(f"❌ Directorio de crops no encontrado: {crops_dir}")
        return
    
    # Leer imagen original
    original_image = cv2.imread(str(original_image_path))
    if original_image is None:
        print(f"❌ No se pudo leer la imagen original")
        return
    
    print(f"✅ Imagen original cargada: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"📁 Analizando crops en: {crops_dir}\n")
    
    # Obtener todos los crops
    crop_files = sorted(crops_dir.glob("crop_*.jpg"))
    print(f"📊 Total de crops: {len(crop_files)}\n")
    
    # Analizar primeros 20 crops como muestra
    results = []
    for i, crop_path in enumerate(crop_files[:20], 1):
        info = parse_crop_filename(crop_path.name)
        
        # Estimar bbox desde el nombre (necesitaríamos guardarlo, pero por ahora estimamos)
        # Por simplicidad, analizamos el crop directamente
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        
        # Métricas básicas del crop
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        sharpness = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
        mean_brightness = crop_gray.mean()
        std_variation = crop_gray.std()
        content_ratio = np.sum((crop_gray > 10) & (crop_gray < 245)) / crop_gray.size
        
        results.append({
            'filename': crop_path.name,
            'index': info.get('index', i),
            'confidence': info.get('confidence', 0),
            'area': info.get('area', 0),
            'sharpness': float(sharpness),
            'mean_brightness': float(mean_brightness),
            'std_variation': float(std_variation),
            'content_ratio': float(content_ratio),
            'size': f"{crop.shape[1]}x{crop.shape[0]}"
        })
    
    # Mostrar resultados
    print("="*80)
    print("ANÁLISIS DE CALIDAD DE CROPS (Muestra de primeros 20)")
    print("="*80)
    print(f"{'#':<4} {'Confianza':<10} {'Área':<10} {'Nitidez':<10} {'Brillo':<10} {'Contenido':<10} {'Tamaño':<12}")
    print("-"*80)
    
    good_crops = 0
    medium_crops = 0
    poor_crops = 0
    
    for r in results:
        quality = "✅" if r['sharpness'] > 50 and r['content_ratio'] > 0.3 else ("⚠️" if r['sharpness'] > 20 and r['content_ratio'] > 0.1 else "❌")
        
        if r['sharpness'] > 50 and r['content_ratio'] > 0.3:
            good_crops += 1
        elif r['sharpness'] > 20 and r['content_ratio'] > 0.1:
            medium_crops += 1
        else:
            poor_crops += 1
        
        print(f"{quality} {r['index']:<3} {r['confidence']:<10.4f} {r['area']:<10} {r['sharpness']:<10.1f} {r['mean_brightness']:<10.1f} {r['content_ratio']:<10.1%} {r['size']:<12}")
    
    print("="*80)
    print(f"\n📊 RESUMEN:")
    print(f"   ✅ Crops de buena calidad: {good_crops}/20 ({good_crops*100/20:.1f}%)")
    print(f"   ⚠️  Crops de calidad media: {medium_crops}/20 ({medium_crops*100/20:.1f}%)")
    print(f"   ❌ Crops de baja calidad: {poor_crops}/20 ({poor_crops*100/20:.1f}%)")
    
    # Estadísticas generales
    avg_sharpness = np.mean([r['sharpness'] for r in results])
    avg_content = np.mean([r['content_ratio'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\n📈 PROMEDIOS:")
    print(f"   Nitidez promedio: {avg_sharpness:.1f} (>50 = buena, 20-50 = aceptable, <20 = borrosa)")
    print(f"   Contenido promedio: {avg_content:.1%} (>30% = buena, 10-30% = aceptable, <10% = vacío)")
    print(f"   Confianza promedio: {avg_confidence:.4f}")
    
    print(f"\n💡 OBSERVACIONES:")
    if avg_sharpness < 20:
        print("   ⚠️  Los crops están muy borrosos - puede ser por:")
        print("      - Imagen original desenfocada")
        print("      - Redimensionamiento agresivo")
        print("      - Objetos muy pequeños")
    elif avg_sharpness < 50:
        print("   ⚠️  Los crops tienen nitidez moderada")
    else:
        print("   ✅ Los crops tienen buena nitidez")
    
    if avg_content < 0.1:
        print("   ⚠️  Los crops tienen poco contenido - muchos pueden ser fondos vacíos")
    elif avg_content < 0.3:
        print("   ⚠️  Los crops tienen contenido moderado")
    else:
        print("   ✅ Los crops tienen buen contenido")
    
    print(f"\n🔍 RECOMENDACIONES:")
    print("   - Revisa crops con nitidez < 20 o contenido < 10%")
    print("   - Considera filtrar crops muy pequeños (área < 100px²)")
    print("   - Crops con confianza muy baja (< 0.05) pueden ser falsos positivos")

if __name__ == "__main__":
    main()

