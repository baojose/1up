"""
Módulo de validación de calidad de imagen y correspondencia thumbnail-contenido.

Usa métricas matemáticas objetivas para:
1. Detectar blur/focus en imágenes
2. Validar que thumbnails corresponden al contenido identificado
3. Rechazar imágenes de baja calidad antes de procesamiento
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def calculate_sharpness_score(image: np.ndarray) -> float:
    """
    Calcula el score de nitidez usando Laplacian Variance.
    
    Métrica matemática: Var(Laplacian(I))
    - Imágenes nítidas: varianza alta (muchos bordes definidos)
    - Imágenes borrosas: varianza baja (bordes suaves)
    
    Args:
        image: Imagen BGR o RGB
        
    Returns:
        Score de nitidez (0-∞, típicamente 0-1000)
        - >100: Buena nitidez
        - 50-100: Nitidez aceptable
        - <50: Borrosa (rechazar)
    """
    if image is None or image.size == 0:
        return 0.0
    
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar filtro Laplacian para detectar bordes
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calcular varianza (métrica de nitidez)
    variance = laplacian.var()
    
    return float(variance)


def is_image_acceptable(image: np.ndarray, min_sharpness: float = 50.0) -> Tuple[bool, float, str]:
    """
    Valida si una imagen tiene calidad suficiente para procesamiento.
    
    Args:
        image: Imagen BGR
        min_sharpness: Umbral mínimo de nitidez (default: 50.0)
        
    Returns:
        Tuple (is_acceptable, sharpness_score, reason)
    """
    if image is None or image.size == 0:
        return False, 0.0, "Imagen vacía o inválida"
    
    sharpness = calculate_sharpness_score(image)
    
    if sharpness < min_sharpness:
        reason = f"Imagen demasiado borrosa (nitidez={sharpness:.1f} < {min_sharpness:.1f})"
        return False, sharpness, reason
    
    return True, sharpness, "OK"


def validate_thumbnail_content_match(
    thumbnail: np.ndarray,
    original_image: np.ndarray,
    bbox: List[int],
    expected_category: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Valida que el thumbnail corresponde al contenido identificado.
    
    Usa múltiples métricas matemáticas:
    1. Overlap de características (SIFT/ORB)
    2. Similitud estructural (SSIM)
    3. Correlación de histogramas
    4. Validación de región (bbox dentro de thumbnail)
    
    Args:
        thumbnail: Crop/thumbnail generado
        original_image: Imagen original completa
        bbox: Bounding box [x, y, w, h] usado para generar el thumbnail
        expected_category: Categoría esperada (opcional, para logging)
        
    Returns:
        Tuple (is_valid, metrics_dict)
    """
    metrics = {
        'bbox_in_thumbnail': False,
        'content_similarity': 0.0,
        'overlap_ratio': 0.0,
        'valid': False
    }
    
    if thumbnail is None or thumbnail.size == 0:
        metrics['error'] = "Thumbnail vacío"
        return False, metrics
    
    if original_image is None or original_image.size == 0:
        metrics['error'] = "Imagen original vacía"
        return False, metrics
    
    if not bbox or len(bbox) != 4:
        metrics['error'] = "Bbox inválido"
        return False, metrics
    
    x, y, w, h = bbox
    img_height, img_width = original_image.shape[:2]
    
    # Validación 1: Bbox debe estar dentro de la imagen
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        metrics['error'] = "Bbox fuera de límites de imagen"
        return False, metrics
    
    # Validación 2: Extraer región original usando bbox
    try:
        # Aplicar padding para comparar
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)
        
        original_region = original_image[y1:y2, x1:x2].copy()
        
        if original_region.size == 0:
            metrics['error'] = "Región original vacía"
            return False, metrics
        
        # Redimensionar para comparar (thumbnail puede estar cuadrado)
        thumb_height, thumb_width = thumbnail.shape[:2]
        orig_height, orig_width = original_region.shape[:2]
        
        # Redimensionar thumbnail o región original para que tengan tamaño similar
        target_size = min(thumb_width, thumb_height, orig_width, orig_height)
        if target_size < 50:
            metrics['error'] = f"Regiones demasiado pequeñas para comparar (target_size={target_size})"
            return False, metrics
        
        thumb_resized = cv2.resize(thumbnail, (target_size, target_size))
        orig_resized = cv2.resize(original_region, (target_size, target_size))
        
        # Validación 3: Similitud estructural usando correlación
        thumb_gray = cv2.cvtColor(thumb_resized, cv2.COLOR_BGR2GRAY) if len(thumb_resized.shape) == 3 else thumb_resized
        orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY) if len(orig_resized.shape) == 3 else orig_resized
        
        # Calcular correlación (similitud)
        correlation = cv2.matchTemplate(thumb_gray, orig_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
        
        # Validación 4: Overlap de características usando ORB
        try:
            orb = cv2.ORB_create(nfeatures=50)
            kp1, des1 = orb.detectAndCompute(thumb_gray, None)
            kp2, des2 = orb.detectAndCompute(orig_gray, None)
            
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # Buscar matches
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Ratio de matches sobre características detectadas
                match_ratio = len(matches) / max(len(kp1), len(kp2), 1)
            else:
                match_ratio = 0.0
        except Exception as e:
            logger.debug(f"Error en matching ORB: {e}")
            match_ratio = 0.0
        
        # Validación 5: Calcular overlap ratio del contenido
        # Comparar histogramas para verificar contenido similar
        hist_thumb = cv2.calcHist([thumb_gray], [0], None, [256], [0, 256])
        hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        hist_correlation = cv2.compareHist(hist_thumb, hist_orig, cv2.HISTCMP_CORREL)
        
        # Score combinado
        # correlation: 0-1 (similitud espacial)
        # match_ratio: 0-1 (features matching)
        # hist_correlation: 0-1 (similitud de distribución)
        combined_score = (correlation * 0.5 + match_ratio * 0.3 + hist_correlation * 0.2)
        
        metrics['bbox_in_thumbnail'] = True
        metrics['content_similarity'] = float(combined_score)
        metrics['overlap_ratio'] = float(correlation)
        metrics['match_ratio'] = float(match_ratio)
        metrics['hist_correlation'] = float(hist_correlation)
        
        # Umbral mínimo para considerar válido
        is_valid = combined_score > 0.3  # Al menos 30% de similitud
        
        metrics['valid'] = is_valid
        
        if not is_valid and expected_category:
            logger.warning(f"  ⚠️  Thumbnail no coincide con contenido esperado ({expected_category})")
            logger.warning(f"     Similitud: {combined_score:.2%}, Correlación: {correlation:.2%}, Matches: {match_ratio:.2%}")
        
        return is_valid, metrics
        
    except Exception as e:
        logger.error(f"Error validando thumbnail: {e}")
        metrics['error'] = str(e)
        return False, metrics


def validate_crop_quality(crop: np.ndarray, bbox_area: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Valida la calidad de un crop generado.
    
    Args:
        crop: Crop/thumbnail a validar
        bbox_area: Área del bbox original en píxeles²
        
    Returns:
        Tuple (is_valid, quality_metrics)
    """
    metrics = {
        'sharpness': 0.0,
        'content_ratio': 0.0,
        'valid': False
    }
    
    if crop is None or crop.size == 0:
        metrics['error'] = "Crop vacío"
        return False, metrics
    
    # Calcular nitidez del crop
    sharpness = calculate_sharpness_score(crop)
    metrics['sharpness'] = sharpness
    
    # Calcular ratio de contenido (no fondo blanco)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Contar píxeles que no son fondo blanco (255)
    # Umbral: píxeles < 240 se consideran contenido
    content_mask = gray < 240
    content_ratio = content_mask.sum() / gray.size
    metrics['content_ratio'] = float(content_ratio)
    
    # Validar: debe tener contenido suficiente (>20% del área) y nitidez mínima
    min_content_ratio = 0.2
    min_sharpness = 30.0  # Más bajo para crops pequeños
    
    is_valid = content_ratio >= min_content_ratio and sharpness >= min_sharpness
    
    metrics['valid'] = is_valid
    
    if not is_valid:
        reasons = []
        if content_ratio < min_content_ratio:
            reasons.append(f"poco contenido ({content_ratio:.1%} < {min_content_ratio:.1%})")
        if sharpness < min_sharpness:
            reasons.append(f"muy borroso (nitidez={sharpness:.1f} < {min_sharpness:.1f})")
        metrics['error'] = ", ".join(reasons)
    
    return is_valid, metrics


def validate_bbox_content(
    image: np.ndarray,
    bbox: List[int],
    min_content_ratio: float = 0.3,
    min_sharpness: float = 10.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Valida que un bbox contiene contenido real antes de generar un crop.
    
    Útil para validar bboxes aproximados de "missing objects" de Claude.
    
    Args:
        image: Imagen completa (BGR)
        bbox: Bounding box [x, y, w, h] a validar
        min_content_ratio: Ratio mínimo de contenido no uniforme (default: 0.3)
        min_sharpness: Nitidez mínima requerida (default: 10.0)
        
    Returns:
        Tuple (is_valid, metrics_dict)
    """
    metrics = {
        'content_ratio': 0.0,
        'sharpness': 0.0,
        'has_content': False,
        'valid': False
    }
    
    if image is None or image.size == 0:
        metrics['error'] = "Imagen vacía"
        return False, metrics
    
    if not bbox or len(bbox) != 4:
        metrics['error'] = "Bbox inválido"
        return False, metrics
    
    x, y, w, h = bbox
    img_height, img_width = image.shape[:2]
    
    # Validar bbox está dentro de límites
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        metrics['error'] = "Bbox fuera de límites"
        return False, metrics
    
    if w <= 0 or h <= 0:
        metrics['error'] = "Dimensiones inválidas"
        return False, metrics
    
    # Extraer región del bbox
    try:
        region = image[y:y+h, x:x+w].copy()
        
        if region.size == 0:
            metrics['error'] = "Región vacía"
            return False, metrics
        
        # Calcular nitidez
        sharpness = calculate_sharpness_score(region)
        metrics['sharpness'] = float(sharpness)
        
        # Calcular contenido (píxeles con variación)
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calcular varianza local (áreas con variación = contenido)
        kernel_size = min(5, min(w, h) // 4)
        if kernel_size >= 3:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            # Ratio de píxeles con variación significativa
            content_mask = local_variance > 10  # Umbral de variación
            content_ratio = content_mask.sum() / gray.size
        else:
            # Para regiones muy pequeñas, usar desviación estándar simple
            std_dev = gray.std()
            content_ratio = 1.0 if std_dev > 15 else std_dev / 15.0
        
        metrics['content_ratio'] = float(content_ratio)
        metrics['has_content'] = content_ratio >= min_content_ratio
        
        # Validar: debe tener contenido y nitidez mínima
        is_valid = content_ratio >= min_content_ratio and sharpness >= min_sharpness
        metrics['valid'] = is_valid
        
        if not is_valid:
            reasons = []
            if content_ratio < min_content_ratio:
                reasons.append(f"poco contenido ({content_ratio:.1%} < {min_content_ratio:.1%})")
            if sharpness < min_sharpness:
                reasons.append(f"muy borroso (nitidez={sharpness:.1f} < {min_sharpness:.1f})")
            metrics['error'] = ", ".join(reasons)
        
        return is_valid, metrics
        
    except Exception as e:
        logger.error(f"Error validando bbox content: {e}")
        metrics['error'] = str(e)
        return False, metrics
