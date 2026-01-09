"""
Centralized filtering module for 1UP pipeline.
All filtering logic is documented and centralized here.
Max 350 lines.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def filter_generic_names(obj_name: str) -> bool:
    """
    Filter out generic/unhelpful names.
    
    Returns:
        True if name should be rejected (is generic), False if acceptable
    """
    generic_keywords = [
        'superficie', 'esquina', 'borde', 'fragmento', 'pedazo',
        'mesa', 'fondo', 'pared', 'suelo', 'mobiliario',
        'objeto rectangular', 'cosa', 'elemento', 'parte',
        'surface', 'corner', 'edge', 'fragment', 'piece',
        'table', 'background', 'wall', 'floor', 'furniture',
        'rectangular object', 'thing', 'element', 'part'
    ]
    
    name_lower = obj_name.lower()
    return any(keyword in name_lower for keyword in generic_keywords)


def filter_by_size(
    bbox: List[int],
    image_shape: tuple,
    max_area_ratio: float = 0.5
) -> bool:
    """
    Filter objects that are too large (likely background).
    
    Args:
        bbox: [x, y, w, h]
        image_shape: (height, width)
        max_area_ratio: Maximum area as ratio of image (default: 0.5 = 50%)
    
    Returns:
        True if object should be rejected (too large), False if acceptable
    """
    img_height, img_width = image_shape
    bbox_area = bbox[2] * bbox[3]  # width * height
    image_area = img_width * img_height
    area_ratio = bbox_area / image_area
    
    return area_ratio > max_area_ratio


def filter_useful_objects(
    analyses: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    image_shape: tuple,
    max_area_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter analyses to keep only useful objects.
    
    This applies post-Claude filtering:
    1. Filter by usefulness (useful="yes")
    2. Filter by size (max_area_ratio)
    3. Filter by generic names
    
    Args:
        analyses: List of Claude analyses (each has 'n', 'useful', 'name', etc.)
        detections: List of SAM detections (for bbox lookup)
        image_shape: (height, width) of image
        max_area_ratio: Maximum area ratio (default: 0.5 = 50%)
    
    Returns:
        List of useful objects with analysis + detection data
    """
    useful_objects = []
    skipped_count = 0
    
    for analysis in analyses:
        n = analysis.get('n', 0)
        
        # Skip if not useful
        if analysis.get('useful', 'no').lower() != 'yes':
            skipped_count += 1
            reason = analysis.get('reason', 'not useful')
            logger.debug(f"  ⏭️  #{n}: Skipped - {reason}")
            continue
        
        # Get corresponding detection
        if n < 1 or n > len(detections):
            logger.warning(f"Invalid object number n={n}, skipping")
            skipped_count += 1
            continue
        
        detection = detections[n - 1]  # n is 1-indexed
        
        # Filter by size (use bbox from analysis if available, otherwise from detection)
        bbox = analysis.get('bbox') or detection.get('bbox')
        if not bbox:
            logger.warning(f"  ⚠️  #{n}: No bbox available, skipping")
            skipped_count += 1
            continue
        
        if filter_by_size(bbox, image_shape, max_area_ratio):
            skipped_count += 1
            logger.debug(f"  ⏭️  #{n}: Skipped - too large")
            continue
        
        # Filter by generic names
        obj_name = analysis.get('name', '')
        if filter_generic_names(obj_name):
            skipped_count += 1
            logger.debug(f"  ⏭️  #{n}: Skipped - generic name: '{obj_name}'")
            continue
        
        # Object passed all filters
        useful_objects.append({
            'analysis': analysis,
            'detection': detection,
            'n': n
        })
    
    logger.info(f"📊 Filtered {len(useful_objects)} useful objects, {skipped_count} skipped")
    return useful_objects

