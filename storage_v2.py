"""
Storage utilities for 1UP - V2
Generates crops AFTER Claude analysis (only for useful objects).
This eliminates the mapping problem: n=1 always corresponds to obj_001.jpg
Max 350 lines.
"""
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def save_crops_for_useful_objects(
    image: np.ndarray,
    analyses: List[Dict[str, Any]],
    useful_objects: List[Dict[str, Any]],
    output_dir: str,
    timestamp: str,
    prefix: str = "obj"
) -> Dict[int, str]:
    """
    Generate crops ONLY for useful objects AFTER Claude analysis.
    
    This eliminates mapping issues: n=1 → obj_001.jpg (always matches).
    
    Args:
        image: Full scene image (BGR)
        analyses: List of Claude analyses (each has 'n' field, 1-indexed, RENUMBERED consecutively)
        useful_objects: List of useful objects (each has 'detection' and 'filtered_index')
        output_dir: Directory to save crops
        timestamp: Scene timestamp for directory naming
        prefix: Prefix for crop filenames (default: "obj")
        
    Returns:
        Dictionary mapping n (1-indexed, renumbered) to crop_path
    """
    crop_dir = Path(output_dir) / timestamp
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    n_to_crop = {}  # n (1-indexed, renumbered) → crop_path
    padding = 30  # Same padding as before
    
    # CRITICAL: Use bboxes directly from Claude's analyses (not from SAM detections)
    # Claude provides precise bboxes for semantic objects, SAM only provides region bboxes
    image_height, image_width = image.shape[:2]
    
    # CRITICAL FIX: Use analysis['n'] to map directly to useful_objects, not index i
    # This ensures correct mapping even if some objects are skipped
    for i, analysis in enumerate(analyses):
        n = analysis.get('n', 0)  # This is the renumbered n (1, 2, 3, 4...)
        
        # Validate n is in valid range
        if n < 1:
            logger.warning(f"  ⚠️  Invalid n={n}, skipping")
            continue
        
        # CRITICAL: Get bbox from Claude first, fallback to SAM detection if missing
        # Claude provides semantic object bboxes (more accurate), but SAM bbox is always available
        bbox = analysis.get('bbox')
        source = "claude"
        
        # Fallback: Use SAM detection bbox if Claude didn't provide one
        if not bbox or len(bbox) != 4:
            logger.warning(f"  ⚠️  Analysis n={n} missing Claude bbox, using SAM fallback...")
            
            # CRITICAL FIX: Use index i directly (analyses[i] corresponds to useful_objects[i])
            # Since we renumbered consecutively, i should match the position in useful_objects
            if i < len(useful_objects):
                obj = useful_objects[i]
                detection = obj.get('detection', {})
                sam_bbox = detection.get('bbox')
                
                if sam_bbox and len(sam_bbox) == 4:
                    bbox = sam_bbox
                    source = "sam_fallback"
                    logger.info(f"     ✅ Using SAM bbox as fallback for n={n} (from useful_objects[{i}])")
                else:
                    logger.error(f"     ❌ No SAM bbox available in useful_objects[{i}], skipping n={n}")
                    continue
            else:
                logger.error(f"  ❌ Index {i} out of range for useful_objects (has {len(useful_objects)} items), skipping n={n}")
                continue
        
        x, y, w, h = bbox
        bbox_area = w * h
        
        # Validate minimum area (RELAXED: allow smaller objects)
        if bbox_area < 100:  # pixels² (reduced from 500 to allow smaller objects)
            logger.warning(f"  ⚠️  Bbox too small for n={n} (area={bbox_area}px²), skipping")
            continue
        
        # Validate bbox is within image bounds
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
            logger.warning(f"  ⚠️  Bbox outside image for n={n}, clipping")
            x = max(0, x)
            y = max(0, y)
            w = min(w, image_width - x)
            h = min(h, image_height - y)
            bbox = [x, y, w, h]
        
        # Verify crop dimensions are valid
        if w <= 0 or h <= 0:
            logger.warning(f"  ⚠️  Invalid crop dimensions for n={n}, bbox=({x},{y},{w},{h})")
            continue
        
        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_width, x + w + padding)
        y2 = min(image_height, y + h + padding)
        
        # Verify crop dimensions are valid
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"  ⚠️  Invalid crop dimensions for n={n}, bbox=({x},{y},{w},{h})")
            continue
        
        # Extract crop
        crop = image[y1:y2, x1:x2].copy()
        
        # Create square crop (1:1 aspect ratio) with object centered
        crop_height, crop_width = crop.shape[:2]
        target_size = max(crop_width, crop_height)
        min_size = 300
        if target_size < min_size:
            target_size = min_size
        
        # Get average border color for padding
        border_color = _get_border_color(crop)
        square_crop = np.full((target_size, target_size, 3), border_color, dtype=np.uint8)
        
        # Center crop in square
        x_offset = (target_size - crop_width) // 2
        y_offset = (target_size - crop_height) // 2
        
        # If crop is larger than target, scale it down
        if crop_width > target_size or crop_height > target_size:
            scale = min(target_size / crop_width, target_size / crop_height)
            new_w = int(crop_width * scale)
            new_h = int(crop_height * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            crop_height, crop_width = crop.shape[:2]
            x_offset = (target_size - crop_width) // 2
            y_offset = (target_size - crop_height) // 2
        
        # Place crop in center of square
        square_crop[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = crop
        
        # VALIDATION: Check crop quality using mathematical metrics
        from image_quality import validate_crop_quality, calculate_sharpness_score
        
        crop_mean = square_crop.mean()
        crop_std = square_crop.std()
        source = "claude"  # All crops now use Claude's bboxes
        
        # Validation 1: Basic content check (quick rejection of empty crops)
        is_valid_crop = True
        validation_warnings = []
        
        # RELAXED THRESHOLDS: Allow dark objects (kettlebell, black shoes, etc.)
        # Only reject if completely black (mean < 5) or completely white (mean > 252)
        if crop_mean < 5:
            is_valid_crop = False
            validation_warnings.append(f"completely black (mean={crop_mean:.1f})")
        elif crop_mean > 252:
            is_valid_crop = False
            validation_warnings.append(f"completely white (mean={crop_mean:.1f})")
        
        # RELAXED: Only reject if VERY low variation (std < 5 = truly empty)
        if crop_std < 5:
            is_valid_crop = False
            validation_warnings.append(f"very low variation (std={crop_std:.1f}, likely empty)")
        
        if not is_valid_crop:
            logger.error(f"  ❌ n={n}: INVALID CROP - {', '.join(validation_warnings)}")
            logger.error(f"     bbox={bbox}, area={bbox_area}px²")
            logger.error(f"     REJECTING this crop - will not be saved")
            continue  # Skip saving this crop
        
        # Validation 2: Mathematical quality check (sharpness)
        crop_sharpness = calculate_sharpness_score(square_crop)
        min_crop_sharpness = 20.0  # Lower threshold for crops (they're smaller)
        
        if crop_sharpness < min_crop_sharpness:
            logger.warning(f"  ⚠️  n={n}: Crop muy borroso (nitidez={crop_sharpness:.1f} < {min_crop_sharpness:.1f})")
            # Continue anyway (log warning) - crops blurry might still be useful for identification
        
        # Validation 3: Simple mathematical check that bbox region matches crop
        # Extract bbox region from original image and compare with crop
        try:
            x, y, w, h = bbox
            img_height, img_width = image.shape[:2]
            
            # Validate bbox is within image bounds
            if x >= 0 and y >= 0 and x + w <= img_width and y + h <= img_height:
                # Extract region from original (with padding matching crop generation)
                padding = 30
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_width, x + w + padding)
                y2 = min(img_height, y + h + padding)
                
                original_region = image[y1:y2, x1:x2].copy()
                
                if original_region.size > 0:
                    # Simple validation: compare histogram correlation
                    crop_gray = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY) if len(square_crop.shape) == 3 else square_crop
                    orig_gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY) if len(original_region.shape) == 3 else original_region
                    
                    # Resize to same size for comparison
                    min_size = min(crop_gray.shape[0], crop_gray.shape[1], orig_gray.shape[0], orig_gray.shape[1])
                    if min_size > 50:
                        crop_resized = cv2.resize(crop_gray, (min_size, min_size))
                        orig_resized = cv2.resize(orig_gray, (min_size, min_size))
                        
                        # Compare histograms
                        hist_crop = cv2.calcHist([crop_resized], [0], None, [256], [0, 256])
                        hist_orig = cv2.calcHist([orig_resized], [0], None, [256], [0, 256])
                        hist_corr = cv2.compareHist(hist_crop, hist_orig, cv2.HISTCMP_CORREL)
                        
                        # If correlation is very low, crop may not match bbox
                        if hist_corr < 0.5:
                            logger.warning(f"  ⚠️  n={n}: Posible discrepancia thumbnail-bbox (correlación histograma={hist_corr:.2f})")
        except Exception as e:
            logger.debug(f"Error en validación thumbnail-bbox para n={n}: {e}")
        
        # Save crop with n as filename (n is 1-indexed, so obj_001.jpg, obj_002.jpg, etc.)
        crop_filename = f"{prefix}_{n:03d}.jpg"
        crop_path = crop_dir / crop_filename
        cv2.imwrite(str(crop_path), square_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store mapping: n → crop_path (relative to project root)
        crop_path_rel = f"images/crops/{timestamp}/{crop_filename}"
        n_to_crop[n] = crop_path_rel
        
        logger.info(f"  ✅ n={n}: bbox={bbox}, area={bbox_area}px², source={source}, mean={crop_mean:.1f}, std={crop_std:.1f}, crop={crop_filename}")
        
        if crop_mean < 20 or crop_mean > 240:
            logger.warning(f"     ⚠️  Crop may be borderline (mean={crop_mean:.1f}, std={crop_std:.1f})")
    
    logger.info(f"✂️  Generated {len(n_to_crop)} crops for useful objects")
    return n_to_crop


def _get_border_color(crop: np.ndarray) -> tuple:
    """Get average border color from crop edges for natural padding."""
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return (255, 255, 255)
    
    border_pixels = np.concatenate([
        crop[0, :].reshape(-1, 3),
        crop[-1, :].reshape(-1, 3),
        crop[:, 0].reshape(-1, 3),
        crop[:, -1].reshape(-1, 3)
    ], axis=0)
    
    avg_color = np.mean(border_pixels, axis=0).astype(np.uint8)
    return tuple(avg_color.tolist())

