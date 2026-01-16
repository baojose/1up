#!/bin/bash
# Script para ejecutar en RunPod
# Crea archivos Python directamente usando heredoc

cd ~/1UP_2


# ==========================================
# Creando detector.py
# ==========================================
cat > ~/1UP_2/detector.py << 'ENDFILE'
"""
SAM 3 Object Detector
Detects all objects in an image using Segment Anything Model 3.
Uses text prompts for concept-based detection.
Max 350 lines.
"""
import torch
import numpy as np
import cv2
import logging
import platform
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("SAM 3 not installed. Install with: pip install -e git+https://github.com/facebookresearch/sam3.git")

logger = logging.getLogger(__name__)


def enhance_for_detection(image: np.ndarray) -> np.ndarray:
    """Enhance image with CLAHE for better detection of dark objects."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)


class SAM3Detector:
    """Detects objects using SAM 3 with concept-based segmentation."""
    
    def __init__(
        self,
        device: str = "mps",
        **kwargs
    ):
        """
        Initialize SAM 3 detector.
        
        Args:
            device: Device to run on (mps, cuda, cpu)
            **kwargs: Additional parameters (for compatibility)
        """
        if not SAM3_AVAILABLE:
            raise ImportError(
                "SAM 3 not available. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e .\n"
                "  # Request access to checkpoints at SAM 3 HuggingFace repo"
            )
        
        logger.info("Loading SAM 3...")
        
        # Detect device (MPS only for Apple Silicon Macs, CUDA for NVIDIA, CPU otherwise)
        # CRITICAL: Intel Macs (pre-2020) do NOT have MPS, must use CPU
        is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.machine().lower()
        
        if device == "mps" and torch.backends.mps.is_available() and is_apple_silicon:
            actual_device = "mps"
            logger.info("✅ Using MPS (Apple Silicon GPU)")
        elif device == "cuda" and torch.cuda.is_available():
            actual_device = "cuda"
            logger.info("✅ Using CUDA (NVIDIA GPU)")
        else:
            actual_device = "cpu"
            if device != "cpu":
                if device == "mps":
                    logger.warning(f"⚠️  MPS requested but not available (Intel Mac detected)")
                    logger.warning(f"   Intel Macs (pre-2020) do not have MPS, using CPU instead")
                else:
                    logger.warning(f"⚠️  Requested device '{device}' not available, using 'cpu'")
            logger.info(f"✅ Using CPU (Intel Mac or no GPU available)")
        
        # Build SAM 3 model (checkpoints downloaded automatically from HuggingFace)
        self.model = build_sam3_image_model()
        
        # Move model to device (use .to() with proper device string)
        device_obj = torch.device(actual_device)
        self.model = self.model.to(device_obj)
        
        # Ensure all parameters and buffers are on the correct device
        for param in self.model.parameters():
            if param.device != device_obj:
                param.data = param.data.to(device_obj)
        for buffer in self.model.buffers():
            if buffer.device != device_obj:
                buffer.data = buffer.data.to(device_obj)
        
        # Initialize processor with MINIMUM confidence threshold for MAXIMUM detection
        # FILOSOFÍA: "Detectar TODO, filtrar después con Claude"
        # Default is 0.5, but we use 0.001 to catch EVERYTHING (small objects, shadows, reflections, fragments)
        # Claude will filter intelligently - we want maximum raw detections
        # Post-filtering in config.yaml is minimal, Claude does the intelligent filtering
        self.processor = Sam3Processor(
            self.model, 
            device=actual_device,
            confidence_threshold=0.001  # MÁXIMA DETECCIÓN: Detecta TODO (objetos, sombras, reflejos, fragmentos)
        )
        self.device = actual_device
        
        logger.info(f"✅ SAM 3 loaded on {actual_device}")
    
    def detect_objects(
        self, 
        image: np.ndarray,
        apply_filtering: bool = False,
        enhance_image: bool = False,
        min_area: Optional[int] = None,
        max_area_ratio: Optional[float] = None,
        min_aspect_ratio: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
        text_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects using SAM 3.
        
        Args:
            image: BGR image
            apply_filtering: Apply smart filtering
            enhance_image: Apply CLAHE enhancement
            text_prompt: Optional text prompt for concept-based detection
                        (e.g., "bag", "shoes", "electronics")
                        If None, uses automatic detection
        """
        # CRITICAL: Resize large images BEFORE processing to prevent MPS out of memory
        # SAM 3 internally resizes to 1008x1008, but loading full 4K into MPS first causes OOM
        # Store original image dimensions for bbox scaling
        original_shape = image.shape[:2]  # (height, width)
        original_h, original_w = original_shape
        
        # CRITICAL: MPS has limited memory (6.8 GB max), SAM 3 needs ~5.5 GB just for model
        # We need to process at much smaller resolution to prevent OOM
        # SAM 3 processor default is 1008x1008, but even that causes OOM on MPS
        # Use 720p (1280x720) which is enough for detection but uses less memory
        max_sam3_dimension = 720  # Reduced from 1008 to prevent MPS OOM
        max_dimension = max(original_h, original_w)
        scale = 1.0  # Initialize scale (no resizing by default)
        
        # Always resize large images (4K) to 720p for SAM 3 processing
        if max_dimension > max_sam3_dimension:
            # Calculate scale to fit within max_sam3_dimension while maintaining aspect ratio
            scale = max_sam3_dimension / max_dimension
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            logger.info(f"Resizing image for SAM 3: {original_w}x{original_h} → {new_w}x{new_h} (scale: {scale:.2f})")
            if original_w >= 3840 or original_h >= 2160:
                logger.info(f"  ✅ Original is 4K ({original_w}x{original_h}) - Crops will be extracted from 4K for maximum quality")
            logger.info(f"  (SAM 3 processes at 720p to prevent OOM, but crops use original {original_w}x{original_h} resolution)")
            
            # Resize image before enhancement (faster, better memory usage)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image
        
        if enhance_image:
            image_resized = enhance_for_detection(image_resized)
            logger.debug("Applied CLAHE enhancement")
        
        # Convert BGR to RGB and PIL Image
        rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Set image in processor
        logger.info("Running SAM 3 detection...")
        logger.debug(f"Image size: {pil_image.size}, mode: {pil_image.mode}")
        
        inference_state = self.processor.set_image(pil_image)
        
        # FILOSOFÍA: "Detectar TODO, filtrar después con Claude"
        # Usar múltiples prompts para máxima cobertura (objetos, contenedores, formas, etc.)
        all_detections = []
        
        if text_prompt:
            # Usar prompt específico si se proporciona
            logger.info(f"Using text prompt: '{text_prompt}'")
            output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            all_detections.append(output)
        else:
            # OPTIMIZACIÓN: Ajustar número de prompts según dispositivo
            # CPU (Intel Mac) = más lento, usar menos prompts para velocidad
            # MPS/CUDA = más rápido, usar más prompts para máxima detección
            if self.device == "cpu":
                # CPU: usar solo 1-2 prompts (más rápido)
                prompts = [
                    "visual",  # Detección general (objetos visuales) - suficiente en CPU
                ]
                logger.info(f"🔍 OPTIMIZED MODE (CPU): Using {len(prompts)} prompt for speed")
            else:
                # MPS/CUDA: usar múltiples prompts (máxima detección)
            prompts = [
                "visual",      # Detección general (objetos visuales)
                "container",   # Contenedores, frascos, botellas, cajas
                "object",      # Objetos genéricos
            ]
                logger.info(f"🔍 MAXIMUM DETECTION MODE ({self.device.upper()}): Using {len(prompts)} prompts to detect EVERYTHING")
            
            for prompt in prompts:
                try:
                    output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
                    if output and output.get("boxes") is not None and len(output.get("boxes", [])) > 0:
                        all_detections.append(output)
                        logger.debug(f"   Prompt '{prompt}': {len(output.get('boxes', []))} objects")
                except Exception as e:
                    logger.warning(f"   Prompt '{prompt}' failed: {e}")
                    continue
        
        # Combinar todas las detecciones
        all_masks = []
        all_boxes = []
        all_scores = []
        
        for output in all_detections:
            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")
            
            if masks is not None and boxes is not None and scores is not None:
                all_masks.append(masks)
                all_boxes.append(boxes)
                all_scores.append(scores)
        
        if not all_masks:
            logger.warning("⚠️  No detections from any prompt")
            return []
        
        # Concatenar todas las detecciones
        masks = torch.cat(all_masks, dim=0) if len(all_masks) > 1 else all_masks[0]
        boxes = torch.cat(all_boxes, dim=0) if len(all_boxes) > 1 else all_boxes[0]
        scores = torch.cat(all_scores, dim=0) if len(all_scores) > 1 else all_scores[0]
        
        num_objects = len(boxes)
        logger.info(f"🔍 SAM 3 multi-prompt detection: {num_objects} total objects (combined from {len(all_detections)} prompts)")
        
        if num_objects == 0:
            logger.warning("⚠️  SAM 3 returned 0 objects. This may indicate:")
            logger.warning("   - Image quality too low")
            logger.warning("   - No objects visible in scene")
            logger.warning("   - Confidence threshold too high")
            return []
        
        # OPTIMIZACIÓN: Limitar objetos iniciales en CPU para velocidad
        # CPU es lento procesando máscaras, limitar a top 500 por score antes de procesar
        if self.device == "cpu" and num_objects > 500:
            # Ordenar por score y tomar top 500
            sorted_indices = torch.argsort(scores, descending=True)[:500]
            masks = masks[sorted_indices]
            boxes = boxes[sorted_indices]
            scores = scores[sorted_indices]
            num_objects = 500
            logger.info(f"⚡ OPTIMIZACIÓN (CPU): Limitando a top 500 objetos por score para velocidad")
        
        logger.debug(f"Processing {num_objects} objects...")
        
        # Convert to our format
        detections = []
        
        for i in range(num_objects):
            # Extract mask (remove batch and channel dimensions)
            mask = masks[i, 0]  # Shape: [H, W]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Convert mask to boolean (SAM 3 may return float 0.0-1.0 or boolean)
            if mask_np.dtype != bool:
                # Threshold at 0.5 for float masks
                mask_np = (mask_np > 0.5).astype(bool)
            
            # Scale mask back to original image size if image was resized
            if scale != 1.0:
                # Scale mask back to original resolution
                if mask_np.shape[:2] != original_shape:
                    mask_resized = cv2.resize(
                        (mask_np.astype(np.uint8) * 255),
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_np = (mask_resized > 127).astype(bool)
            
            # Calculate area from mask (now in original resolution)
            area = int(np.sum(mask_np))
            
            # Convert box from [x0, y0, x1, y1] to [x, y, w, h]
            # Scale bbox coordinates back to original size if image was resized
            box = boxes[i].cpu().numpy()
            x0, y0, x1, y1 = box
            # Ensure coordinates are valid (non-negative)
            x0 = max(0, float(x0))
            y0 = max(0, float(y0))
            x1 = max(x0, float(x1))  # x1 must be >= x0
            y1 = max(y0, float(y1))  # y1 must be >= y0
            
            # Scale coordinates back to original image size
            if scale != 1.0:
                x0 = x0 / scale
                y0 = y0 / scale
                x1 = x1 / scale
                y1 = y1 / scale
            
            x = int(x0)
            y = int(y0)
            w = max(1, int(x1 - x0))  # Ensure width >= 1
            h = max(1, int(y1 - y0))  # Ensure height >= 1
            
            # Ensure bbox is within original image bounds
            x = max(0, min(x, original_w - 1))
            y = max(0, min(y, original_h - 1))
            w = max(1, min(w, original_w - x))
            h = max(1, min(h, original_h - y))
            
            bbox = [x, y, w, h]
            
            # Calculate mask coverage ratio (how well mask covers bbox)
            # This helps identify fragments or bad detections
            bbox_area = w * h
            coverage_ratio = area / bbox_area if bbox_area > 0 else 0.0
            
            # Extract score
            score = float(scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i])
            
            detection = {
                'id': i,
                'original_index': i,  # CRITICAL: Store original index BEFORE sorting/filtering
                'bbox': bbox,  # Now in original resolution (4K) for crops
                'mask': mask_np,  # Mask in original resolution (4K)
                'confidence': float(score),
                'area': area,  # Area in original resolution
                'coverage_ratio': float(coverage_ratio)  # Store coverage for filtering
            }
            detections.append(detection)
        
        logger.info(f"🔍 SAM 3 raw detection: {len(detections)} objects (before filtering)")
        
        # Log raw detections for debugging
        if len(detections) > 0:
            logger.debug(f"   Raw detections (first 5):")
            for i, det in enumerate(detections[:5]):
                x, y, w, h = det['bbox']
                area = det.get('area', 0)
                conf = det.get('confidence', 0.0)
                logger.debug(f"     #{i+1}: bbox=({x},{y},{w},{h}), área={area}px², conf={conf:.3f}")
        
        # Apply smart filtering if enabled
        if apply_filtering:
            detections = self._filter_detections(
                detections,
                image_shape=original_shape,  # Use original shape (4K) for filtering (bboxes are already in 4K)
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio
            )
        
        # Sort by area (biggest first) for better filtering
        detections.sort(key=lambda d: d['area'], reverse=True)
        
        # Step 0: Filter by mask coverage (reject detections with very low coverage = fragments/bad detections)
        # Low coverage (< 15%) means bbox is much larger than actual object = likely fragment or bad detection
        # Relaxed from 20% to 15% to avoid filtering valid objects with imperfect masks
        before_coverage_filter = len(detections)
        detections = [d for d in detections if d.get('coverage_ratio', 1.0) >= 0.15]
        filtered_by_coverage = before_coverage_filter - len(detections)
        if filtered_by_coverage > 0:
            logger.info(f"📊 Filtered {filtered_by_coverage} detections with very low mask coverage (< 15%)")
        
        # OPTIMIZACIÓN: Filtrado más simple y rápido en CPU
        if self.device == "cpu":
            # CPU: usar filtrado simplificado (más rápido)
            # Step 1: Filter contained boxes (más rápido que grupos)
            detections = self._filter_contained_boxes(detections)
            
            # Step 2: Sort by confidence and apply simple NMS (más rápido que grupos complejos)
            detections.sort(key=lambda d: d['confidence'], reverse=True)
            nms_threshold = nms_iou_threshold if nms_iou_threshold is not None else 0.5
            detections = self._filter_duplicates_nms(detections, iou_threshold=nms_threshold)
            
            # Step 3: Final sort by area
            detections.sort(key=lambda d: d['area'], reverse=True)
            
            # Step 4: Limitar a top 200 objetos finales en CPU para velocidad
            if len(detections) > 200:
                detections = detections[:200]
                logger.info(f"⚡ OPTIMIZACIÓN (CPU): Limitando a top 200 objetos finales para velocidad")
        else:
            # MPS/CUDA: usar filtrado completo (máxima calidad)
        # Step 1: Filter contained boxes
        detections = self._filter_contained_boxes(detections)
        
        # Step 2: Keep largest in groups
        detections = self._keep_largest_in_group(detections, iou_threshold=0.8)
        
        # Step 3: Sort by confidence for NMS
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        
        # Step 4: Apply NMS
        nms_threshold = nms_iou_threshold if nms_iou_threshold is not None else 0.5
        detections = self._filter_duplicates_nms(detections, iou_threshold=nms_threshold)
        
        # Step 5: Final sort by area
        detections.sort(key=lambda d: d['area'], reverse=True)
        
        # Reassign IDs and original_index after all filtering
        # CRITICAL: original_index must be renumbered to match final positions
        # This prevents out-of-range indices (e.g., original_index=151 when only 55 detections remain)
        for i, det in enumerate(detections):
            det['id'] = i
            det['original_index'] = i  # Reassign to match final position after filtering
        
        logger.info(f"✅ Final result: {len(detections)} quality objects")
        return detections
    
    def _filter_detections(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        min_area: Optional[int] = None,
        max_area_ratio: Optional[float] = None,
        min_aspect_ratio: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Apply intelligent filtering: area, aspect ratio, visibility."""
        h, w = image_shape
        min_area = min_area or 2000
        max_area = (h * w) * (max_area_ratio or 0.4)
        min_ar = min_aspect_ratio or 0.2
        max_ar = max_aspect_ratio or 5.0
        filtered = []
        
        # Log filtering parameters for debugging
        logger.debug(f"🔍 Filtering: min_area={min_area}, max_area={max_area} ({max_area_ratio or 0.4:.1%} of {h*w}px²)")
        
        filtered_by_area = 0
        filtered_by_aspect = 0
        filtered_by_visibility = 0
        
        for det in detections:
            area, x, y, width, height = det['area'], *det['bbox']
            
            # Filter by area (too small or too large = background/scene)
            if area < min_area:
                filtered_by_area += 1
                logger.debug(f"  ❌ Filtered (too small): area={area}px² < {min_area}px²")
                continue
            if area > max_area:
                filtered_by_area += 1
                area_ratio = area / (h * w)
                logger.info(f"  ❌ Filtered (too large/background): area={area}px² ({area_ratio:.1%} of image) > {max_area}px² ({max_area_ratio or 0.4:.1%})")
                continue
            
            if height > 0:
                ar = width / height
                if ar < min_ar or ar > max_ar:
                    filtered_by_aspect += 1
                    logger.debug(f"  ❌ Filtered (aspect ratio): ar={ar:.2f} not in [{min_ar:.2f}, {max_ar:.2f}]")
                    continue
            
            bbox_area = width * height
            if bbox_area <= 0:
                continue
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + width, w), min(y + height, h)
            visible_ratio = ((x2 - x1) * (y2 - y1)) / bbox_area
            if visible_ratio < 0.8:
                filtered_by_visibility += 1
                logger.debug(f"  ❌ Filtered (visibility): visible_ratio={visible_ratio:.2f} < 0.8")
                continue
            
            filtered.append(det)
        
        total_filtered = len(detections) - len(filtered)
        if total_filtered > 0:
            logger.info(f"📊 Filtered {total_filtered} detections: {filtered_by_area} by area, {filtered_by_aspect} by aspect, {filtered_by_visibility} by visibility")
        return filtered
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        x_left, y_top = max(x1_1, x1_2), max(y1_1, y1_2)
        x_right, y_bottom = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _filter_contained_boxes(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove small boxes completely contained inside larger ones."""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det in enumerate(detections):
            x1, y1, w1, h1 = det['bbox']
            is_contained = False
            
            for other in detections:
                if other['area'] <= det['area']:
                    continue
                x2, y2, w2, h2 = other['bbox']
                if (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    is_contained = True
                    break
            
            if not is_contained:
                filtered.append(det)
        
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.info(f"📦 Filtered contained boxes: {len(detections)} → {len(filtered)} (removed {removed})")
        return filtered
    
    def _keep_largest_in_group(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """For very high overlap groups (>80%), keep only the largest box."""
        if len(detections) <= 1:
            return detections
        
        groups, used = [], set()
        for i, det in enumerate(detections):
            if i in used:
                continue
            group = [det]
            for j, other in enumerate(detections[i + 1:], i + 1):
                if j in used:
                    continue
                if self._calculate_iou(det['bbox'], other['bbox']) > iou_threshold:
                    group.append(other)
                    used.add(j)
            groups.append(max(group, key=lambda d: d['area']))
        
        removed = len(detections) - len(groups)
        if removed > 0:
            logger.info(f"🎯 Kept largest in groups: {len(detections)} → {len(groups)} (removed {removed})")
        return groups
    
    def _filter_duplicates_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filter duplicates using NMS."""
        if len(detections) <= 1:
            return detections
        
        filtered, suppressed = [], set()
        for i, det1 in enumerate(detections):
            if i in suppressed:
                continue
            filtered.append(det1)
            for j, det2 in enumerate(detections[i + 1:], start=i + 1):
                if j not in suppressed and self._calculate_iou(det1['bbox'], det2['bbox']) > iou_threshold:
                    suppressed.add(j)
        
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.info(f"🔍 NMS filtering: {len(detections)} → {len(filtered)} (removed {removed})")
        return filtered
    
    def crop_object(
        self,
        image: np.ndarray,
        bbox: List[int],
        mask: Optional[np.ndarray] = None,
        padding: int = 30
    ) -> np.ndarray:
        """
        Crop object with padding, optionally using mask for precise cropping.
        
        Args:
            image: Full scene image
            bbox: Bounding box [x, y, w, h]
            mask: Optional boolean mask (same size as image) for precise cropping
            padding: Padding around object in pixels
            
        Returns:
            Cropped image with object centered, background removed if mask provided
        """
        x, y, w, h = bbox
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
        
        crop = image[y1:y2, x1:x2].copy()
        
        # If mask provided, apply it to remove background
        # NOTE: We keep the full bbox crop but could use mask for better centering later
        # For now, we use bbox to ensure we capture the entire object
        # The mask can be used for visualization but not for cropping (to avoid missing parts)
        
        return crop
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        max_objects: int = None
    ) -> np.ndarray:
        """Draw bounding boxes."""
        vis = image.copy()
        for i, det in enumerate(detections[:max_objects] if max_objects else detections):
            x, y, w, h = det['bbox']
            conf = det['confidence']
            color = (0, 255, 0) if conf > 0.9 else (0, 255, 255) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"#{i+1} ({conf:.2f})", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis
    
    def save_crops(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_dir: str,
        prefix: str = "obj"
    ) -> List[str]:
        """
        Save crops with standardized 1:1 aspect ratio, object centered.
        Uses masks from SAM 3 for precise cropping (removes background).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for i, det in enumerate(detections):
            # Use bbox crop (not mask) to ensure we capture the entire object
            # Masks are used for visualization, but bbox ensures complete capture
            crop = self.crop_object(image, det['bbox'])
            
            crop_height, crop_width = crop.shape[:2]
            target_size = max(crop_width, crop_height)
            min_size = 300
            if target_size < min_size:
                target_size = min_size
            
            border_color = self._get_border_color(crop)
            square_crop = np.full((target_size, target_size, 3), border_color, dtype=np.uint8)
            
            x_offset = (target_size - crop_width) // 2
            y_offset = (target_size - crop_height) // 2
            
            if crop_width > target_size or crop_height > target_size:
                scale = min(target_size / crop_width, target_size / crop_height)
                new_w = int(crop_width * scale)
                new_h = int(crop_height * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                crop_height, crop_width = crop.shape[:2]
                x_offset = (target_size - crop_width) // 2
                y_offset = (target_size - crop_height) // 2
            
            square_crop[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = crop
            
            # CRITICAL: Use original_index to preserve order before sorting/filtering
            # This ensures crops match the original SAM detection order, not the sorted order
            original_index = det.get('original_index', det['id'])
            filepath = output_path / f"{prefix}_{original_index:03d}.jpg"
            cv2.imwrite(str(filepath), square_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(str(filepath))
        
        # CRITICAL: Return crops in order of original_index (0, 1, 2, ...)
        # This ensures saved_paths[i] corresponds to the crop with original_index=i
        # Build a dictionary first, then create ordered list
        saved_paths_dict = {}
        for i, det in enumerate(detections):
            original_idx = det.get('original_index', i)
            if i < len(saved_paths):
                saved_paths_dict[original_idx] = saved_paths[i]
        
        # Create ordered list filling gaps with None
        max_original_idx = max(saved_paths_dict.keys()) if saved_paths_dict else -1
        ordered_paths = [None] * (max_original_idx + 1)
        for idx, crop_path in saved_paths_dict.items():
            ordered_paths[idx] = crop_path
        
        return ordered_paths
    
    def _get_border_color(self, crop: np.ndarray) -> tuple:
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
ENDFILE
echo '✅ detector.py creado'

# ==========================================
# Creando analyzer.py
# ==========================================
cat > ~/1UP_2/analyzer.py << 'ENDFILE'
"""
Claude Vision Analyzer
Analyzes object images using Claude Sonnet 4 vision.
Max 350 lines.
"""
import anthropic
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ClaudeAnalyzer:
    """Analyzes objects using Claude Sonnet 4 vision."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1000,
        temperature: float = 0
    ):
        """
        Initialize Claude analyzer.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 = deterministic)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"✅ Claude analyzer initialized ({model})")
    
    def analyze_scene_with_bboxes(
        self,
        scene_path: str,
        detections: List[Dict[str, Any]],
        language: str = "spanish"
    ) -> List[Dict[str, Any]]:
        """
        Analyze objects detected in complete scene.
        
        Sends ONLY the scene image + list of bounding boxes in text.
        Claude analyzes each region directly in the image.
        
        Args:
            scene_path: Path to complete scene image
            detections: List of SAM detections with bbox, area, confidence
            language: Response language
            
        Returns:
            List of analysis results (one per detection, in order)
        """
        logger.info(f"🤖 Analyzing {len(detections)} objects in scene (1 image + bboxes)...")
        
        # Encode ONLY the scene image (1 image total)
        scene_data = self._encode_image(scene_path)
        
        # Build bbox descriptions for prompt
        bbox_descriptions = []
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            area = det.get('area', w * h)
            conf = det.get('confidence', 0.0)
            bbox_descriptions.append(
                f"Objeto {i+1}: bbox [x={x}, y={y}, ancho={w}, alto={h}], "
                f"área={area}px², confianza={conf:.2f}"
            )
        
        # Create prompt with bbox list
        prompt = self._create_bbox_analysis_prompt(bbox_descriptions, len(detections), language)
        
        # Build API message (1 image + text)
        content = [
            {
                "type": "image",
                "source": scene_data
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        try:
            # Single API call with 1 image
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # Parse response
            response_text = message.content[0].text.strip()
            
            # Check if response was truncated
            if hasattr(message, 'stop_reason') and message.stop_reason == 'max_tokens':
                logger.warning(f"⚠️  Response truncated (max_tokens reached). Consider increasing max_tokens or reducing object count.")
                logger.warning(f"   Response length: {len(response_text)} chars")
            
            results = self._parse_batch_response(response_text, len(detections))
            
            logger.info(f"✅ Claude analyzed {len(results)} objects (1 API call, 1 image)")
            return results
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return self._create_fallback_batch(len(detections))
        except Exception as e:
            logger.exception("Unexpected error in scene analysis")
            return self._create_fallback_batch(len(detections))
    
    def _encode_image(self, image_path: str) -> Dict[str, str]:
        """Encode image to base64 with proper media type."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(suffix, 'image/jpeg')
        
        return {
            "type": "base64",
            "media_type": media_type,
            "data": image_data,
        }
    
    def _create_bbox_analysis_prompt(
        self,
        bbox_descriptions: List[str],
        num_objects: int,
        language: str
    ) -> str:
        """Create prompt for scene analysis with bounding boxes."""
        bboxes_text = "\n".join(bbox_descriptions)
        
        if language == "spanish":
            return f"""Analiza esta escena de un punto limpio (centro de reciclaje).

He detectado automáticamente {num_objects} objetos. Para CADA objeto, mira la región indicada por sus coordenadas en la imagen:

{bboxes_text}

Responde EXCLUSIVAMENTE con array JSON (sin markdown, sin ```json):

[
  {{"n":1, "useful":"yes", "name":"nombre específico del objeto", "category":"categoría", "condition":"excellent/good/fair/poor", "description":"descripción detallada en español (2-3 frases)", "estimated_value":"rango opcional en euros"}},
  {{"n":2, "useful":"no", "reason":"por qué no es útil"}},
  ...
]

CRITERIOS ESTRICTOS:

"useful": "yes" SOLO SI:
✅ Objeto COMPLETO y funcional (no fragmento)
✅ Tiene identidad clara y específica (NO "superficie blanca", "cosa gris", "objeto rectangular")
✅ Alguien querría llevárselo para reutilizar
✅ NO es fondo/mobiliario del punto limpio (mesa, pared, suelo, sombra)

"useful": "no" SI:
❌ Basura, papel arrugado, envoltorio
❌ Fragmento incompleto (hoja suelta, cable sin dispositivo, esquina de objeto)
❌ Fondo/mobiliario del punto limpio (mesa, pared, suelo, sombra)
❌ Partes de planta (hojas sueltas)
❌ Muy deteriorado sin posibilidad de uso
❌ Nombre genérico ("superficie", "esquina", "borde", "fragmento")

CATEGORÍAS VÁLIDAS:
furniture, electronics, books, tools, kitchenware, sports, toys, decoration, clothing, containers, other

CONDICIÓN:
- excellent: Como nuevo, sin defectos
- good: Buen estado, uso normal
- fair: Aceptable, signos de desgaste
- poor: Deteriorado, necesita reparación

IMPORTANTE:
- Responde para TODOS los {num_objects} objetos (números 1 a {num_objects})
- Sé ESTRICTO: si dudas, marca "useful": "no"
- Si useful="yes" incluye todos los campos (name, category, condition, description, estimated_value)
- Si useful="no" solo incluye n y reason
- NO incluyas markdown (```json)
- NO incluyas texto adicional fuera del array JSON
- El array debe tener exactamente {num_objects} elementos"""
        else:  # English
            return f"""Analyze this recycling center scene.

I've automatically detected {num_objects} objects. For EACH object, look at the region indicated by its coordinates in the image:

{bboxes_text}

Respond EXCLUSIVELY with JSON array (no markdown, no ```json):

[
  {{"n":1, "useful":"yes", "name":"specific object name", "category":"category", "condition":"excellent/good/fair/poor", "description":"detailed description (2-3 sentences)", "estimated_value":"optional price range"}},
  {{"n":2, "useful":"no", "reason":"why not useful"}},
  ...
]

STRICT CRITERIA:

"useful": "yes" ONLY IF:
✅ Complete and functional object (not fragment)
✅ Clear, specific identity (NOT "white surface", "gray thing", "rectangular object")
✅ Someone would want to take it for reuse
✅ NOT background/furniture of recycling center (table, wall, floor, shadow)

"useful": "no" IF:
❌ Trash, crumpled paper, wrapper
❌ Incomplete fragment (loose sheet, cable without device, object corner)
❌ Background/furniture of recycling center (table, wall, floor, shadow)
❌ Plant parts (loose leaves)
❌ Too deteriorated without possibility of use
❌ Generic name ("surface", "corner", "edge", "fragment")

VALID CATEGORIES:
furniture, electronics, books, tools, kitchenware, sports, toys, decoration, clothing, containers, other

CONDITION:
- excellent: Like new, no defects
- good: Good condition, normal wear
- fair: Acceptable, signs of wear
- poor: Deteriorated, needs repair

IMPORTANT:
- Respond for ALL {num_objects} objects (numbers 1 to {num_objects})
- Be STRICT: if in doubt, mark "useful": "no"
- If useful="yes" include all fields (name, category, condition, description, estimated_value)
- If useful="no" only include n and reason
- NO markdown (```json)
- NO additional text outside JSON array
- Array must have exactly {num_objects} elements"""
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse Claude's batch JSON response."""
        # Remove markdown code blocks if present
        text = response_text.strip()
        
        # Try to extract JSON from markdown code blocks
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    text = part[4:].strip()
                    break
                elif part.startswith('['):
                    text = part
                    break
        
        # Find JSON array (may have text before/after)
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error("No JSON array found in response")
            logger.debug(f"Response preview (first 1000 chars): {text[:1000]}")
            logger.debug(f"Response length: {len(text)} chars")
            return self._create_fallback_batch(expected_count)
        
        json_text = text[start_idx:end_idx + 1]
        
        try:
            results = json.loads(json_text)
            
            if not isinstance(results, list):
                logger.error(f"Expected list, got {type(results)}")
                return self._create_fallback_batch(expected_count)
            
            # Validate and normalize results
            normalized = []
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    logger.warning(f"Result {i} is not a dict, skipping")
                    continue
                
                # Ensure 'n' field matches index
                if 'n' not in result:
                    result['n'] = i + 1
                
                # Normalize useful field
                if 'useful' not in result:
                    result['useful'] = 'no'
                    result['reason'] = 'Missing useful field'
                
                normalized.append(result)
            
            if len(normalized) != expected_count:
                logger.warning(
                    f"Expected {expected_count} results, got {len(normalized)}. "
                    "Padding with fallback entries."
                )
                while len(normalized) < expected_count:
                    normalized.append({
                        'n': len(normalized) + 1,
                        'useful': 'no',
                        'reason': 'Missing from Claude response'
                    })
            
            return normalized[:expected_count]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Error at position {e.pos}: {e.msg}")
            logger.debug(f"JSON text around error (chars {max(0, e.pos-100)}-{min(len(json_text), e.pos+100)}): {json_text[max(0, e.pos-100):min(len(json_text), e.pos+100)]}")
            logger.debug(f"Full JSON text length: {len(json_text)} chars")
            # Try to save response for debugging
            try:
                import os
                debug_file = "claude_response_debug.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Original response:\n{response_text}\n\n")
                    f.write(f"Extracted JSON:\n{json_text}\n")
                logger.debug(f"Saved debug response to {debug_file}")
            except:
                pass
            return self._create_fallback_batch(expected_count)
    
    def _create_fallback_batch(self, count: int) -> List[Dict[str, Any]]:
        """Create fallback batch results if Claude fails."""
        return [
            {
                'n': i + 1,
                'useful': 'no',
                'reason': 'Analysis failed. Please review manually.',
                'error': True
            }
            for i in range(count)
        ]
    
    def analyze_scene_with_validation(
        self,
        scene_path: str,
        detections: List[Dict[str, Any]],
        language: str = "spanish"
    ) -> Dict[str, Any]:
        """
        Analyze scene with validation and missing object detection.
        
        Claude validates detected objects AND suggests missing objects
        that SAM didn't detect (e.g., white objects on light backgrounds).
        
        Args:
            scene_path: Path to complete scene image
            detections: List of SAM detections with bbox, area, confidence
            language: Response language
            
        Returns:
            Dict with:
            - validated_objects: List of validated analyses (same format as analyze_scene_with_bboxes)
            - missing_objects: List of missing objects with approximate bboxes
        """
        logger.info(f"🔍 Validating {len(detections)} objects + searching for missing objects...")
        
        # Encode scene image
        scene_data = self._encode_image(scene_path)
        
        # Get image dimensions for relative size calculation
        import cv2
        img = cv2.imread(scene_path)
        img_height, img_width = img.shape[:2] if img is not None else (960, 1280)
        total_pixels = img_width * img_height
        
        # Build bbox descriptions with detailed info including relative size
        bbox_descriptions = []
        logger.info(f"📤 Preparando {len(detections)} objetos para Claude:")
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            area = det.get('area', w * h)
            conf = det.get('confidence', 0.0)
            area_percent = (area / total_pixels) * 100 if total_pixels > 0 else 0
            bbox_descriptions.append(
                f"Objeto {i+1}: bbox [x={x}, y={y}, ancho={w}, alto={h}], "
                f"área={area}px² ({area_percent:.1f}% de la imagen), confianza={conf:.2f}"
            )
            logger.info(f"   Objeto {i+1}: bbox=({x},{y},{w},{h}), área={area}px² ({area_percent:.1f}%), conf={conf:.3f}")
        
        logger.info("")
        
        # Create validation prompt
        prompt = self._create_validation_prompt(bbox_descriptions, len(detections), language)
        
        # Build API message
        content = [
            {
                "type": "image",
                "source": scene_data
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            response_text = message.content[0].text.strip()
            result = self._parse_validation_response(response_text, len(detections))
            
            validated_count = len(result.get('validated_objects', []))
            missing_count = len(result.get('missing_objects', []))
            
            logger.info(f"✅ Validation complete: {validated_count} validated, {missing_count} missing found")
            
            # CRITICAL: Warn if Claude didn't validate all objects
            if validated_count < len(detections):
                logger.warning(f"⚠️  Claude only validated {validated_count}/{len(detections)} objects!")
                logger.warning(f"   Expected {len(detections)} validated objects, got {validated_count}")
                logger.warning(f"   Missing {len(detections) - validated_count} validations")
            
            return result
            
        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            return {
                'validated_objects': self._create_fallback_batch(len(detections)),
                'missing_objects': []
            }
    
    def _create_validation_prompt(
        self,
        bbox_descriptions: List[str],
        num_objects: int,
        language: str
    ) -> str:
        """Create prompt for validation + analysis + missing object detection (ALL IN ONE)."""
        bboxes_text = "\n".join(bbox_descriptions)
        
        if language == "spanish":
            return f"""Analiza esta escena de un punto limpio. El sistema detecta TODO (sombras, fragmentos, fondo). Tu trabajo: identificar TODOS los objetos útiles completos.

He detectado {num_objects} regiones con estas coordenadas:

{bboxes_text}

TAREAS:
1. VALIDAR cada región: mira el bbox [x, y, ancho, alto], identifica el objeto usando el tamaño como contexto:
   - Grandes (5-15%): muebles, electrodomésticos
   - Medianos (1-5%): libros, bolsos, ropa, contenedores
   - Pequeños (0.1-1%): frascos, botellas, juguetes, objetos decorativos
   - <0.1%: fragmentos (rechazar)
   
   Si useful="yes": name, category, condition, description, estimated_value, bbox [x,y,w,h], bbox_confidence
   Si useful="no": reason

2. BUSCAR objetos faltantes/superpuestos no detectados.

Responde SOLO JSON (sin markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "Pesa negra", "category": "sports", "condition": "good", "description": "Pesa de entrenamiento negra redonda", "estimated_value": "20-40€", "bbox": [x,y,w,h], "bbox_confidence": "high"}},
    {{"n": 2, "useful": "no", "reason": "Fragmento de fondo"}},
    ...
  ],
  "missing_objects": [
    {{"name": "frasco", "bbox": [x,y,w,h], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIOS:
✅ Útil: objeto completo funcional, grupos similares ("Especiero con 6 frascos" NO 6 separados)
✅ Útil: contenedores, botellas, frascos, objetos decorativos, plantas, utensilios de cocina
❌ No útil: fragmentos, sombras, fondo (pared/suelo/mesa), duplicados, partes del cuerpo

AGRUPACIÓN CRÍTICA (Muy importante):
- Si hay 3+ objetos similares superpuestos/cercanos → agrupa en 1 objeto grupal con cantidad
- Ejemplo: Si ves Objetos 3, 4, 5, 6, 7, 8, 9 que son 7 frascos de especias juntos:
  * Responde para TODOS: Objeto 3, Objeto 4, Objeto 5, Objeto 6, Objeto 7, Objeto 8, Objeto 9
  * Para el PRIMER objeto del grupo (ej: Objeto 3): useful="yes", name="Especiero con 7 frascos de especias", bbox=[bbox del grupo completo]
  * Para los demás del grupo (ej: Objetos 4-9): useful="no", reason="Agrupado en Especiero con 7 frascos"

BBOX PARA OBJETOS AGRUPADOS (CRÍTICO):
- Si agrupas objetos: el bbox del objeto agrupado debe cubrir TODO el grupo completo
- Ejemplo: Si agrupas Objetos 3, 4, 5, 6, 7, 8, 9 en "Especiero con 7 frascos":
  * Calcula el bbox que contiene TODOS los bboxes de esos objetos
  * min_x = mínimo x de todos los bboxes
  * min_y = mínimo y de todos los bboxes  
  * max_x = máximo (x+w) de todos los bboxes
  * max_y = máximo (y+h) de todos los bboxes
  * bbox del grupo = [min_x, min_y, max_x-min_x, max_y-min_y]
- El bbox del grupo debe ser más grande que cualquiera de los bboxes individuales

BBOX: [x,y,width,height] SIEMPRE presente si useful="yes" (OBLIGATORIO, puede ser aproximado pero debe existir).
⚠️ CRÍTICO: Si useful="yes", DEBES proporcionar bbox. Si agrupaste objetos, usa el bbox del grupo completo.

IMPORTANTE: Sé INCLUSIVO. Si un objeto es claramente visible y útil, márcalo como useful="yes", incluso si es pequeño o parcialmente oculto.

RESPONDE TODOS los {num_objects} objetos. Sé INCLUSIVO pero INTELIGENTE."""
        else:  # English
            return f"""Analyze this recycling center scene. System detects EVERYTHING (shadows, fragments, background). Your job: identify ONLY complete useful objects.

I've detected {num_objects} regions with these coordinates:

{bboxes_text}

TASKS:
1. VALIDATE each region: look at bbox [x, y, width, height], identify object using size as context:
   - Large (5-15%): furniture, appliances
   - Medium (1-5%): books, bags, clothing
   - Small (0.1-1%): jars, bottles, toys
   - <0.1%: fragments (reject)
   
   If useful="yes": name, category, condition, description, estimated_value, bbox [x,y,w,h], bbox_confidence
   If useful="no": reason

2. FIND missing/overlapping objects not detected.

Respond ONLY JSON (no markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "Black weight", "category": "sports", "condition": "good", "description": "Black round training weight", "estimated_value": "20-40€", "bbox": [x,y,w,h], "bbox_confidence": "high"}},
    {{"n": 2, "useful": "no", "reason": "Background fragment"}},
    ...
  ],
  "missing_objects": [
    {{"name": "medicine bottle", "bbox": [x,y,w,h], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIA:
✅ Useful: complete functional object, similar groups ("Spice rack with 6 jars" NOT 6 separate)
❌ Not useful: fragments, shadows, background (wall/floor/table), duplicates, body parts

GROUPING: If 3+ similar overlapping objects → 1 group object with quantity.

BBOX: [x,y,width,height] always present if useful="yes" (may be approximate).

RESPOND to ALL {num_objects} objects. Mark "useful":"no" for 60-70% (system noise). Be STRICT."""
    
    def _parse_validation_response(
        self,
        response_text: str,
        expected_count: int
    ) -> Dict[str, Any]:
        """Parse Claude's validation response with validated_objects and missing_objects."""
        text = response_text.strip()
        
        # Remove markdown
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    text = part[4:].strip()
                    break
                elif part.startswith('{'):
                    text = part
                    break
        
        # Find JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error("❌ No JSON object found in validation response")
            logger.debug(f"Response text (first 500 chars): {text[:500]}")
            return {
                'validated_objects': [],
                'missing_objects': []
            }
        
        json_text = text[start_idx:end_idx + 1]
        
        try:
            result = json.loads(json_text)
            
            # Validate structure
            validated = result.get('validated_objects', [])
            missing = result.get('missing_objects', [])
            
            # CRITICAL: Do NOT pad missing objects - only use what Claude validated
            # If Claude didn't validate an object, it means it's likely not useful
            if len(validated) != expected_count:
                logger.warning(f"⚠️  Expected {expected_count} validated objects, got {len(validated)}")
                logger.warning(f"   Claude did not validate {expected_count - len(validated)} objects")
                logger.warning(f"   These objects will be SKIPPED (not saved)")
                # DO NOT pad - only use what Claude actually validated
            
            # Validate missing_objects format
            validated_missing = []
            for missing in missing:
                if isinstance(missing, dict) and 'bbox' in missing and 'name' in missing:
                    bbox = missing['bbox']
                    if isinstance(bbox, list) and len(bbox) == 4:
                        validated_missing.append(missing)
                    else:
                        logger.warning(f"Invalid bbox format in missing object: {missing.get('name')}")
                else:
                    logger.warning(f"Invalid missing object format: {missing}")
            
            return {
                'validated_objects': validated,
                'missing_objects': validated_missing
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse validation JSON: {e}")
            logger.error(f"   JSON text (first 1000 chars): {json_text[:1000]}")
            logger.error(f"   This may indicate Claude's response format is incorrect")
            return {
                'validated_objects': [],
                'missing_objects': []
            }

ENDFILE
echo '✅ analyzer.py creado'

# ==========================================
# Creando filters.py
# ==========================================
cat > ~/1UP_2/filters.py << 'ENDFILE'
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


def filter_objects(
    analyses: List[Dict[str, Any]],
    image_shape: tuple,
    max_area_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Simple filter for analyses (post-Claude filtering).
    Filters by size and generic names.
    
    Args:
        analyses: List of Claude analyses (already filtered by useful="yes")
        image_shape: (height, width) of image
        max_area_ratio: Maximum area ratio (default: 0.5 = 50%)
    
    Returns:
        Filtered list of analyses
    """
    filtered = []
    
    for analysis in analyses:
        # Filter by size
        bbox = analysis.get('bbox')
        if bbox and filter_by_size(bbox, image_shape, max_area_ratio):
            continue
        
        # Filter by generic names
        obj_name = analysis.get('name', '')
        if filter_generic_names(obj_name):
            continue
        
        filtered.append(analysis)
    
    return filtered

ENDFILE
echo '✅ filters.py creado'

# ==========================================
# Creando image_quality.py
# ==========================================
cat > ~/1UP_2/image_quality.py << 'ENDFILE'
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
ENDFILE
echo '✅ image_quality.py creado'

# ==========================================
# Creando camera_utils.py
# ==========================================
cat > ~/1UP_2/camera_utils.py << 'ENDFILE'
"""
Camera utilities for 1UP
Handles camera detection and enumeration, especially for macOS.
Max 350 lines.
"""
import cv2
import logging
import platform
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union

# Try to import cv2-enumerate-cameras (best method for macOS)
try:
    from cv2_enumerate_cameras import enumerate_cameras as cv2_enumerate_cameras
    HAS_ENUMERATE = True
except ImportError:
    HAS_ENUMERATE = False
    cv2_enumerate_cameras = None

logger = logging.getLogger(__name__)


def enumerate_cameras(max_index: int = 10, allow_iphone: bool = False) -> List[Dict[str, any]]:
    """
    Enumerate all available cameras.
    Works around macOS OpenCV index bug.
    By default excludes iPhone/Continuity Camera to avoid interfering with phone.
    
    Args:
        max_index: Maximum camera index to check
        allow_iphone: If True, allow iPhone/Continuity Camera (default: False)
        
    Returns:
        List of available cameras with their properties
    """
    available_cameras = []
    
    # Try to get camera names to filter iPhone/Continuity
    camera_names = {}
    if HAS_ENUMERATE:
        try:
            is_mac = platform.system() == 'Darwin'
            backend = cv2.CAP_AVFOUNDATION if is_mac else None
            for cam_info in cv2_enumerate_cameras(backend):
                camera_names[cam_info.index] = cam_info.name.upper()
        except:
            pass
    
    # Only exclude iPhone if allow_iphone is False
    exclude_keywords = []
    if not allow_iphone:
        exclude_keywords = ['IPHONE', 'CONTINUITY', 'FACETIME', 'IPAD']
    
    logger.info(f"Scanning for cameras (0 to {max_index})...")
    
    # Use correct backend for macOS to avoid OpenCV error messages
    is_mac = platform.system() == 'Darwin'
    
    # Suppress OpenCV warnings during enumeration (it's normal for some indices to not exist)
    # Save current log level and set to 0 (SILENT) to suppress warnings
    # 0 = SILENT, 1 = ERROR, 2 = WARN, 3 = INFO, 4 = DEBUG
    original_log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)  # SILENT - suppress all warnings during enumeration
    
    try:
        for i in range(max_index):
            # Skip if we know it's an iPhone/Continuity Camera
            if i in camera_names:
                cam_name = camera_names[i]
                if any(exclude in cam_name for exclude in exclude_keywords):
                    logger.info(f"  ⏭️  Camera {i}: Skipping {camera_names[i]} (iPhone/Continuity Camera)")
                    continue
            
            # Use backend from the start to avoid OpenCV trying multiple backends
            # Only pass backend if it's not None (OpenCV doesn't accept None as second arg)
            if is_mac:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Try to read a frame to verify it's working
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Try to get backend name
                    backend_name = cap.getBackendName()
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend_name,
                        'working': True
                    }
                    
                    available_cameras.append(camera_info)
                    logger.info(f"  ✅ Camera {i}: {width}x{height} @ {fps:.1f}fps ({backend_name})")
                else:
                    logger.debug(f"  ⚠️  Camera {i}: Opened but can't read frames")
                
                cap.release()
            else:
                logger.debug(f"  ❌ Camera {i}: Not available")
    finally:
        # Restore original log level
        cv2.setLogLevel(original_log_level)
    
    if allow_iphone:
        logger.info(f"Found {len(available_cameras)} working camera(s) (iPhone/Continuity allowed)")
    else:
        logger.info(f"Found {len(available_cameras)} working camera(s) (iPhone/Continuity excluded)")
    return available_cameras


def find_camera_by_name(name_keywords: List[str], backend=None, allow_iphone: bool = False) -> Optional[Tuple[int, cv2.VideoCapture]]:
    """
    Find camera by name using cv2-enumerate-cameras (most reliable on macOS).
    By default excludes iPhone/Continuity Camera to avoid interfering with phone.
    
    Args:
        name_keywords: List of keywords to search for (e.g., ["C270", "UVC", "Logitech"])
        backend: OpenCV backend to use
        allow_iphone: If True, allow iPhone/Continuity Camera (default: False)
        
    Returns:
        (camera_index, cap) or (None, None) if not found
    """
    if not HAS_ENUMERATE:
        logger.debug("cv2-enumerate-cameras not available, skipping name search")
        return None, None
    
    is_mac = platform.system() == 'Darwin'
    if backend is None and is_mac:
        backend = cv2.CAP_AVFOUNDATION
    
    # Keywords to EXCLUDE (iPhone, Continuity Camera, FaceTime)
    # Only exclude if allow_iphone is False
    exclude_keywords = []
    if not allow_iphone:
        exclude_keywords = ['IPHONE', 'CONTINUITY', 'FACETIME', 'IPAD']
    
    logger.info(f"Searching for camera by name: {name_keywords}...")
    if exclude_keywords:
        logger.info(f"Excluding: {exclude_keywords}")
    else:
        logger.info("iPhone/Continuity Camera allowed")
    
    try:
        for camera_info in cv2_enumerate_cameras(backend):
            camera_name = camera_info.name.upper()
            logger.info(f"  Checking camera {camera_info.index}: {camera_info.name}")
            
            # Skip iPhone/Continuity Camera
            if any(exclude in camera_name for exclude in exclude_keywords):
                logger.info(f"  ⏭️  Skipping {camera_info.name} (iPhone/Continuity Camera)")
                continue
            
            for keyword in name_keywords:
                if keyword.upper() in camera_name:
                    logger.info(f"✅ Found! {camera_info.name} at index {camera_info.index} (backend: {camera_info.backend})")
                    
                    # Use the backend from enumerate_cameras (important on macOS!)
                    cap = cv2.VideoCapture(camera_info.index, camera_info.backend)
                    if cap.isOpened():
                        # Set resolution and test
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        
                        # Read a few frames to ensure it works
                        for _ in range(5):
                            ret, _ = cap.read()
                            if not ret:
                                break
                            time.sleep(0.05)
                        
                        # Verify we got the right camera by checking resolution
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        logger.info(f"   Camera opened: {actual_width}x{actual_height}")
                        
                        return camera_info.index, cap
                    else:
                        logger.warning(f"  ⚠️  Could not open camera {camera_info.index}")
        
        logger.info(f"  ❌ No camera found with names: {name_keywords}")
        return None, None
        
    except Exception as e:
        logger.warning(f"  ⚠️  Error using cv2-enumerate-cameras: {e}")
        return None, None


def find_camera_by_resolution(target_width: int = 1920, target_height: int = 1080, backend=None) -> Optional[Tuple[int, cv2.VideoCapture]]:
    """
    Find camera with specific resolution (fallback method).
    
    Args:
        target_width: Expected width
        target_height: Expected height
        backend: OpenCV backend to use
        
    Returns:
        (camera_index, cap) or (None, None) if not found
    """
    is_mac = platform.system() == 'Darwin'
    if backend is None and is_mac:
        backend = cv2.CAP_AVFOUNDATION
    
    logger.info(f"Searching for camera with resolution {target_width}x{target_height}...")
    
    for index in range(6):
        try:
            if backend:
                cap = cv2.VideoCapture(index, backend)
            else:
                cap = cv2.VideoCapture(index)
            
            if not cap.isOpened():
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            
            # Read frames to get actual resolution
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    break
                time.sleep(0.05)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.debug(f"  Camera {index}: {actual_width}x{actual_height}")
            
            if actual_width == target_width and actual_height == target_height:
                logger.info(f"✅ Found! Camera {index} has {target_width}x{target_height}")
                return index, cap
            
            cap.release()
            time.sleep(0.1)
            
        except Exception as e:
            logger.debug(f"  ⚠️  Error testing camera {index}: {e}")
            continue
    
    logger.info(f"  ❌ No camera found with resolution {target_width}x{target_height}")
    return None, None


def find_external_camera(preferred_index: Optional[int] = None, allow_iphone: bool = False) -> Optional[int]:
    """
    Find external camera using multiple methods (most reliable on macOS).
    
    Strategy:
    1. If preferred_index specified, try to use it directly (even if iPhone)
    2. Try to find by name (C270, UVC, Logitech) - most reliable on macOS
    3. Try to find by resolution (1920x1080) - fallback
    4. Use any non-zero camera - last resort
    
    Args:
        preferred_index: If provided, try this index first (even if iPhone)
        allow_iphone: If True, allow iPhone/Continuity Camera (default: False)
        
    Returns:
        Camera index if found, None otherwise
    """
    is_mac = platform.system() == 'Darwin'
    
    # Method 0: If preferred_index is specified, try it directly (even if iPhone)
    # This allows user to explicitly use iPhone by setting index in config.yaml
    if preferred_index is not None:
        logger.info(f"Trying preferred camera index {preferred_index} (iPhone allowed if specified)...")
        backend = cv2.CAP_AVFOUNDATION if is_mac else None
        
        if backend:
            cap = cv2.VideoCapture(preferred_index, backend)
        else:
            cap = cv2.VideoCapture(preferred_index)
        
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                logger.info(f"✅ Using preferred camera {preferred_index} ({width}x{height})")
                return preferred_index
            cap.release()
        logger.warning(f"Preferred camera {preferred_index} not working, trying other methods...")
    
    # Method 1: Search by name FIRST (most reliable on macOS, ignores index confusion)
    # This is the key - we prioritize name search over index to avoid macOS index bugs
    # Only exclude iPhone if allow_iphone is False
    if HAS_ENUMERATE:
        logger.info("🔍 Searching for camera by name (C270, UVC, Logitech)...")
        search_keywords = ["C270", "UVC", "Logitech", "USB", "External"]
        if allow_iphone:
            search_keywords.extend(["iPhone", "Continuity"])
        index, cap = find_camera_by_name(search_keywords, allow_iphone=allow_iphone)
        if cap:
            # Get actual resolution to verify
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"✅ Found external camera by name at index {index} ({actual_w}x{actual_h})")
            cap.release()
            return index
    
    
    # Method 3: Search by resolution (fallback)
    
    # Method 3: Search by resolution (1920x1080 = external camera)
    logger.info("🔍 Searching for camera by resolution (1920x1080)...")
    index, cap = find_camera_by_resolution(1920, 1080)
    if cap:
        cap.release()
        return index
    
    # Method 4: Fallback - use any non-zero camera (iPhone already excluded by enumerate_cameras)
    logger.info("Fallback: Looking for any non-built-in camera...")
    cameras = enumerate_cameras()
    external_cameras = [cam for cam in cameras if cam['index'] != 0]
    
    if external_cameras:
        best = max(external_cameras, key=lambda c: c['width'] * c['height'])
        logger.info(f"✅ Using external camera {best['index']} ({best['width']}x{best['height']})")
        return best['index']
    
    # Last resort: use any available camera (including built-in if that's all we have)
    if cameras:
        fallback = cameras[0]['index']
        if fallback == 0:
            logger.warning(f"⚠️  Only built-in camera (index 0) available - using it as fallback")
        else:
            logger.info(f"Using camera {fallback} (only camera available)")
        return fallback
    
    logger.error("❌ No cameras found")
    return None


def _read_frame_with_timeout(cap: cv2.VideoCapture, timeout: float = 5.0) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Reads a frame from the camera with a timeout.
    
    Args:
        cap: OpenCV VideoCapture object
        timeout: Maximum time to wait for frame (seconds)
        
    Returns:
        Tuple of (success: bool, frame: Optional[np.ndarray])
    """
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            return True, frame
        if time.time() - start_time > timeout:
            logger.warning(f"⚠️  Timeout ({timeout:.1f}s) reached while reading frame.")
            return False, None
        time.sleep(0.01)  # Small delay to prevent busy-waiting


def open_camera(
    source: Union[int, str],
    resolution: Optional[Tuple[int, int]] = None,
    buffer_size: Optional[int] = None,
    backend: Optional[int] = None
) -> Optional[cv2.VideoCapture]:
    """
    Open camera from source (index or RTSP URL).
    
    Args:
        source: Camera source - can be:
            - int: Camera index (0, 1, 2...) for USB/webcam
            - str: RTSP URL (e.g., "rtsp://user:pass@ip:port/stream") for IP cameras
        resolution: Optional (width, height) tuple to set resolution
        buffer_size: Optional buffer size (important for RTSP streams, use 1 for low latency)
        backend: Optional OpenCV backend (e.g., cv2.CAP_AVFOUNDATION for macOS)
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    is_rtsp = isinstance(source, str) and source.startswith('rtsp://')
    
    if is_rtsp:
        logger.info(f"Opening RTSP stream: {source[:50]}...")
        
        # For RTSP, add options to prevent blocking/timeouts
        # Use TCP transport for reliability (UDP can be faster but less reliable)
        # Note: Some cameras may not support TCP, so we try both
        rtsp_with_tcp = f"{source}?rtsp_transport=tcp"
        
        # Try with TCP first (more reliable but potentially slower)
        cap = cv2.VideoCapture(rtsp_with_tcp, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            # Try without TCP transport as fallback
            logger.warning("Failed with TCP transport, trying default...")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {source[:50]}...")
            return None
        
        # Set buffer size for RTSP (critical for low latency)
        # For RTSP, buffer_size=1 prevents accumulation of old frames
        if buffer_size is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            logger.info(f"Set RTSP buffer size: {buffer_size}")
        
        # Set timeout properties for RTSP (in milliseconds)
        # OpenCV doesn't support direct timeout, but we can set read timeout via environment
        # For now, we rely on buffer_size=1 to minimize delay
        
        # Set resolution if provided
        if resolution:
            w, h = resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            logger.info(f"Requested resolution: {w}x{h}")
        
        # Verify connection by reading a frame
        # ⚠️ IMPORTANTE: Para 4K HEVC en Mac Intel, el primer frame puede tardar más (10-15s)
        # Los errores HEVC son warnings de FFmpeg, no bloquean, pero ralentizan la decodificación
        logger.info("Reading first frame (this may take 10-15s for 4K HEVC on Mac Intel)...")
        ret, frame = _read_frame_with_timeout(cap, timeout=15.0)  # Increased timeout for 4K HEVC
        
        if not ret or frame is None:
            logger.warning("⚠️  RTSP stream opened but first frame read failed after 15s timeout")
            logger.warning("   For 4K HEVC on Mac Intel, this may be normal (HEVC decoding is slow)")
            logger.warning("   The system will retry automatically in the main loop")
            logger.warning("   ⚠️  Note: HEVC errors in logs are warnings, not fatal - processing continues")
            # Don't fail immediately - let the main loop handle retries
            # Some RTSP streams need a few attempts to start, especially 4K HEVC
        else:
            logger.info(f"✅ First frame read successfully: {frame.shape[1]}x{frame.shape[0]}")
        
        # Still return the cap even if first frame fails - main loop will handle retries
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"✅ RTSP stream opened: {actual_w}x{actual_h}")
        return cap
    
    else:
        # Integer index - USB/webcam camera
        camera_index = int(source)
        logger.info(f"Opening camera index: {camera_index}")
        
        # Use backend if provided (important for macOS)
        if backend is not None:
            cap = cv2.VideoCapture(camera_index, backend)
        else:
            # Auto-detect backend for macOS
            if platform.system() == 'Darwin':
                cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera index {camera_index}")
            return None
        
        # Set resolution if provided
        if resolution:
            w, h = resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        # Verify by reading a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error(f"Camera {camera_index} opened but cannot read frames")
            cap.release()
            return None
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"✅ Camera {camera_index} opened: {actual_w}x{actual_h}")
        return cap


def test_camera(index: int, duration_seconds: int = 5) -> bool:
    """
    Test camera by capturing frames for specified duration.
    
    Args:
        index: Camera index to test
        duration_seconds: How long to test
        
    Returns:
        True if camera works, False otherwise
    """
    logger.info(f"Testing camera {index} for {duration_seconds} seconds...")
    
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {index}")
        return False
    
    import time
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.error(f"Failed to read frame from camera {index}")
                return False
            
            frame_count += 1
            
            # Show preview
            cv2.imshow(f'Camera {index} Test - Press Q to skip', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        fps = frame_count / duration_seconds
        logger.info(f"✅ Camera {index} working: {frame_count} frames in {duration_seconds}s ({fps:.1f} fps)")
        return True
        
    except Exception as e:
        logger.error(f"Error testing camera {index}: {e}")
        return False
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

ENDFILE
echo '✅ camera_utils.py creado'

# ==========================================
# Creando storage_v2.py
# ==========================================
cat > ~/1UP_2/storage_v2.py << 'ENDFILE'
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
        
        # STANDARDIZED CROP SIZE: Todos los crops tienen el mismo tamaño
        # Tamaño estándar: 512x512 (suficiente para e-commerce, buena calidad)
        STANDARD_CROP_SIZE = 512
        
        crop_height, crop_width = crop.shape[:2]
        
        # Calculate scale to fit crop into standard size while maintaining aspect ratio
        # Object will be centered in the 512x512 square
        scale = min(STANDARD_CROP_SIZE / crop_width, STANDARD_CROP_SIZE / crop_height)
        new_w = int(crop_width * scale)
        new_h = int(crop_height * scale)
        
        # Resize crop to fit within standard size (maintains aspect ratio)
        if new_w != crop_width or new_h != crop_height:
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            crop_resized = crop.copy()
        
        # Get average border color for padding (natural look)
        border_color = _get_border_color(crop_resized)
        
        # Create standard-sized square crop (512x512) with object centered
        square_crop = np.full((STANDARD_CROP_SIZE, STANDARD_CROP_SIZE, 3), border_color, dtype=np.uint8)
        
        # Center the resized crop in the square
        x_offset = (STANDARD_CROP_SIZE - new_w) // 2
        y_offset = (STANDARD_CROP_SIZE - new_h) // 2
        
        # Place resized crop centered in square
        square_crop[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = crop_resized
        
        # Crop is now standardized: 512x512, object centered
        
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

ENDFILE
echo '✅ storage_v2.py creado'

# ==========================================
# Creando storage.py
# ==========================================
cat > ~/1UP_2/storage.py << 'ENDFILE'
"""
Storage utilities for 1UP
Handles saving images, crops, and metadata.
Prepared for future multi-camera, multi-user, multi-system support.
Max 350 lines.
"""
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImageStorage:
    """Handles saving images and crops with metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize storage with configuration.
        
        Args:
            config: Configuration dict with paths and settings
        """
        self.config = config
        self.raw_images_dir = Path(config['paths']['raw_images'])
        self.crops_dir = Path(config['paths']['crops'])
        
        # Create directories
        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Storage initialized: raw={self.raw_images_dir}, crops={self.crops_dir}")
    
    def save_scene(
        self,
        image: cv2.Mat,
        detections: List[Dict[str, Any]],
        camera_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save complete scene with all crops.
        
        Args:
            image: Full scene image
            detections: List of detections
            camera_id: Camera identifier (for future multi-camera)
            user_id: User identifier (for future multi-user)
            system_id: System identifier (for future multi-system)
            metadata: Additional metadata
            
        Returns:
            Dict with saved paths and scene info
        """
        # ISO 8601 format: YYYY-MM-DD_HH-MM-SS (sortable, readable, filesystem-safe)
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        timestamp_iso = now.isoformat()  # Full ISO 8601 for metadata
        
        # Build scene ID with optional identifiers
        # Format: [CAM{camera_id}_]YYYY-MM-DD_HH-MM-SS
        scene_parts = [timestamp]
        if camera_id:
            scene_parts.insert(0, f"cam{camera_id}")
        if user_id:
            scene_parts.insert(0, f"user{user_id}")
        if system_id:
            scene_parts.insert(0, f"sys{system_id}")
        
        scene_id = "_".join(scene_parts)
        
        logger.info(f"💾 Saving scene: {scene_id}")
        
        # Verify image is valid
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image: shape={image.shape if image is not None else None}")
        
        logger.info(f"  📸 Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Save raw image
        raw_filename = f"scene_{scene_id}.jpg"
        raw_path = self.raw_images_dir / raw_filename
        success = cv2.imwrite(str(raw_path), image)
        if not success:
            raise RuntimeError(f"Failed to save image to {raw_path}")
        logger.info(f"  📸 Raw image: {raw_path.name} ({raw_path.stat().st_size} bytes)")
        
        # Save visualization (with contours)
        vis_image = self._create_visualization(image, detections)
        vis_filename = f"scene_{scene_id}_viz.jpg"
        vis_path = self.raw_images_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"  📊 Visualization: {vis_path.name}")
        
        # NOTE: Crops are generated AFTER Claude analysis (only for useful objects)
        # This eliminates mapping issues. See storage_v2.save_crops_for_useful_objects()
        # DO NOT call _save_crops() here - it would generate crops with corrupt original_index
        crop_paths = []
        logger.info(f"  ✂️  Crops: Will be generated after Claude analysis (only for useful objects)")
        
        # Create scene metadata
        scene_data = {
            'scene_id': scene_id,
            'timestamp': timestamp,  # Human-readable: YYYY-MM-DD_HH-MM-SS
            'datetime': timestamp_iso,  # ISO 8601 full format
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'camera_id': camera_id,
            'user_id': user_id,
            'system_id': system_id,
            'raw_image': str(raw_path),
            'visualization': str(vis_path),
            'num_objects': len(detections),
            'crops': crop_paths,
            'detections': [
                {
                    'id': det.get('id', i),  # Use index as fallback if 'id' is missing
                    'bbox': det.get('bbox', [0, 0, 0, 0]),
                    'confidence': det.get('confidence', 0.0),
                    'area': det.get('area', 0),
                    'crop_path': crop_paths[i] if i < len(crop_paths) else None
                }
                for i, det in enumerate(detections)
            ],
            'metadata': metadata or {}
        }
        
        # Save scene metadata JSON
        metadata_path = self.raw_images_dir / f"scene_{scene_id}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(scene_data, f, indent=2)
        logger.info(f"  📄 Metadata: {metadata_path.name}")
        
        return scene_data
    
    def _create_visualization(
        self,
        image: cv2.Mat,
        detections: List[Dict[str, Any]]
    ) -> cv2.Mat:
        """Create visualization with contours (like Meta SAM 3)."""
        import numpy as np
        
        vis = image.copy()
        colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
            (255, 0, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255),
            (0, 255, 128), (255, 0, 128)
        ]
        
        overlay = np.zeros_like(vis, dtype=np.float32)
        
        for i, detection in enumerate(detections[:20]):
            mask = detection.get('mask')
            color = colors[i % len(colors)]
            
            if mask is not None:
                if mask.dtype != np.uint8:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                else:
                    mask_uint8 = mask
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 4)
                
                lighter_color = tuple(min(255, int(c * 1.4)) for c in color)
                cv2.drawContours(vis, contours, -1, lighter_color, 2)
                
                overlay[mask_uint8 > 0] = np.array(color, dtype=np.float32) * 0.12
        
        # Apply screen blending
        base = vis.astype(np.float32)
        screen_result = 255 - ((255 - base) * (255 - overlay) / 255)
        vis = np.clip(screen_result, 0, 255).astype(np.uint8)
        
        return vis
    
    def _save_crops(
        self,
        image: cv2.Mat,
        detections: List[Dict[str, Any]],
        scene_id: str
    ) -> List[str]:
        """
        DEPRECATED: Crops are now generated AFTER Claude analysis (only for useful objects).
        See storage_v2.save_crops_for_useful_objects() instead.
        
        This method is kept for backward compatibility but returns empty list.
        
        Args:
            image: Full scene image
            detections: List of detections
            scene_id: Scene identifier
            
        Returns:
            Empty list (crops generated later)
        """
        # DO NOT generate crops here - they will be generated after Claude analysis
        # This prevents mapping issues with corrupt original_index
        logger.debug(f"  ⚠️  _save_crops() called but crops will be generated after Claude analysis")
        return []
    
    def _get_border_color(self, crop: np.ndarray) -> tuple:
        """
        Get average border color from crop edges for natural padding.
        
        Args:
            crop: Image crop
            
        Returns:
            BGR color tuple (B, G, R)
        """
        h, w = crop.shape[:2]
        if h < 3 or w < 3:
            return (255, 255, 255)  # White fallback
        
        # Sample border pixels (top, bottom, left, right edges)
        border_pixels = np.concatenate([
            crop[0, :].reshape(-1, 3),  # Top
            crop[-1, :].reshape(-1, 3),  # Bottom
            crop[:, 0].reshape(-1, 3),  # Left
            crop[:, -1].reshape(-1, 3)  # Right
        ], axis=0)
        
        # Calculate average BGR color
        avg_color = np.mean(border_pixels, axis=0).astype(np.uint8)
        return tuple(avg_color.tolist())
    
    def cleanup_old_scenes(
        self,
        keep_days: int = 7,
        keep_useful_only: bool = True
    ) -> Dict[str, int]:
        """
        Clean up old scene images and crops.
        
        Args:
            keep_days: Keep scenes from last N days (default: 7)
            keep_useful_only: If True, only keep crops from useful objects
            
        Returns:
            Dict with cleanup statistics
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cutoff_timestamp = cutoff_date.strftime('%Y-%m-%d')
        
        logger.info(f"🧹 Cleaning up scenes older than {keep_days} days (before {cutoff_timestamp})...")
        
        stats = {
            'scenes_deleted': 0,
            'crops_deleted': 0,
            'images_deleted': 0,
            'bytes_freed': 0
        }
        
        # Load database to check which crops are useful
        useful_crops = set()
        if keep_useful_only:
            db_path = Path(self.config['paths']['database'])
            if db_path.exists():
                try:
                    with open(db_path) as f:
                        db = json.load(f)
                    for obj in db:
                        if 'thumbnail' in obj:
                            useful_crops.add(Path(obj['thumbnail']).name)
                except Exception as e:
                    logger.warning(f"Could not load database for cleanup: {e}")
        
        # Clean raw images (scenes)
        for img_path in self.raw_images_dir.glob("scene_*.jpg"):
            try:
                # Extract date from filename: scene_YYYY-MM-DD_HH-MM-SS.jpg
                parts = img_path.stem.split('_')
                if len(parts) >= 3:
                    date_str = parts[-2]  # YYYY-MM-DD
                    if date_str < cutoff_timestamp:
                        size = img_path.stat().st_size
                        img_path.unlink()
                        stats['images_deleted'] += 1
                        stats['bytes_freed'] += size
                        
                        # Also delete corresponding viz and metadata
                        viz_path = img_path.parent / img_path.name.replace('.jpg', '_viz.jpg')
                        if viz_path.exists():
                            size = viz_path.stat().st_size
                            viz_path.unlink()
                            stats['bytes_freed'] += size
                        
                        meta_path = img_path.parent / img_path.name.replace('.jpg', '_meta.json')
                        if meta_path.exists():
                            meta_path.unlink()
                        
                        stats['scenes_deleted'] += 1
            except Exception as e:
                logger.debug(f"Error processing {img_path}: {e}")
        
        # Clean crop directories
        for crop_dir in self.crops_dir.iterdir():
            if not crop_dir.is_dir():
                continue
            
            try:
                # Extract date from dir name: YYYY-MM-DD_HH-MM-SS or cam0_YYYY-MM-DD_HH-MM-SS
                dir_parts = crop_dir.name.split('_')
                date_str = None
                for part in dir_parts:
                    if len(part) == 10 and part.count('-') == 2:  # YYYY-MM-DD
                        date_str = part
                        break
                
                if date_str and date_str < cutoff_timestamp:
                    # Delete entire crop directory
                    dir_size = sum(f.stat().st_size for f in crop_dir.rglob('*') if f.is_file())
                    
                    # If keep_useful_only, check if any crops are useful
                    if keep_useful_only:
                        has_useful = any(f.name in useful_crops for f in crop_dir.glob('*.jpg'))
                        if not has_useful:
                            import shutil
                            shutil.rmtree(crop_dir)
                            stats['crops_deleted'] += len(list(crop_dir.glob('*.jpg')))
                            stats['bytes_freed'] += dir_size
                    else:
                        import shutil
                        shutil.rmtree(crop_dir)
                        stats['crops_deleted'] += len(list(crop_dir.glob('*.jpg')))
                        stats['bytes_freed'] += dir_size
            except Exception as e:
                logger.debug(f"Error processing {crop_dir}: {e}")
        
        logger.info(f"✅ Cleanup complete:")
        logger.info(f"   Scenes deleted: {stats['scenes_deleted']}")
        logger.info(f"   Crops deleted: {stats['crops_deleted']}")
        logger.info(f"   Images deleted: {stats['images_deleted']}")
        logger.info(f"   Space freed: {stats['bytes_freed'] / 1024 / 1024:.2f} MB")
        
        return stats
    
    def delete_scene(self, scene_data: Dict[str, Any]) -> bool:
        """
        Delete a specific scene and all its associated files.
        
        Args:
            scene_data: Dict returned from save_scene() with paths to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            deleted = []
            
            # Delete raw image
            if 'raw_image' in scene_data:
                raw_path = Path(scene_data['raw_image'])
                if raw_path.exists():
                    raw_path.unlink()
                    deleted.append(f"raw: {raw_path.name}")
            
            # Delete visualization
            if 'visualization' in scene_data:
                vis_path = Path(scene_data['visualization'])
                if vis_path.exists():
                    vis_path.unlink()
                    deleted.append(f"viz: {vis_path.name}")
            
            # Delete metadata
            if 'raw_image' in scene_data:
                raw_path = Path(scene_data['raw_image'])
                meta_path = raw_path.parent / raw_path.name.replace('.jpg', '_meta.json')
                if meta_path.exists():
                    meta_path.unlink()
                    deleted.append(f"meta: {meta_path.name}")
            
            # Delete all crops
            if 'crops' in scene_data and scene_data['crops']:
                crop_dir = Path(scene_data['crops'][0]).parent
                if crop_dir.exists() and crop_dir.is_dir():
                    import shutil
                    shutil.rmtree(crop_dir)
                    deleted.append(f"crops: {crop_dir.name}")
            
            if deleted:
                logger.debug(f"🧹 Deleted scene files: {', '.join(deleted)}")
            
            # Ensure directories still exist (don't delete the parent directories)
            self.raw_images_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete scene: {e}")
            return False
    
    def create_display_frame(
        self,
        image: cv2.Mat,
        detections: List[Dict[str, Any]],
        object_analyses: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> cv2.Mat:
        """
        Create display frame with auras/masks and labels (for live view).
        Shows masks as semi-transparent overlays with colored auras for better visibility.
        
        Args:
            image: Original image
            detections: List of detections
            object_analyses: Optional dict of analyses (obj_id -> analysis)
            
        Returns:
            Display frame with masks, auras, and labels
        """
        import numpy as np
        
        display_frame = image.copy()
        colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
            (255, 0, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255),
            (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
            (255, 128, 128), (128, 255, 128), (128, 128, 255)
        ]
        
        # Create overlay for mask coloring (more visible)
        overlay = np.zeros_like(display_frame, dtype=np.float32)
        
        # Draw masks with auras for better visibility
        for i, detection in enumerate(detections[:50]):  # Show up to 50 objects
            x, y, w, h = detection['bbox']
            conf = detection['confidence']
            mask = detection.get('mask')
            color = colors[i % len(colors)]
            color_bgr = tuple(int(c) for c in color)
            
            if mask is not None:
                # Convert mask to uint8 if needed
                if mask.dtype != np.uint8:
                    if mask.max() <= 1.0:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    else:
                        mask_uint8 = mask.astype(np.uint8)
                else:
                    mask_uint8 = mask
                
                # Ensure mask is binary
                if mask_uint8.max() > 1:
                    _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
                
                # Resize mask to image size if needed
                if mask_uint8.shape[:2] != image.shape[:2]:
                    mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                
                # Create dilated mask for aura effect (glow around object)
                kernel_aura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask_dilated = cv2.dilate(mask_uint8, kernel_aura, iterations=2)
                
                # Draw aura (dilated mask, lighter color, more transparent)
                aura_mask = (mask_dilated > 0) & (mask_uint8 == 0)  # Only the aura, not the object
                overlay[aura_mask] = np.array(color, dtype=np.float32) * 0.15  # Light aura
                
                # Draw mask overlay (object itself, more visible)
                mask_bool = mask_uint8 > 0
                overlay[mask_bool] = np.array(color, dtype=np.float32) * 0.25  # More visible overlay
                
                # Draw contours for clear edges
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Outer contour (white, thick) for visibility
                    cv2.drawContours(display_frame, contours, -1, (255, 255, 255), 3)
                    # Inner contour (colored, medium)
                    cv2.drawContours(display_frame, contours, -1, color_bgr, 2)
            else:
                # Fallback: draw rectangle if no mask available
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color_bgr, 4)
                # Add semi-transparent fill
                overlay[y:y+h, x:x+w] = np.array(color, dtype=np.float32) * 0.15
        
        # Apply overlay with alpha blending for better visibility
        base = display_frame.astype(np.float32)
        # Blend: result = base * (1 - alpha) + overlay * alpha
        alpha = 0.35  # More visible overlay
        display_frame = (base * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        # Draw labels with better visibility
        if object_analyses:
            for i, detection in enumerate(detections[:50]):  # Show labels for up to 50 objects
                x, y, w, h = detection['bbox']
                color_bgr = colors[i % len(colors)]
                color_tuple = tuple(int(c) for c in color_bgr)
                obj_id = detection['id']
                
                if obj_id in object_analyses:
                    obj_name = object_analyses[obj_id].get('name', f'#{i+1}')
                    category = object_analyses[obj_id].get('category', '')
                    if category:
                        label = f"#{i+1}: {obj_name[:20]} [{category[:10]}]"
                    else:
                    label = f"#{i+1}: {obj_name[:25]}"
                else:
                    conf = detection.get('confidence', 0)
                    label = f"#{i+1} (conf:{conf:.2f})"
                
                # Calculate text size and position
                font_scale = 0.7 if image.shape[0] > 2000 else 0.5  # Scale font for 4K
                thickness = 2 if image.shape[0] > 2000 else 1
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y = max(y - 5, text_h + 10)
                label_x = max(x, 0)
                label_w = text_w + 10
                label_h = text_h + baseline + 10
                
                # Ensure label fits within image bounds
                if label_x + label_w > image.shape[1]:
                    label_x = image.shape[1] - label_w - 5
                if label_y - label_h < 0:
                    label_y = label_h + 5
                
                # Draw semi-transparent background for label (black rectangle)
                label_bg_y1 = label_y - label_h
                label_bg_y2 = label_y
                label_bg_x1 = label_x
                label_bg_x2 = label_x + label_w
                
                if (label_bg_y1 >= 0 and label_bg_y2 <= image.shape[0] and 
                    label_bg_x1 >= 0 and label_bg_x2 <= image.shape[1]):
                    roi = display_frame[label_bg_y1:label_bg_y2, label_bg_x1:label_bg_x2]
                    if roi.size > 0:
                        # Blend with black background (semi-transparent)
                        alpha_bg = 0.7
                        label_bg = np.zeros_like(roi, dtype=np.float32)
                        label_bg[:] = (0, 0, 0)  # Black background
                        blended = (roi.astype(np.float32) * (1 - alpha_bg) + label_bg * alpha_bg).astype(np.uint8)
                        display_frame[label_bg_y1:label_bg_y2, label_bg_x1:label_bg_x2] = blended
                
                # Draw label text (white for contrast, then colored border)
                text_x = label_x + 5
                text_y = label_y - baseline - 5
                if text_x >= 0 and text_y >= 0 and text_x + text_w <= image.shape[1]:
                    # Draw colored border first (thicker, for visibility)
                    cv2.putText(display_frame, label, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_tuple, thickness + 2)
                    # Draw white text on top (better contrast)
                    cv2.putText(display_frame, label, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return display_frame

ENDFILE
echo '✅ storage.py creado'

# ==========================================
# Creando api.py
# ==========================================
cat > ~/1UP_2/server/api.py << 'ENDFILE'
"""
1UP Server API - FastAPI endpoint for GPU processing
Receives 4K images, processes with SAM3 + Claude, returns results.
Max 350 lines.
"""
import base64
import logging
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from detector import SAM3Detector
from analyzer import ClaudeAnalyzer
from storage_v2 import save_crops_for_useful_objects
from filters import filter_objects
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="1UP Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector and analyzer
detector = None
analyzer = None
config = None


class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    image_base64: str  # Base64 encoded image
    timestamp: str  # Scene timestamp
    config_override: Dict[str, Any] = {}  # Optional config overrides


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    success: bool
    detections: List[Dict[str, Any]]
    crops: Dict[str, str]  # n -> crop_path
    metadata: Dict[str, Any]
    error: str = None


def load_config():
    """Load server configuration."""
    config_path = Path(__file__).parent / "config_server.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config for RunPod GPU
        return {
            'sam3': {
                'device': 'cuda',
                'filtering': {'enabled': False},
                'enhance_image': True
            },
            'claude': {
                'api_key_env': 'CLAUDE_API_KEY',
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 16000
            },
            'storage': {
                'crops_dir': 'images/crops',
                'raw_dir': 'images/raw'
            }
        }


def initialize_models():
    """Initialize SAM3 detector and Claude analyzer."""
    global detector, analyzer, config
    
    config = load_config()
    logger.info("Loading configuration...")
    
    # Initialize SAM3 detector (GPU/CUDA)
    try:
        device = config['sam3'].get('device', 'cuda')
        logger.info(f"Initializing SAM3 detector on {device}...")
        detector = SAM3Detector(device=device)
        logger.info("✅ SAM3 detector initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SAM3: {e}")
        raise
    
    # Initialize Claude analyzer
    api_key = os.environ.get(config['claude']['api_key_env'])
    if not api_key:
        logger.warning("⚠️  Claude API key not found, analysis disabled")
        analyzer = None
    else:
        try:
            analyzer = ClaudeAnalyzer(
                api_key=api_key,
                model=config['claude'].get('model', 'claude-sonnet-4-20250514'),
                max_tokens=config['claude'].get('max_tokens', 16000)
            )
            logger.info("✅ Claude analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("🚀 Starting 1UP Detection API...")
    initialize_models()
    logger.info("✅ API ready")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "analyzer_ready": analyzer is not None
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Process image with SAM3 + Claude, return detections and crops.
    """
    global detector, analyzer, config
    
    if detector is None:
        raise HTTPException(status_code=500, detail="SAM3 detector not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        logger.info(f"📸 Processing image: {image.shape[1]}x{image.shape[0]}")
        
        # Run SAM3 detection
        filter_config = config['sam3'].get('filtering', {})
        enhance_image = config['sam3'].get('enhance_image', False)
        
        detections = detector.detect_objects(
            image,
            apply_filtering=filter_config.get('enabled', False),
            enhance_image=enhance_image,
            text_prompt=None
        )
        
        logger.info(f"🔍 SAM3 detected {len(detections)} objects")
        
        # Claude analysis (if available)
        analyses = []
        useful_objects = []
        
        if analyzer and detections:
            logger.info(f"🤖 Analyzing with Claude...")
            
            # Save temporary scene image for Claude
            temp_dir = Path(config['storage']['raw_dir'])
            temp_dir.mkdir(parents=True, exist_ok=True)
            scene_path = temp_dir / f"temp_scene_{request.timestamp}.jpg"
            cv2.imwrite(str(scene_path), image)
            
            # Analyze with Claude
            analyses = analyzer.analyze_scene_with_bboxes(
                str(scene_path),
                detections,
                language="spanish"
            )
            
            # Filter useful objects
            useful_analyses = [a for a in analyses if a.get('useful') == 'yes']
            logger.info(f"✅ Claude found {len(useful_analyses)} useful objects")
            
            # Apply post-filters
            useful_analyses = filter_objects(useful_analyses, image.shape)
            
            # Build useful_objects structure (needed for save_crops_for_useful_objects)
            # Structure: [{'detection': {...}, 'analysis': {...}, 'n': ...}]
            useful_objects = []
            for i, analysis in enumerate(useful_analyses):
                n = analysis.get('n', i + 1)
                # Find corresponding detection by n (1-indexed)
                det_idx = n - 1
                if 0 <= det_idx < len(detections):
                    detection = detections[det_idx]
                else:
                    # Fallback: use first detection if n is out of range
                    logger.warning(f"⚠️  n={n} out of range, using first detection")
                    detection = detections[0] if detections else {}
                
                useful_objects.append({
                    'detection': detection,
                    'analysis': analysis,
                    'n': n,
                    'filtered_index': i
                })
            
            # Renumber analyses consecutively (1, 2, 3, 4...)
            analyses_for_crops = []
            for new_n, obj in enumerate(useful_objects, start=1):
                analysis = obj['analysis'].copy()
                analysis['n'] = new_n  # Renumber consecutively
                
                # Ensure bbox is present
                if not analysis.get('bbox') or len(analysis.get('bbox', [])) != 4:
                    # Fallback to SAM detection bbox
                    detection = obj.get('detection', {})
                    sam_bbox = detection.get('bbox')
                    if sam_bbox and len(sam_bbox) == 4:
                        analysis['bbox'] = sam_bbox
                    else:
                        logger.warning(f"⚠️  No bbox available for n={new_n}, skipping")
                        continue
                
                analyses_for_crops.append(analysis)
                obj['new_n'] = new_n
            
            # Generate crops for useful objects
            crops_dir = config['storage']['crops_dir']
            n_to_crop = save_crops_for_useful_objects(
                image,
                analyses_for_crops,  # Renumbered analyses
                useful_objects,  # Full structure with detection and analysis
                crops_dir,
                request.timestamp
            )
            
            # Convert crop paths to relative strings
            crops = {str(n): str(path) for n, path in n_to_crop.items()}
        else:
            crops = {}
            logger.warning("⚠️  Claude analyzer not available, skipping analysis")
        
        # Prepare response (extract analyses from useful_objects if available)
        if analyzer and useful_objects:
            response_detections = [obj['analysis'] for obj in useful_objects]
        else:
            response_detections = detections
        
        # Convert numpy arrays to Python lists for JSON serialization
        def convert_numpy(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        response_detections = convert_numpy(response_detections)
        
        # Prepare response
        response = DetectionResponse(
            success=True,
            detections=response_detections,
            crops=crops,
            metadata={
                'timestamp': request.timestamp,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'total_detections': len(detections),
                'useful_objects': len(useful_objects) if analyzer else len(detections)
            }
        )
        
        logger.info(f"✅ Detection complete: {len(response.detections)} objects, {len(crops)} crops")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}", exc_info=True)
        return DetectionResponse(
            success=False,
            detections=[],
            crops={},
            metadata={},
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
ENDFILE
echo '✅ api.py creado'

# ==========================================
# Creando capture_client.py
# ==========================================
cat > ~/1UP_2/client/capture_client.py << 'ENDFILE'
"""
1UP Client - Captures frames from Reolink and sends to server
Client-side capture only, processing happens on server.
Max 350 lines.
"""
import sys
import cv2
import yaml
import logging
import base64
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_utils import open_camera
from image_quality import calculate_sharpness_score
import platform
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaptureClient:
    """Client for capturing frames and sending to server."""
    
    def __init__(self, config_path: str = "client/config_client.yaml"):
        """Initialize client with configuration."""
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.server_url = self.config['server']['url']
        self.camera_source = self.config['camera']['source']
        self.camera_resolution = self.config['camera']['resolution']
        self.cap = None
        
        logger.info(f"📡 Client initialized: server={self.server_url}")
    
    def open_camera(self):
        """Open camera connection."""
        logger.info("📷 Opening camera...")
        
        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else None
        self.cap = open_camera(
            source=self.camera_source,
            resolution=self.camera_resolution,
            buffer_size=self.config['camera'].get('buffer_size', 1),
            backend=backend
        )
        
        if self.cap is None:
            raise RuntimeError("Failed to open camera")
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"✅ Camera opened: {actual_w}x{actual_h}")
        
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera."""
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning("⚠️  Failed to capture frame")
            return None
        
        return frame
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """Validate frame quality."""
        quality_config = self.config['camera'].get('quality_check', {})
        if not quality_config.get('enabled', True):
            return True
        
        sharpness = calculate_sharpness_score(frame)
        min_sharpness = quality_config.get('min_sharpness', 20.0)
        
        if sharpness < min_sharpness:
            logger.warning(f"⚠️  Frame rejected (sharpness={sharpness:.1f} < {min_sharpness})")
            return False
        
        return True
    
    def encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def send_to_server(self, image: np.ndarray, timestamp: str) -> Dict[str, Any]:
        """
        Send image to server for processing.
        
        Returns:
            Server response with detections and crops
        """
        logger.info(f"📤 Sending image to server: {self.server_url}/detect")
        
        image_base64 = self.encode_image(image)
        
        payload = {
            "image_base64": image_base64,
            "timestamp": timestamp
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/detect",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info(f"✅ Server response: {len(result.get('detections', []))} objects, {len(result.get('crops', {}))} crops")
                return result
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ Server error: {error}")
                raise RuntimeError(f"Server processing failed: {error}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            raise
    
    def check_server_health(self) -> bool:
        """Check if server is available."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            logger.info(f"✅ Server health: {health}")
            return health.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"❌ Server health check failed: {e}")
            return False
    
    def close(self):
        """Close camera connection."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("📷 Camera closed")


def main():
    """Main client loop."""
    client = CaptureClient()
    
    # Check server health
    if not client.check_server_health():
        logger.error("❌ Server is not available. Please start the server first.")
        return
    
    # Open camera
    try:
        client.open_camera()
    except Exception as e:
        logger.error(f"❌ Failed to open camera: {e}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("1UP CLIENT - Server Processing Mode")
    logger.info("  SPACE = Capture and send to server")
    logger.info("  Q = Quit")
    logger.info("="*60 + "\n")
    
    try:
        while True:
            # Read frame for preview
            ret, frame = client.cap.read()
            if not ret:
                continue
            
            # Show preview (resize for display)
            preview = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow('1UP Client - SPACE=Capture, Q=Quit', preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Capture and process
                logger.info("\n📸 Capturing frame...")
                
                # Capture fresh frame
                capture_frame = client.capture_frame()
                if capture_frame is None:
                    continue
                
                # Validate quality
                if not client.validate_frame(capture_frame):
                    logger.warning("⚠️  Frame rejected, try again")
                    continue
                
                # Generate timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Send to server
                try:
                    result = client.send_to_server(capture_frame, timestamp)
                    
                    # Log results
                    detections = result.get('detections', [])
                    crops = result.get('crops', {})
                    
                    logger.info(f"\n✅ Processing complete:")
                    logger.info(f"   Objects: {len(detections)}")
                    logger.info(f"   Crops: {len(crops)}")
                    
                    # Display first few objects
                    for i, obj in enumerate(detections[:5]):
                        name = obj.get('name', 'Unknown')
                        category = obj.get('category', 'N/A')
                        logger.info(f"   {i+1}. {name} ({category})")
                    
                except Exception as e:
                    logger.error(f"❌ Processing failed: {e}")
            
            elif key == ord('q') or key == ord('Q'):
                break
    
    finally:
        client.close()
        cv2.destroyAllWindows()
        logger.info("👋 Client closed")


if __name__ == "__main__":
    main()
ENDFILE
echo '✅ capture_client.py creado'
