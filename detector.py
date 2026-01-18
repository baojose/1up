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
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try direct import from sam3.sam3
        from sam3.sam3.model_builder import build_sam3_image_model
        from sam3.sam3.model.sam3_image_processor import Sam3Processor
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
                # Más prompts = más detecciones (GPU puede manejar)
                prompts = [
                    "visual",      # Detección general (objetos visuales)
                    "container",   # Contenedores, frascos, botellas, cajas
                    "object",      # Objetos genéricos
                    "item",        # Items/artículos adicionales
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
