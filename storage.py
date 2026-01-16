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

