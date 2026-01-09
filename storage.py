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
        Create display frame with contours and labels (for live view).
        
        Args:
            image: Original image
            detections: List of detections
            object_analyses: Optional dict of analyses (obj_id -> analysis)
            
        Returns:
            Display frame with contours and labels
        """
        import numpy as np
        
        display_frame = image.copy()
        colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
            (255, 0, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255),
            (0, 255, 128), (255, 0, 128)
        ]
        
        overlay = np.zeros_like(display_frame, dtype=np.float32)
        
        # Draw contours
        for i, detection in enumerate(detections[:20]):
            x, y, w, h = detection['bbox']
            conf = detection['confidence']
            mask = detection.get('mask')
            color = colors[i % len(colors)]
            
            if mask is not None:
                if mask.dtype != np.uint8:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                else:
                    mask_uint8 = mask
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_frame, contours, -1, color, 4)
                
                lighter_color = tuple(min(255, int(c * 1.4)) for c in color)
                cv2.drawContours(display_frame, contours, -1, lighter_color, 2)
                
                overlay[mask_uint8 > 0] = np.array(color, dtype=np.float32) * 0.12
            else:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 4)
        
        # Apply screen blending
        base = display_frame.astype(np.float32)
        screen_result = 255 - ((255 - base) * (255 - overlay) / 255)
        display_frame = np.clip(screen_result, 0, 255).astype(np.uint8)
        
        # Draw labels
        if object_analyses:
            for i, detection in enumerate(detections[:20]):
                x, y, w, h = detection['bbox']
                color = colors[i % len(colors)]
                obj_id = detection['id']
                
                if obj_id in object_analyses:
                    obj_name = object_analyses[obj_id].get('name', f'#{i+1}')
                    label = f"#{i+1}: {obj_name[:25]}"
                else:
                    label = f"#{i+1} ({detection['confidence']:.2f})"
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y - 5, text_h + 10)
                cv2.rectangle(display_frame, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return display_frame

