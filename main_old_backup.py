"""
1UP Main Pipeline - Local Development
Captures, detects, analyzes, and stores objects.
Max 350 lines.
"""
import cv2
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import os

from detector import SAM3Detector
from analyzer import ClaudeAnalyzer
from hybrid_detector import HybridDetector
from storage_v2 import save_crops_for_useful_objects
from filters import filter_useful_objects

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """1UP detection and analysis pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        
        # Load config
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Create directories
        self._create_directories()
        
        # Initialize detector
        logger.info("Initializing SAM 3 detector...")
        self.detector = SAM3Detector(
            device=self.config['sam3']['device']
        )
        
        # Initialize analyzer
        logger.info("Initializing Claude analyzer...")
        api_key = os.environ.get(self.config['claude']['api_key_env'])
        if not api_key:
            raise ValueError(
                f"Environment variable {self.config['claude']['api_key_env']} not set"
            )
        
        self.analyzer = ClaudeAnalyzer(
            api_key=api_key,
            model=self.config['claude']['model'],
            max_tokens=self.config['claude']['max_tokens'],
            temperature=self.config['claude'].get('temperature', 0)
        )
        
        # Initialize hybrid detector (SAM + Claude validation)
        self.hybrid_detector = HybridDetector(self.detector, self.analyzer)
        logger.info("✅ Hybrid detector initialized (SAM + Claude validation)")
        
        # Load database
        self.database = self._load_database()
        
        logger.info("✅ Pipeline initialized")
    
    def _create_directories(self):
        """Create necessary directories."""
        for path_key in ['raw_images', 'crops']:
            path = Path(self.config['paths'][path_key])
            path.mkdir(parents=True, exist_ok=True)
        
        # Create database directory
        db_path = Path(self.config['paths']['database'])
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_database(self) -> List[Dict[str, Any]]:
        """Load object database from JSON."""
        db_path = Path(self.config['paths']['database'])
        
        if db_path.exists():
            with open(db_path) as f:
                return json.load(f)
        else:
            return []
    
    def _save_database(self):
        """Save object database to JSON."""
        db_path = Path(self.config['paths']['database'])
        
        with open(db_path, 'w') as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Database saved ({len(self.database)} objects)")
    
    def process_image(self, image: cv2.Mat) -> Dict[str, Any]:
        """
        Process a single image through full pipeline.
        
        Args:
            image: OpenCV image
            
        Returns:
            Processing results with objects detected
        """
        # ISO 8601 format: YYYY-MM-DD_HH-MM-SS (sortable, readable, filesystem-safe)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scene {timestamp}")
        logger.info(f"{'='*60}\n")
        
        # Save raw image
        raw_path = Path(self.config['paths']['raw_images']) / f"scene_{timestamp}.jpg"
        cv2.imwrite(str(raw_path), image)
        logger.info(f"📸 Saved raw image: {raw_path}")
        
        # Detect objects with smart filtering if enabled
        filter_config = self.config.get('sam3', {}).get('filtering', {})
        apply_filtering = filter_config.get('enabled', False)
        
        # Get enhancement setting and text prompt
        enhance_image = self.config.get('sam3', {}).get('enhance_image', False)
        text_prompt = self.config.get('sam3', {}).get('text_prompt', '') or None
        
        if apply_filtering:
            logger.info("🎯 Smart filtering enabled (PRE-Claude)")
            initial_detections = self.detector.detect_objects(
                image,
                apply_filtering=True,
                enhance_image=enhance_image,
                text_prompt=text_prompt,
                min_area=filter_config.get('min_area'),
                max_area_ratio=filter_config.get('max_area_ratio'),
                min_aspect_ratio=filter_config.get('min_aspect_ratio'),
                max_aspect_ratio=filter_config.get('max_aspect_ratio'),
                nms_iou_threshold=filter_config.get('nms_iou_threshold')
            )
        else:
            initial_detections = self.detector.detect_objects(
                image, 
                enhance_image=enhance_image,
                text_prompt=text_prompt
            )
        
        logger.info(f"🔍 SAM initial detection: {len(initial_detections)} objects")
        
        # Use hybrid detector to validate and find missing objects
        logger.info(f"🔄 Running hybrid detection (SAM + Claude validation)...")
        logger.info(f"   ⏳ This may take 30-60 seconds...\n")
        
        try:
            final_detections, _ = self.hybrid_detector.detect_with_validation(
                image=image,
                scene_path=str(raw_path)
            )
            
            if final_detections:
                detections = final_detections
                logger.info(f"✅ Hybrid detection complete: {len(detections)} objects")
                logger.info(f"   (SAM: {len(initial_detections)}, "
                          f"Claude found: {sum(1 for d in detections if d.get('source', '').startswith('claude'))})")
            else:
                detections = initial_detections
                logger.warning("⚠️  Hybrid detection returned no objects, using SAM detections")
                
        except Exception as e:
            logger.error(f"❌ Hybrid detection failed: {e}")
            logger.warning("   Falling back to SAM detections only")
            detections = initial_detections
        
        logger.info(f"🔍 Final detected: {len(detections)} objects")
        
        if len(detections) == 0:
            logger.warning("No objects detected")
            return {
                'timestamp': timestamp,
                'raw_image': str(raw_path),
                'objects': []
            }
        
        # Save visualization
        vis_image = self.detector.visualize_detections(image, detections)
        vis_path = Path(self.config['paths']['raw_images']) / f"scene_{timestamp}_viz.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"📊 Saved visualization: {vis_path}")
        
        # Save crops
        # Generate crops (for thumbnails in web, NOT sent to Claude)
        crop_dir = Path(self.config['paths']['crops']) / timestamp
        crop_paths = self.detector.save_crops(
            image, detections, str(crop_dir), prefix="obj"
        )
        logger.info(f"✂️  Generated {len(crop_paths)} crops (for thumbnails)")
        
        # Pre-filter: Only send large objects to Claude (reduce noise and API costs)
        # CRITICAL: Maintain mapping from filtered index to original index
        min_area_for_analysis = self.config.get('sam3', {}).get('min_area_for_analysis', 5000)
        large_detections = []
        original_indices = []  # Maps filtered_index → original_index
        
        for i, d in enumerate(detections):
            if d.get('area', 0) >= min_area_for_analysis:
                large_detections.append(d)
                # CRITICAL: Use original_index if available (preserves order before sorting)
                # Otherwise fall back to enumeration index
                original_idx = d.get('original_index', i)
                original_indices.append(original_idx)  # Store original index
        
        logger.info(f"\n🔍 SAM detected: {len(detections)} total objects")
        logger.info(f"📊 Large objects (>{min_area_for_analysis}px²): {len(large_detections)}")
        logger.info(f"   (Skipping {len(detections) - len(large_detections)} small fragments)")
        
        if not large_detections:
            logger.warning("⚠️  No large objects detected. Nothing to analyze.")
            return {
                'timestamp': timestamp,
                'raw_image': str(raw_path),
                'visualization': str(vis_path),
                'objects': []
            }
        
        # Analyze scene with Claude (1 API call: validation + analysis + missing objects)
        logger.info(f"\n🤖 Analyzing with Claude (1 API call: validation + analysis + missing objects)...")
        logger.info(f"   Scene image + {len(large_detections)} bounding boxes")
        logger.info(f"   ⏳ This may take 30-50 seconds...\n")
        
        try:
            # SINGLE API CALL: validation + analysis + missing objects
            claude_response = self.analyzer.analyze_scene_with_validation(
                scene_path=str(raw_path),
                detections=large_detections,  # Only large objects
                language="spanish"
            )
            
            # Extract validated objects (with full analysis) and missing objects
            validated_analyses = claude_response.get('validated_objects', [])
            missing_objects = claude_response.get('missing_objects', [])
            
            logger.info(f"✅ Claude response: {len(validated_analyses)} validated, {len(missing_objects)} missing found")
            
            # Use validated_analyses as the main analyses list
            analyses = validated_analyses
            
            # Process results: filter useful objects only
            useful_objects = []
            skipped_count = 0
            
            # Get image dimensions for size filtering
            img_height, img_width = image.shape[:2]
            max_area_ratio = 0.5  # Filter objects larger than 50% of image
            
            # CRITICAL: Create map from detections index to crop_path
            # crop_paths is now indexed by original_index, not enumeration order
            index_to_crop = {}
            for i, det in enumerate(detections):
                original_idx = det.get('original_index', i)
                # crop_paths is indexed by original_index, so use original_idx to get crop
                if original_idx < len(crop_paths) and crop_paths[original_idx] is not None:
                    index_to_crop[i] = crop_paths[original_idx]
                elif i < len(crop_paths) and crop_paths[i] is not None:
                    # Fallback to enumeration index if original_index mapping fails
                    index_to_crop[i] = crop_paths[i]
                    logger.warning(f"  ⚠️  Using enumeration index {i} instead of original_index {original_idx} for crop")
            
            # Process analyses using 'n' field to map back to correct detection/crop
            # CRITICAL: Claude's 'n' (1-65) corresponds to index in large_detections (0-64)
            # But crops are from original detections (0-89)
            # Use original_indices mapping to get correct crop
            for analysis in analyses:
                n = analysis.get('n', 0)
                
                # Validate n is in valid range (relative to large_detections)
                if n < 1 or n > len(large_detections):
                    logger.warning(f"Invalid object number n={n}, skipping")
                    skipped_count += 1
                    continue
                
                # Map from Claude's n (1-indexed) to filtered index (0-indexed)
                filtered_index = n - 1
                
                # CRITICAL: Get original index using mapping
                if filtered_index >= len(original_indices):
                    logger.warning(f"  ⚠️  #{n}: Filtered index {filtered_index} out of range, skipping")
                    skipped_count += 1
                    continue
                
                detection_index = original_indices[filtered_index]
                
                # Validate original index is in range
                if detection_index >= len(detections):
                    logger.warning(f"  ⚠️  #{n}: Original index {detection_index} out of range, skipping")
                    skipped_count += 1
                    continue
                
                # Use the original index to get detection and crop
                detection = detections[detection_index]
                crop_path = index_to_crop.get(detection_index)
                
                # Validate crop_path exists
                if not crop_path:
                    logger.warning(f"  ⚠️  #{n}: No crop found for detection index={detection_index}, skipping")
                    skipped_count += 1
                    continue
                
                # Filter by size (skip objects too large - likely background)
                bbox = detection['bbox']
                bbox_area = bbox[2] * bbox[3]  # width * height
                image_area = img_width * img_height
                area_ratio = bbox_area / image_area
                
                if area_ratio > max_area_ratio:
                    skipped_count += 1
                    logger.debug(f"  ⏭️  #{n}: Skipped - too large ({area_ratio:.1%} of image)")
                    continue
                
                # Filter by usefulness
                useful = analysis.get('useful', 'no').lower() == 'yes'
                
                if not useful:
                    skipped_count += 1
                    reason = analysis.get('reason', 'not useful')
                    logger.debug(f"  ⏭️  #{n}: Skipped - {reason}")
                    continue
                
                # Post-filter: reject generic names (fragments, surfaces, backgrounds)
                obj_name = analysis.get('name', '').lower()
                generic_keywords = [
                    'superficie', 'esquina', 'borde', 'fragmento', 'pedazo',
                    'mesa', 'fondo', 'pared', 'suelo', 'mobiliario',
                    'objeto rectangular', 'cosa', 'elemento', 'parte',
                    'surface', 'corner', 'edge', 'fragment', 'piece',
                    'table', 'background', 'wall', 'floor', 'furniture',
                    'rectangular object', 'thing', 'element', 'part'
                ]
                
                if any(keyword in obj_name for keyword in generic_keywords):
                    skipped_count += 1
                    logger.debug(f"  ⏭️  #{n}: Skipped - generic name: '{analysis.get('name')}'")
                    continue
                
                # Object is useful - create database entry
                # Convert crop_path to relative path if needed
                crop_path_rel = str(Path(crop_path).relative_to(Path.cwd())) if crop_path and Path(crop_path).is_absolute() else crop_path
                # Ensure it starts with 'images/' for consistency
                if crop_path_rel and not crop_path_rel.startswith('images/'):
                    crop_path_rel = crop_path_rel.replace('\\', '/')  # Windows compatibility
                
                obj = {
                    'id': f"obj_{timestamp}_{len(useful_objects)+1:03d}",
                    'timestamp': timestamp,
                    'detection_number': n,  # Keep original n from Claude
                    'thumbnail': crop_path_rel or '',
                    'bbox': detection['bbox'],
                    'confidence': detection.get('confidence', 0.0),
                    'area': detection.get('area', 0),
                    'name': analysis.get('name', 'Unknown object'),
                    'category': analysis.get('category', 'other'),
                    'condition': analysis.get('condition', 'unknown'),
                    'description': analysis.get('description', ''),
                    'estimated_value': analysis.get('estimated_value')
                }
                
                useful_objects.append(obj)
                logger.info(f"  ✅ #{n}: {obj['name']}")
                logger.info(f"     Category: {obj['category']}, Condition: {obj['condition']}")
                logger.debug(f"     Detection index: {detection_index}, Thumbnail: {crop_path_rel}")
            
            # Process missing objects from Claude (using bbox matching)
            if missing_objects:
                logger.info(f"\n🔍 Processing {len(missing_objects)} missing objects from Claude...")
                
                # Helper function to calculate IoU between two bboxes
                def calculate_iou(bbox1, bbox2):
                    """Calculate IoU between two bboxes [x, y, w, h]."""
                    x1, y1, w1, h1 = bbox1
                    x2, y2, w2, h2 = bbox2
                    
                    x_left = max(x1, x2)
                    y_top = max(y1, y2)
                    x_right = min(x1 + w1, x2 + w2)
                    y_bottom = min(y1 + h1, y2 + h2)
                    
                    if x_right < x_left or y_bottom < y_top:
                        return 0.0
                    
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    union = w1 * h1 + w2 * h2 - intersection
                    return intersection / union if union > 0 else 0.0
                
                # Match missing objects to crops by bbox IoU
                for missing in missing_objects:
                    missing_bbox = missing.get('bbox')
                    if not missing_bbox or len(missing_bbox) != 4:
                        continue
                    
                    # Find best matching detection by IoU
                    best_match_idx = None
                    best_iou = 0.3  # Minimum IoU threshold
                    
                    for i, det in enumerate(detections):
                        det_bbox = det.get('bbox')
                        if not det_bbox:
                            continue
                        
                        iou = calculate_iou(missing_bbox, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = i
                    
                    # If no good match, skip (would need to generate crop)
                    if best_match_idx is not None and best_match_idx < len(crop_paths):
                        crop_path = crop_paths[best_match_idx]
                        logger.info(f"  ✅ Matched missing '{missing.get('name')}' to existing crop (IoU: {best_iou:.2f})")
                    else:
                        # Claude detected an object that SAM didn't - this is useful info to see
                        logger.warning(f"  ⚠️  Claude detected '{missing.get('name')}' but SAM didn't (best IoU match: {best_iou:.2f})")
                        continue
                    
                    # Create object entry for missing object
                    crop_path_rel = str(Path(crop_path).relative_to(Path.cwd())) if crop_path and Path(crop_path).is_absolute() else crop_path
                    if crop_path_rel and not crop_path_rel.startswith('images/'):
                        crop_path_rel = crop_path_rel.replace('\\', '/')
                    
                    missing_obj = {
                        'id': f"obj_{timestamp}_{len(useful_objects)+1:03d}",
                        'timestamp': timestamp,
                        'detection_number': len(useful_objects) + 1,
                        'thumbnail': crop_path_rel or '',
                        'bbox': missing_bbox,
                        'confidence': 0.85,  # High confidence (Claude saw it)
                        'area': missing_bbox[2] * missing_bbox[3],
                        'name': missing.get('name', 'Unknown object'),
                        'category': missing.get('category', 'other'),
                        'condition': 'unknown',  # Claude didn't analyze condition
                        'description': f"Objeto encontrado por Claude: {missing.get('name', 'Unknown')}",
                        'estimated_value': None,
                        'source': 'claude_missing'
                    }
                    
                    useful_objects.append(missing_obj)
                    logger.info(f"  ✅ Added missing: {missing_obj['name']}")
            
            logger.info(f"\n📊 Results: {len(useful_objects)} useful, {skipped_count} skipped")
            
            # Save only useful objects to database
            if useful_objects:
                self.database.extend(useful_objects)
                self._save_database()
                logger.info(f"💾 Database saved ({len(self.database)} total objects)")
            else:
                logger.warning("⚠️  No useful objects found. Nothing saved to database.")
            
            # Testing mode: Images are kept during session for web display
            # They will be deleted at next startup if auto_cleanup is enabled
            
            objects = useful_objects
            
        except Exception as e:
            logger.error(f"❌ Scene analysis failed: {e}")
            logger.exception("Full error traceback:")
            objects = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Processing complete: {len(objects)} objects")
        logger.info(f"{'='*60}\n")
        
        return {
            'timestamp': timestamp,
            'raw_image': str(raw_path),
            'visualization': str(vis_path),
            'objects': objects
        }
    
    def run_interactive(self):
        """
        Run in interactive mode with webcam.
        Press SPACE to capture, Q to quit.
        """
        # Auto-detect camera if available, otherwise use config
        try:
            from camera_utils import find_external_camera, enumerate_cameras
            
            logger.info("Detecting available cameras...")
            cameras = enumerate_cameras()
            
            if cameras:
                preferred_index = self.config['camera'].get('index')
                camera_index = find_external_camera(preferred_index=preferred_index)
                if camera_index is None:
                    camera_index = self.config['camera']['index']
            else:
                camera_index = self.config['camera']['index']
                logger.warning(f"No cameras auto-detected, using config index {camera_index}")
        except ImportError:
            # Fallback if camera_utils not available
            camera_index = self.config['camera']['index']
            logger.info(f"Using camera index from config: {camera_index}")
        
        # Open camera
        logger.info(f"Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Set resolution
        w, h = self.config['camera']['resolution']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        logger.info(f"✅ Camera opened ({w}x{h})")
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE MODE")
        logger.info("  SPACE = Capture and process")
        logger.info("  Q = Quit")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Show live view
                cv2.imshow('1UP - Press SPACE to capture, Q to quit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE
                    result = self.process_image(frame)
                    logger.info("Ready for next capture\n")
                    
                elif key == ord('q'):  # Q
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("\n👋 Bye!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="1UP Object Detection Pipeline")
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--image',
        help='Process single image file instead of interactive mode'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(config_path=args.config)
    
    if args.image:
        # Process single image
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Failed to load image: {args.image}")
            return
        
        result = pipeline.process_image(image)
        logger.info(f"\nResults: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        # Interactive mode
        pipeline.run_interactive()


if __name__ == "__main__":
    main()

