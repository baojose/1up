"""
1UP Main Pipeline - OPTIMIZED VERSION
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
    """1UP detection and analysis pipeline - OPTIMIZED."""
    
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
        Process a single image through full pipeline - OPTIMIZED.
        
        OPTIMIZATIONS:
        - SAM executes only once
        - Crops generated AFTER Claude (only for useful objects)
        - Simplified index mapping using n directly
        
        Args:
            image: OpenCV image
            
        Returns:
            Processing results with objects detected
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scene {timestamp}")
        logger.info(f"{'='*60}\n")
        
        # Save raw image
        raw_path = Path(self.config['paths']['raw_images']) / f"scene_{timestamp}.jpg"
        cv2.imwrite(str(raw_path), image)
        logger.info(f"📸 Saved raw image: {raw_path}")
        
        # STEP 1: SAM detection (ONLY ONCE)
        filter_config = self.config.get('sam3', {}).get('filtering', {})
        apply_filtering = filter_config.get('enabled', False)
        enhance_image = self.config.get('sam3', {}).get('enhance_image', False)
        text_prompt = self.config.get('sam3', {}).get('text_prompt', '') or None
        
        logger.info("🔍 STEP 1: Running SAM 3 detection...")
        if apply_filtering:
            sam_detections = self.detector.detect_objects(
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
            sam_detections = self.detector.detect_objects(
                image, 
                enhance_image=enhance_image,
                text_prompt=text_prompt
            )
        
        logger.info(f"✅ SAM detected: {len(sam_detections)} objects")
        
        if len(sam_detections) == 0:
            logger.warning("No objects detected")
            return {
                'timestamp': timestamp,
                'raw_image': str(raw_path),
                'objects': []
            }
        
        # Save visualization
        vis_image = self.detector.visualize_detections(image, sam_detections)
        vis_path = Path(self.config['paths']['raw_images']) / f"scene_{timestamp}_viz.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"📊 Saved visualization: {vis_path}")
        
        # STEP 2: Pre-filter for Claude (only large objects)
        min_area_for_analysis = self.config.get('sam3', {}).get('min_area_for_analysis', 100)
        large_detections = [d for d in sam_detections if d.get('area', 0) >= min_area_for_analysis]
        
        logger.info(f"📊 Large objects (>{min_area_for_analysis}px²): {len(large_detections)}")
        logger.info(f"   (Skipping {len(sam_detections) - len(large_detections)} small fragments)")
        
        if not large_detections:
            logger.warning("⚠️  No large objects detected. Nothing to analyze.")
            return {
                'timestamp': timestamp,
                'raw_image': str(raw_path),
                'visualization': str(vis_path),
                'objects': []
            }
        
        # STEP 3: Claude validation (OPTIMIZED: uses existing detections, doesn't re-run SAM)
        logger.info(f"\n🤖 STEP 2: Analyzing with Claude (validation + missing objects)...")
        logger.info(f"   Scene image + {len(large_detections)} bounding boxes")
        logger.info(f"   ⏳ This may take 30-50 seconds...\n")
        
        try:
            # OPTIMIZED: Pass existing detections, no re-execution of SAM
            claude_response = self.hybrid_detector.detect_with_validation(
                scene_path=str(raw_path),
                sam_detections=large_detections
            )
            
            validated_analyses = claude_response.get('validated_objects', [])
            missing_objects = claude_response.get('missing_objects', [])
            
            logger.info(f"✅ Claude response: {len(validated_analyses)} validated, {len(missing_objects)} missing found")
            
        except Exception as e:
            logger.error(f"❌ Claude analysis failed: {e}")
            logger.exception("Full error traceback:")
            validated_analyses = []
            missing_objects = []
        
        # STEP 4: Filter useful objects
        img_height, img_width = image.shape[:2]
        max_area_ratio = 0.5
        
        useful_objects_list = filter_useful_objects(
            analyses=validated_analyses,
            detections=large_detections,
            image_shape=(img_height, img_width),
            max_area_ratio=max_area_ratio
        )
        
        logger.info(f"📊 Useful objects after filtering: {len(useful_objects_list)}")
        
        # STEP 5: Renumber analyses consecutively (1, 2, 3, 4...)
        analyses_for_crops = []
        for new_n, obj in enumerate(useful_objects_list, start=1):
            analysis = obj['analysis'].copy()
            analysis['n'] = new_n  # Renumber consecutively
            analyses_for_crops.append(analysis)
            obj['new_n'] = new_n
        
        # STEP 6: Generate crops AFTER Claude (OPTIMIZED: only for useful objects)
        logger.info(f"\n✂️  STEP 3: Generating crops for {len(useful_objects_list)} useful objects...")
        
        n_to_crop = save_crops_for_useful_objects(
            image=image,
            analyses=analyses_for_crops,
            useful_objects=useful_objects_list,
            output_dir=self.config['paths']['crops'],
            timestamp=timestamp
        )
        
        # STEP 7: Create final objects with crop paths
        final_objects = []
        for obj in useful_objects_list:
            analysis = obj['analysis']
            detection = obj['detection']
            n = obj['new_n']
            
            crop_path = n_to_crop.get(n)
            if not crop_path:
                logger.warning(f"  ⚠️  No crop for object n={n}, skipping")
                continue
            
            final_obj = {
                'id': f"obj_{timestamp}_{n:03d}",
                'timestamp': timestamp,
                'detection_number': n,
                'thumbnail': crop_path,
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0.0),
                'area': detection.get('area', 0),
                'name': analysis.get('name', 'Unknown object'),
                'category': analysis.get('category', 'other'),
                'condition': analysis.get('condition', 'unknown'),
                'description': analysis.get('description', ''),
                'estimated_value': analysis.get('estimated_value')
            }
            
            final_objects.append(final_obj)
            logger.info(f"  ✅ #{n}: {final_obj['name']}")
        
        # Process missing objects from Claude
        if missing_objects:
            logger.info(f"\n🔍 Processing {len(missing_objects)} missing objects from Claude...")
            # TODO: Generate crops for missing objects if needed
            # For now, we skip them as they don't have corresponding crops
        
        # STEP 8: Save to database
        if final_objects:
            self.database.extend(final_objects)
            self._save_database()
            logger.info(f"💾 Database saved ({len(self.database)} total objects)")
        else:
            logger.warning("⚠️  No useful objects found. Nothing saved to database.")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Processing complete: {len(final_objects)} objects")
        logger.info(f"{'='*60}\n")
        
        return {
            'timestamp': timestamp,
            'raw_image': str(raw_path),
            'visualization': str(vis_path),
            'objects': final_objects
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

