"""
Test script for object detection only (no Claude analysis)
Fast way to test SAM 3 detection with camera.
Max 350 lines.
"""
import cv2
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from detector import SAM3Detector
from camera_utils import find_external_camera, enumerate_cameras

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_detection_only():
    """Test SAM 3 detection without Claude analysis."""
    
    # Load config
    logger.info("Loading configuration...")
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Find camera
    logger.info("\n" + "="*60)
    logger.info("Detecting cameras...")
    logger.info("="*60)
    
    cameras = enumerate_cameras()
    if not cameras:
        logger.error("No cameras found!")
        return
    
    # Use preferred index from config, or auto-detect
    preferred_index = config['camera'].get('index')
    camera_index = find_external_camera(preferred_index=preferred_index)
    
    if camera_index is None:
        logger.error("Could not find suitable camera")
        return
    
    # Initialize detector (this may take 10-30 seconds)
    logger.info("\n" + "="*60)
    logger.info("Initializing SAM 3 detector...")
    logger.info("(This may take 10-30 seconds on first run)")
    logger.info("="*60)
    
    try:
        detector = SAM3Detector(
            device=config['sam3']['device']
        )
    except Exception as e:
        logger.error(f"Failed to initialize SAM 3: {e}")
        logger.error("Make sure:")
        logger.error("1. SAM 3 is installed: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .")
        logger.error("2. You have access to SAM 3 checkpoints on HuggingFace")
        logger.error("3. You are authenticated: hf auth login")
        return
    
    # Open camera
    logger.info(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return
    
    # Set resolution
    w, h = config['camera']['resolution']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    logger.info(f"✅ Camera opened ({w}x{h})")
    logger.info("\n" + "="*60)
    logger.info("DETECTION TEST MODE")
    logger.info("  SPACE = Capture and detect objects")
    logger.info("  D = Show detection visualization")
    logger.info("  Q = Quit")
    logger.info("="*60 + "\n")
    
    # Create output directories
    raw_dir = Path(config['paths']['raw_images'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    crops_dir = Path(config['paths']['crops'])
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    show_detections = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Show live view or detection overlay
            display_frame = frame.copy()
            
            if show_detections:
                # Run detection (this is slow, so only on demand)
                logger.info("Running detection...")
                detections = detector.detect_objects(frame)
                logger.info(f"Detected {len(detections)} objects")
                
                # Visualize
                display_frame = detector.visualize_detections(frame, detections)
                
                # Show info
                for i, det in enumerate(detections[:5]):  # Show top 5
                    logger.info(f"  Object {i+1}: bbox={det['bbox']}, conf={det['confidence']:.2f}, area={det['area']}px²")
            
            cv2.imshow('1UP Detection Test - SPACE=Capture, D=Detect, Q=Quit', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Capture and process
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing scene {timestamp}")
                logger.info(f"{'='*60}\n")
                
                # Save raw image
                raw_path = raw_dir / f"scene_{timestamp}.jpg"
                cv2.imwrite(str(raw_path), frame)
                logger.info(f"📸 Saved: {raw_path}")
                
                # Detect objects
                logger.info("Running SAM 3 detection...")
                detections = detector.detect_objects(frame)
                logger.info(f"🔍 Detected {len(detections)} objects")
                
                if len(detections) == 0:
                    logger.warning("No objects detected")
                    continue
                
                # Save visualization
                vis_image = detector.visualize_detections(frame, detections)
                vis_path = raw_dir / f"scene_{timestamp}_viz.jpg"
                cv2.imwrite(str(vis_path), vis_image)
                logger.info(f"📊 Saved visualization: {vis_path}")
                
                # Save crops
                crop_dir = crops_dir / timestamp
                crop_paths = detector.save_crops(
                    frame, detections, str(crop_dir), prefix="obj"
                )
                logger.info(f"✂️  Saved {len(crop_paths)} crops to {crop_dir}")
                
                # Show summary
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ Detection complete: {len(detections)} objects")
                logger.info(f"{'='*60}\n")
                
                # Show top detections
                for i, det in enumerate(detections[:10]):
                    logger.info(f"  #{i+1}: bbox={det['bbox']}, conf={det['confidence']:.3f}, area={det['area']}px²")
                
                logger.info("\nReady for next capture\n")
                
            elif key == ord('d') or key == ord('D'):  # D - Toggle detection overlay
                show_detections = not show_detections
                if show_detections:
                    logger.info("Detection overlay ON (press D again to turn off)")
                else:
                    logger.info("Detection overlay OFF")
                    
            elif key == ord('q'):  # Q - Quit
                break
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("\n👋 Bye!")


if __name__ == "__main__":
    test_detection_only()

