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
