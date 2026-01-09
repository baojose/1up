"""
⚠️ TEMPORAL: Smart Camera with Intelligent Autofocus

NOTA IMPORTANTE:
Esta es una solución TEMPORAL para el dispositivo externo (cámara USB).
Una vez que el proyecto avance (app móvil, punto limpio automático), 
este módulo debe ser ELIMINADO.

TODO: Eliminar cuando se implemente la app móvil o sistema automático.

Camera utilities for optimal focus and capture.
Max 350 lines.
"""
import cv2
import time
import logging
import numpy as np
import platform
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SmartCamera:
    """
    Camera with intelligent autofocus management.
    
    ⚠️ TEMPORAL: Solo para dispositivo externo (cámara USB).
    Eliminar cuando el proyecto avance.
    """
    
    def __init__(
        self,
        camera_index: int = 1,
        width: int = 1920,
        height: int = 1080,
        autofocus_delay: float = 2.0,
        num_focus_attempts: int = 5
    ):
        """
        Initialize smart camera with autofocus.
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
            autofocus_delay: Seconds to wait for autofocus
            num_focus_attempts: Number of frames to capture for best focus
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.autofocus_delay = autofocus_delay
        self.num_focus_attempts = num_focus_attempts
        
        self.cap = None
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup camera with optimal configuration"""
        logger.info(f"Opening camera {self.camera_index}...")
        
        # Use correct backend for macOS
        is_mac = platform.system() == 'Darwin'
        if is_mac:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Enable autofocus (if supported)
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except:
            logger.debug("Autofocus property not supported on this camera")
        
        # Auto exposure (0.75 = 3/4 auto, gives good balance)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except:
            logger.debug("Auto exposure property not supported on this camera")
        
        logger.info(f"✅ Camera configured: {self.width}x{self.height}")
        
        # Warm up camera (first frames are often bad)
        self._warmup()
    
    def _warmup(self, num_frames: int = 10):
        """Discard initial frames to let camera stabilize"""
        logger.info("Warming up camera...")
        for _ in range(num_frames):
            self.cap.read()
        time.sleep(0.5)
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher value = sharper image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    
    def trigger_autofocus(self):
        """
        Trigger autofocus by toggling it.
        This forces the camera to refocus.
        """
        logger.info("🎯 Triggering autofocus...")
        
        try:
            # Toggle autofocus to force refocus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(0.1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except:
            logger.debug("Autofocus toggle not supported, using delay only")
        
        # Wait for focus to settle
        time.sleep(self.autofocus_delay)
    
    def capture_sharp_image(
        self,
        min_sharpness: float = 20.0,
        max_attempts: int = 3
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Capture the sharpest possible image.
        
        Args:
            min_sharpness: Minimum acceptable sharpness
            max_attempts: Number of autofocus attempts if below threshold
            
        Returns:
            (frame, sharpness) or (None, 0) if failed
        """
        best_sharpness = 0.0
        best_frame = None
        
        for attempt in range(max_attempts):
            logger.info(f"📸 Capture attempt {attempt + 1}/{max_attempts}")
            
            # Trigger autofocus
            self.trigger_autofocus()
            
            # Capture multiple frames and pick the sharpest
            attempt_best_frame = None
            attempt_best_sharpness = 0
            
            for i in range(self.num_focus_attempts):
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    continue
                
                sharpness = self._calculate_sharpness(frame)
                brightness = self._calculate_brightness(frame)
                
                logger.debug(
                    f"  Frame {i+1}/{self.num_focus_attempts}: "
                    f"sharpness={sharpness:.1f}, brightness={brightness:.1f}"
                )
                
                if sharpness > attempt_best_sharpness:
                    attempt_best_sharpness = sharpness
                    attempt_best_frame = frame.copy()
                
                time.sleep(0.2)  # Small delay between captures
            
            logger.info(f"  Best sharpness: {attempt_best_sharpness:.1f}")
            
            # Track overall best
            if attempt_best_sharpness > best_sharpness:
                best_sharpness = attempt_best_sharpness
                best_frame = attempt_best_frame
            
            # Check if acceptable
            if attempt_best_sharpness >= min_sharpness:
                logger.info(f"✅ Sharp image captured (sharpness: {attempt_best_sharpness:.1f})")
                return attempt_best_frame, attempt_best_sharpness
            else:
                logger.warning(
                    f"⚠️ Image too blurry ({attempt_best_sharpness:.1f} < {min_sharpness}), "
                    f"retrying..."
                )
        
        # All attempts failed
        logger.error(
            f"❌ Failed to capture sharp image after {max_attempts} attempts. "
            f"Best sharpness: {best_sharpness:.1f}"
        )
        
        # Return best attempt anyway
        return best_frame, best_sharpness
    
    def capture_for_preview(self) -> np.ndarray:
        """Quick capture for preview/live view (no focus optimization)"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def capture_optimal_image(
    camera_index: int = 1,
    min_sharpness: float = 20.0
) -> Tuple[Optional[np.ndarray], float]:
    """
    One-shot function to capture a sharp image.
    
    ⚠️ TEMPORAL: Solo para dispositivo externo.
    
    Args:
        camera_index: Camera device index
        min_sharpness: Minimum acceptable sharpness
        
    Returns:
        (frame, sharpness) or (None, 0) if failed
    """
    with SmartCamera(camera_index=camera_index) as camera:
        return camera.capture_sharp_image(min_sharpness=min_sharpness)

