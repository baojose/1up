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

