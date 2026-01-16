"""
Live Camera Detection - Visual Object Recognition
Shows camera feed and detects objects on demand.
Max 350 lines.
"""
import sys
import cv2
import yaml
import logging
import os
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Print immediately to verify script starts
print("📦 Loading imports...", flush=True)
sys.stdout.flush()

try:
    from detector import SAM3Detector
    from analyzer import ClaudeAnalyzer
    # NOTE: hybrid_detector is temporarily disabled (causes original_index issues)
    # from hybrid_detector import HybridDetector
    from camera_utils import find_external_camera, enumerate_cameras, open_camera
    from storage import ImageStorage  # Used for save_scene() and create_display_frame()
    from storage_v2 import save_crops_for_useful_objects  # Used for generating crops after Claude
    import json
    from datetime import datetime
    print("✅ Imports loaded successfully", flush=True)
except ImportError as e:
    print(f"❌ Import error: {e}", flush=True)
    print("   Make sure all dependencies are installed:", flush=True)
    print("   pip install -r requirements.txt", flush=True)
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration (useful when script is run multiple times)
)
logger = logging.getLogger(__name__)
print("✅ Logging configured", flush=True)


def _merge_similar_objects(objects: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Fusiona objetos con nombres muy similares para evitar duplicados.
    
    Args:
        objects: Lista de objetos a fusionar
        similarity_threshold: Umbral de similitud (0.0-1.0)
    
    Returns:
        Lista de objetos fusionados
    """
    if len(objects) <= 1:
        return objects
    
    merged = []
    skip_indices = set()
    
    for i, obj1 in enumerate(objects):
        if i in skip_indices:
            continue
        
        name1 = obj1['name'].lower()
        category1 = obj1.get('category', '').lower()
        
        # Buscar objetos similares
        similar_group = [obj1]
        
        for j, obj2 in enumerate(objects[i+1:], start=i+1):
            if j in skip_indices:
                continue
            
            name2 = obj2['name'].lower()
            category2 = obj2.get('category', '').lower()
            
            # Detectar similitud por palabras clave comunes
            is_similar = False
            
            # Caso 1: Mismo tipo de objeto (frasco, botella, etc.)
            keywords = ['frasco', 'botella', 'especias', 'condimento', 'aceite', 'vinagre', 
                       'cuchillo', 'cuchillos', 'plato', 'platos', 'taza', 'tazas']
            
            for keyword in keywords:
                if keyword in name1 and keyword in name2:
                    is_similar = True
                    break
            
            # Caso 2: Misma categoría y nombres muy similares
            if category1 == category2 and category1 != 'other':
                # Calcular similitud simple (palabras comunes)
                words1 = set(name1.split())
                words2 = set(name2.split())
                common_words = words1.intersection(words2)
                if len(common_words) >= 2:  # Al menos 2 palabras comunes
                    is_similar = True
            
            # Caso 3: Nombres casi idénticos (diferencia mínima)
            if name1 in name2 or name2 in name1:
                if abs(len(name1) - len(name2)) <= 5:  # Diferencia pequeña
                    is_similar = True
            
            if is_similar:
                similar_group.append(obj2)
                skip_indices.add(j)
        
        # Si hay grupo de similares, crear objeto fusionado
        if len(similar_group) > 1:
            # Usar el objeto con mayor área como base
            base_obj = max(similar_group, key=lambda o: o.get('area', 0))
            
            # Crear nombre grupal
            base_name = base_obj['name']
            count = len(similar_group)
            
            # Detectar tipo de objeto
            if 'frasco' in base_name.lower() or 'especias' in base_name.lower():
                merged_name = f"Especiero con {count} frascos de especias"
            elif 'botella' in base_name.lower():
                merged_name = f"Set de {count} botellas"
            elif 'cuchillo' in base_name.lower():
                merged_name = f"Set de {count} cuchillos"
            elif 'plato' in base_name.lower():
                merged_name = f"Set de {count} platos"
            else:
                merged_name = f"Grupo de {count} {base_obj.get('category', 'objetos')}"
            
            merged_obj = {
                'id': base_obj['id'],
                'timestamp': base_obj['timestamp'],
                'detection_number': base_obj.get('detection_number', 1),
                'thumbnail': base_obj['thumbnail'],  # Usar thumbnail del objeto base
                'bbox': base_obj['bbox'],  # Usar bbox del objeto base
                'confidence': max(o.get('confidence', 0.0) for o in similar_group),
                'area': sum(o.get('area', 0) for o in similar_group),  # Sumar áreas
                'name': merged_name,
                'category': base_obj.get('category', 'other'),
                'condition': base_obj.get('condition', 'unknown'),
                'description': f"Conjunto de {count} objetos similares: {base_obj.get('description', '')}",
                'estimated_value': base_obj.get('estimated_value')
            }
            
            merged.append(merged_obj)
            logger.info(f"   🔗 Merged {count} similar objects: '{base_name}' → '{merged_name}'")
        else:
            # Objeto único, mantenerlo
            merged.append(obj1)
    
    return merged


def live_detection():
    """Live camera feed with on-demand object detection."""
    
    try:
    # Load config
        logger.info("="*60)
        logger.info("Starting live detection...")
        logger.info("="*60)
    logger.info("Loading configuration...")
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded")
    except FileNotFoundError:
        logger.error("❌ config.yaml not found!")
        logger.error("   Make sure you're running from the project root directory")
        return
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Testing mode: Clean up previous images at startup
    if config.get('testing', {}).get('auto_cleanup', False):
        logger.info("\n🧹 Testing mode: Cleaning up previous images...")
        from pathlib import Path
        import shutil
        
        # Clean database
        db_path = Path('database/objects.json')
        if db_path.exists():
            with open(db_path, 'w') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            logger.info("   ✅ Base de datos limpiada")
        
        # Delete all raw images
        raw_dir = Path('images/raw')
        if raw_dir.exists():
            removed = sum(1 for f in raw_dir.glob("*") if f.is_file() and f.unlink() or True)
            if removed > 0:
                logger.info(f"   ✅ {removed} imágenes raw eliminadas")
        
        # Delete all crops
        crops_dir = Path('images/crops')
        if crops_dir.exists():
            removed_dirs = 0
            removed_files = 0
            for crop_dir in crops_dir.iterdir():
                if crop_dir.is_dir():
                    crop_count = len(list(crop_dir.glob('*.jpg')))
                    shutil.rmtree(crop_dir)
                    removed_dirs += 1
                    removed_files += crop_count
            if removed_dirs > 0:
                logger.info(f"   ✅ {removed_dirs} directorios de crops eliminados ({removed_files} crops)")
        
        logger.info("   ✅ Limpieza completa\n")
    
    # Get camera source from config (can be int index or RTSP URL)
    camera_source = config['camera'].get('source', config['camera'].get('index', 1))
    resolution = config['camera']['resolution']
    buffer_size = config['camera'].get('buffer_size', 1)
    
    import platform
    import time
    
    # Check if source is RTSP URL or integer index
    is_rtsp = isinstance(camera_source, str) and camera_source.startswith('rtsp://')
    
    if is_rtsp:
        # RTSP stream - use open_camera() directly
        logger.info("\n" + "="*60)
        logger.info("Opening RTSP camera stream...")
        logger.info("="*60)
        
        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else None
        cap = open_camera(
            source=camera_source,
            resolution=resolution,
            buffer_size=buffer_size,
            backend=backend
        )
        
        if cap is None:
            logger.error("Failed to open RTSP stream")
            return
        
        # RTSP streams may need a moment to stabilize
        time.sleep(1.0)
        
    else:
        # USB/webcam camera - use existing detection logic
    logger.info("\n" + "="*60)
    logger.info("Detecting cameras...")
    logger.info("="*60)
    
        preferred_index = int(camera_source) if isinstance(camera_source, (int, str)) and str(camera_source).isdigit() else camera_source
    allow_iphone = config['camera'].get('allow_iphone', False)
    logger.info(f"Config specifies camera index: {preferred_index}")
    logger.info(f"Allow iPhone/Continuity Camera: {allow_iphone}")
    
        logger.info("Enumerating cameras (this may take a few seconds)...")
    cameras = enumerate_cameras(allow_iphone=allow_iphone)
        logger.info(f"✅ Camera enumeration complete: {len(cameras)} camera(s) found")
    if not cameras:
        logger.error("No cameras found!")
        logger.error("Please connect a camera and try again")
        return
    
    # Use smart detection (by name first, then fallback to index)
    logger.info("Using smart camera detection (by name preferred)...")
    camera_index = find_external_camera(preferred_index=preferred_index, allow_iphone=allow_iphone)
    
    if camera_index is None:
        # Fallback: use first available camera (even if it's built-in)
        if cameras:
            camera_index = cameras[0]['index']
            logger.warning(f"⚠️  No external camera found, using available camera {camera_index}")
            logger.warning(f"   (This may be the built-in camera)")
        else:
            logger.error("Could not find any camera")
            return
    
    # Log which camera was selected
    for cam in cameras:
        if cam['index'] == camera_index:
            logger.info(f"✅ Selected camera {camera_index}: {cam['width']}x{cam['height']}")
            break
    
        # Open camera using open_camera() function
    logger.info(f"\nOpening camera {camera_index}...")
        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else None
        cap = open_camera(
            source=camera_index,
            resolution=resolution,
            backend=backend
        )
    
        if cap is None:
        logger.error(f"Failed to open camera {camera_index}")
        return
    
    # Wait a bit for camera to initialize
    time.sleep(0.5)
    
    # Read a few frames to "warm up" the camera and keep it active
    logger.info("Warming up camera...")
    warmup_success = False
    for i in range(10):
        ret, _ = cap.read()
        if ret:
            warmup_success = True
            break
        time.sleep(0.1)
    
    if not warmup_success:
        logger.error("⚠️  Camera opened but cannot read frames during warmup")
        logger.error("   This may be a permissions issue or the camera is in use by another app")
        logger.error("   Try:")
        logger.error("   1. Check System Preferences > Security & Privacy > Camera")
        logger.error("   2. Close other apps using the camera")
        logger.error("   3. Try a different camera index")
        cap.release()
        return
    
    # Verify resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"✅ Camera opened: {actual_w}x{actual_h}")
    
    # Keep camera active by reading frames periodically during SAM 3 init
    logger.info("\n" + "="*60)
    logger.info("Initializing SAM 3 detector...")
    logger.info("(This may take 10-30 seconds on first run, camera will stay active)")
    logger.info("="*60)
    
    # Initialize detector (camera stays open during this)
    try:
        logger.info("Loading SAM 3 model (this is the slow part)...")
        detector = SAM3Detector(
            device=config['sam3']['device']
        )
        logger.info("✅ SAM 3 initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SAM 3: {e}")
        logger.error("Make sure:")
        logger.error("1. SAM 3 is installed: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .")
        logger.error("2. You have access to SAM 3 checkpoints on HuggingFace")
        logger.error("3. You are authenticated: hf auth login")
        cap.release()
        return
    
    # Initialize Claude analyzer (optional, for object recognition)
    analyzer = None
    hybrid_detector = None
    api_key = os.environ.get(config['claude']['api_key_env'])
    if api_key:
        try:
            analyzer = ClaudeAnalyzer(
                api_key=api_key,
                model=config['claude']['model'],
                max_tokens=config['claude']['max_tokens'],
                temperature=config['claude'].get('temperature', 0)
            )
            logger.info("✅ Claude analyzer ready")
            
            # NOTE: hybrid_detector is temporarily disabled (causes original_index issues)
            # Initialize hybrid detector (SAM + Claude validation)
            # hybrid_detector = HybridDetector(detector, analyzer)
            # logger.info("✅ Hybrid detector initialized (SAM + Claude validation)")
        except Exception as e:
            logger.warning(f"⚠️  Claude analyzer not available: {e}")
    else:
        logger.info("ℹ️  Claude API key not set - object recognition disabled")
        logger.info("   Set CLAUDE_API_KEY to enable object recognition")
    
    # Verify camera still works after initialization
    logger.info("\nVerifying camera is still active...")
    time.sleep(0.2)
    ret, verify_frame = cap.read()
    if not ret or verify_frame is None:
        logger.error("⚠️  Camera stopped working after initialization")
        logger.error("   Attempting to reopen camera...")
        cap.release()
        time.sleep(1)
        
        # Try to reopen camera using open_camera()
        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else None
        if is_rtsp:
            cap = open_camera(
                source=camera_source,
                resolution=resolution,
                buffer_size=buffer_size,
                backend=backend
            )
        else:
            cap = open_camera(
                source=camera_index,
                resolution=resolution,
                backend=backend
            )
        
        if cap is None:
            logger.error(f"❌ Failed to reopen camera")
            logger.error("   Camera may be in use by another app")
            logger.error("   Try closing other apps or restarting the camera")
            return
        
        # Test again
        time.sleep(0.5)
        ret, verify_frame = cap.read()
        if not ret or verify_frame is None:
            logger.error("❌ Camera reopened but cannot read frames")
            logger.error("   This may be a permissions issue")
            cap.release()
            return
        
        logger.info("✅ Camera reopened successfully")
    
    logger.info("\n" + "="*60)
    logger.info("LIVE DETECTION MODE")
    logger.info("  SPACE = Detect objects in current frame")
    logger.info("  S = Save scene (image + crops)")
    logger.info("  A = Analyze objects with Claude (if available)")
    logger.info("  L = List detected objects")
    logger.info("  C = Clear detections")
    logger.info("  Q = Quit")
    logger.info("="*60 + "\n")
    
    # Initialize storage
    storage = ImageStorage(config)
    
    # Load database for saving useful objects
    db_path = Path(config['paths']['database'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        with open(db_path) as f:
            database = json.load(f)
    else:
        database = []
    logger.debug(f"Database loaded: {len(database)} objects")
    
    # State
    current_detections: List[Dict[str, Any]] = []
    detection_frame = None
    object_analyses: Dict[int, Dict[str, Any]] = {}  # obj_id -> analysis
    temp_dir = Path(tempfile.mkdtemp(prefix="1up_crops_"))
    logger.debug(f"Temporary crops directory: {temp_dir}")
    
    try:
        consecutive_failures = 0
        max_failures = 10
        first_frame_logged = False
        
        # For RTSP streams, add timeout handling to prevent indefinite blocking
        is_rtsp = isinstance(camera_source, str) and camera_source.startswith('rtsp://')
        frame_timeout = 5.0 if is_rtsp else 1.0  # Longer timeout for RTSP
        
        import time
        while True:
            # For RTSP, implement timeout to prevent indefinite blocking
            frame_start_time = time.time()
            ret, frame = cap.read()
            frame_read_time = time.time() - frame_start_time
            
            # Log if read took too long (warns about potential issues)
            # ⚠️ IMPORTANTE: Para 4K HEVC en Mac Intel, frames pueden tardar 5-15s
            # Los errores HEVC son warnings de FFmpeg, no bloquean, pero ralentizan
            if is_rtsp and frame_read_time > 3.0:
                if frame_read_time > 10.0:
                    logger.warning(f"⚠️  Frame read took {frame_read_time:.1f}s (4K HEVC decoding on Mac Intel)")
                    logger.debug("   ⚠️  HEVC errors in logs are warnings, not fatal - processing continues")
                else:
                    logger.warning(f"⚠️  Frame read took {frame_read_time:.1f}s (RTSP may be slow or unstable)")
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error(f"Failed to read frame {max_failures} times in a row")
                    if is_rtsp:
                        logger.error("RTSP stream may have disconnected")
                        logger.error("  - Check camera network connection")
                        logger.error("  - Verify camera is online and accessible")
                        logger.error("  - Try reducing resolution or FPS in config.yaml")
                    else:
                    logger.error("Camera may have disconnected or is in use by another app")
                    break
                time.sleep(0.1)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Verify frame is valid
            if frame is None or frame.size == 0:
                logger.warning("⚠️  Received empty frame, skipping...")
                continue
            
            # Log frame info on first successful read (debug)
            if not first_frame_logged:
                logger.info(f"✅ Reading frames: {frame.shape[1]}x{frame.shape[0]}")
                first_frame_logged = True
            
            # Create display frame
            # If we have detections, show the captured frame with detections (frozen)
            # Otherwise, show live camera feed
            if current_detections and detection_frame is not None:
                # Use storage to create display frame with contours
                display_frame = storage.create_display_frame(
                    detection_frame,
                    current_detections,
                    object_analyses
                )
                
                # Add text overlay with detection count
                status_text = f"Objects: {len(current_detections)}"
                if object_analyses:
                    status_text += f" | Analyzed: {len(object_analyses)}"
                status_text += " (FROZEN - C=clear, A=analyze, L=list)"
                
                cv2.putText(
                    display_frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                # Show live camera feed (no detections)
                display_frame = frame.copy()
            
            # Verify display_frame is valid before showing
            if display_frame is None or display_frame.size == 0:
                logger.warning("⚠️  Display frame is empty, skipping...")
                continue
            
            # CRITICAL: Resize display frame for preview (4K is too large, causes pixelation)
            # Use fixed window size to prevent OpenCV automatic scaling issues
            display_h, display_w = display_frame.shape[:2]
            target_preview_width = 1280  # Fixed preview width (HD)
            target_preview_height = 720  # Fixed preview height (HD 16:9)
            
            # Calculate scale to fit within target while maintaining aspect ratio
            scale_w = target_preview_width / display_w
            scale_h = target_preview_height / display_h
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
            
            if scale < 1.0:
                new_display_w = int(display_w * scale)
                new_display_h = int(display_h * scale)
                
                # Resize with high-quality interpolation (LANCZOS4 is best for downscaling)
                display_frame_preview = cv2.resize(
                    display_frame,
                    (new_display_w, new_display_h),
                    interpolation=cv2.INTER_LANCZOS4  # Best quality for downscaling
                )
            else:
                display_frame_preview = display_frame.copy()
                new_display_w, new_display_h = display_w, display_h
            
            # Create fixed-size canvas with black background (prevents OpenCV window scaling issues)
            canvas = np.zeros((target_preview_height, target_preview_width, 3), dtype=np.uint8)
            
            # Center the preview frame in the canvas
            y_offset = (target_preview_height - new_display_h) // 2
            x_offset = (target_preview_width - new_display_w) // 2
            canvas[y_offset:y_offset + new_display_h, x_offset:x_offset + new_display_w] = display_frame_preview
            
            # Show fixed-size frame (prevents pixelation from OpenCV window scaling)
            cv2.imshow('1UP Live Detection - SPACE=Detect, C=Clear, Q=Quit', canvas)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - Detect objects
                logger.info("\n🔍 Running detection...")
                logger.info("(This may take 5-15 seconds)")
                
                # Capture a fresh frame for detection
                ret, capture_frame = cap.read()
                if not ret or capture_frame is None or capture_frame.size == 0:
                    logger.error("⚠️  Failed to capture frame. Try again.")
                    continue
                
                logger.info(f"📸 Frame captured: {capture_frame.shape[1]}x{capture_frame.shape[0]}")
                
                # Run detection on the captured frame
                try:
                    # Detect objects with smart filtering if enabled
                    filter_config = config.get('sam3', {}).get('filtering', {})
                    apply_filtering = filter_config.get('enabled', False)
                    enhance_image = config.get('sam3', {}).get('enhance_image', False)
                    text_prompt = config.get('sam3', {}).get('text_prompt', '') or None
                    
                    logger.info(f"🔍 Running SAM 3 detection...")
                    logger.info(f"   Filtering: {apply_filtering}")
                    logger.info(f"   Text prompt: {text_prompt or 'automatic (visual)'}")
                    
                    if apply_filtering:
                        current_detections = detector.detect_objects(
                            capture_frame,
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
                        current_detections = detector.detect_objects(
                            capture_frame, 
                            enhance_image=enhance_image,
                            text_prompt=text_prompt
                        )
                    
                    # CRITICAL: Validate and fix original_index for all detections immediately after SAM
                    # This ensures original_index is always in valid range [0, len(current_detections))
                    if current_detections:
                    logger.debug(f"🔍 Validating original_index for {len(current_detections)} detections...")
                    fixed_count = 0
                    for i, det in enumerate(current_detections):
                        orig_idx = det.get('original_index', -1)
                        if orig_idx < 0 or orig_idx >= len(current_detections):
                            logger.warning(f"  🔧 Fixing invalid original_index for detection {i}: was {orig_idx}, setting to {i} (total detections: {len(current_detections)})")
                            det['original_index'] = i
                            fixed_count += 1
                    if fixed_count > 0:
                        logger.warning(f"  ⚠️  Fixed {fixed_count} invalid original_index values")
                    else:
                        logger.debug(f"  ✅ All original_index values are valid")
                    
                    detection_frame = capture_frame.copy()
                    else:
                        logger.warning("⚠️  No objects detected")
                        detection_frame = None
                    
                    # CRITICAL: Log detections IMMEDIATELY - use print AND logger to ensure visibility
                    print("\n" + "="*60)
                    print("📋 LISTA DE OBJETOS DETECTADOS POR SAM 3:")
                    print("="*60)
                    logger.info("\n" + "="*60)
                    logger.info("📋 LISTA DE OBJETOS DETECTADOS POR SAM 3:")
                    logger.info("="*60)
                    
                    if len(current_detections) == 0:
                        print("  ⚠️  No objects detected. Try adjusting camera or lighting.")
                        logger.warning("  ⚠️  No objects detected. Try adjusting camera or lighting.")
                    else:
                        for i, det in enumerate(current_detections):
                            x, y, w, h_bbox = det['bbox']
                            area = det.get('area', 0)
                            conf = det.get('confidence', 0.0)
                            line = f"  #{i+1}: bbox=({x},{y},{w},{h_bbox}), conf={conf:.3f}, área={area}px²"
                            print(line)
                            logger.info(line)
                        print("="*60)
                        print(f"✅ Total: {len(current_detections)} objetos detectados por SAM 3\n")
                        logger.info("="*60)
                        logger.info(f"✅ Total: {len(current_detections)} objetos detectados por SAM 3\n")
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower() or "MPS" in error_msg:
                        logger.error(f"❌ MPS out of memory: {error_msg}")
                        logger.error("")
                        logger.error("💡 SOLUCIÓN: SAM 3 requiere demasiada memoria en MPS")
                        logger.error("")
                        logger.error("Opciones:")
                        logger.error("  1. Cambiar a CPU en config.yaml: sam3.device: 'cpu'")
                        logger.error("  2. Reducir resolución cámara a 1080p en config.yaml")
                        logger.error("  3. Procesar imágenes más pequeñas")
                        logger.error("")
                        logger.error("💡 CPU es más lento pero funciona (no hay límite de memoria)")
                        current_detections = []
                        detection_frame = None
                    else:
                        raise  # Re-raise if it's a different error
                except Exception as e:
                    logger.error(f"❌ Error during detection: {e}")
                    logger.error("   This might be a device mismatch issue (CPU/MPS)")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.info("   Try again or check config.yaml device settings")
                    # Don't crash - just continue with empty detections
                    current_detections = []
                    detection_frame = None
            
            elif key == ord('s') or key == ord('S'):  # S - Save scene with hybrid detection
                logger.info("\n" + "="*60)
                logger.info("💾 SAVE SCENE - Starting process...")
                logger.info("="*60)
                
                # ⚠️ TEMPORAL: Smart autofocus capture (only for external USB camera)
                # TODO: Eliminar cuando se implemente app móvil o sistema automático
                autofocus_config = config.get('camera', {}).get('autofocus', {})
                autofocus_enabled = autofocus_config.get('enabled', False)
                
                if autofocus_enabled:
                    logger.info("\n🎯 TEMPORAL: Using smart autofocus (external USB camera only)...")
                    logger.info("   ⚠️  This will be removed when mobile app is implemented")
                    try:
                        from smart_camera import SmartCamera
                        smart_cam = SmartCamera(
                            camera_index=camera_index,
                            width=config['camera']['resolution'][0],
                            height=config['camera']['resolution'][1],
                            autofocus_delay=autofocus_config.get('autofocus_delay', 2.0),
                            num_focus_attempts=autofocus_config.get('focus_attempts', 5)
                        )
                        
                        min_sharpness = config.get('camera', {}).get('quality_check', {}).get('min_sharpness', 20.0)
                        max_attempts = autofocus_config.get('max_autofocus_attempts', 3)
                        
                        logger.info(f"📸 Capturing image with autofocus optimization...")
                        autofocus_frame, sharpness = smart_cam.capture_sharp_image(
                            min_sharpness=min_sharpness,
                            max_attempts=max_attempts
                        )
                        smart_cam.release()
                        
                        if autofocus_frame is not None:
                            detection_frame = autofocus_frame
                            logger.info(f"✅ Autofocus capture complete: nitidez={sharpness:.1f}")
                            # Re-run detection on the new sharp frame
                            logger.info("🔍 Re-running detection on sharp frame...")
                            filter_config = config.get('sam3', {}).get('filtering', {})
                            apply_filtering = filter_config.get('enabled', False)
                            enhance_image = config.get('sam3', {}).get('enhance_image', False)
                            text_prompt = config.get('sam3', {}).get('text_prompt', '') or None
                            
                            current_detections = detector.detect_objects(
                                detection_frame,
                                apply_filtering=apply_filtering,
                                enhance_image=enhance_image,
                                text_prompt=text_prompt,
                                min_area=filter_config.get('min_area') if apply_filtering else None,
                                max_area_ratio=filter_config.get('max_area_ratio') if apply_filtering else None,
                                min_aspect_ratio=filter_config.get('min_aspect_ratio') if apply_filtering else None,
                                max_aspect_ratio=filter_config.get('max_aspect_ratio') if apply_filtering else None,
                                nms_iou_threshold=filter_config.get('nms_iou_threshold') if apply_filtering else None
                            )
                            logger.info(f"✅ Detection complete: {len(current_detections)} objects")
                        else:
                            logger.error("❌ Failed to capture sharp image with autofocus")
                            logger.warning("   Falling back to existing detection_frame (if available)")
                    except Exception as e:
                        logger.warning(f"⚠️  Smart autofocus failed: {e}")
                        logger.warning("   Falling back to existing detection_frame (if available)")
                
                if detection_frame is None:
                    logger.warning("⚠️  No frame to save. Press SPACE first or enable autofocus.")
                    continue
                
                # Verify detection_frame is valid
                if detection_frame is None or detection_frame.size == 0:
                    logger.error("⚠️  Invalid detection frame. Press SPACE to detect again.")
                    continue
                
                logger.info(f"✅ Frame valid: {detection_frame.shape[1]}x{detection_frame.shape[0]}")
                logger.info(f"✅ Detections: {len(current_detections)} objects")
                
                # Step 0: Validate image quality (blur detection)
                from image_quality import is_image_acceptable, calculate_sharpness_score
                quality_config = config.get('camera', {}).get('quality_check', {})
                quality_enabled = quality_config.get('enabled', True)
                min_sharpness = quality_config.get('min_sharpness', 20.0)  # Umbral para rechazar (muy borrosa)
                warning_sharpness = quality_config.get('warning_sharpness', 50.0)  # Umbral para warning
                
                if quality_enabled:
                    logger.info(f"\n🔍 Validating image quality (blur detection)...")
                    sharpness = calculate_sharpness_score(detection_frame)
                    
                    # Rechazar solo si está MUY borrosa (< min_sharpness)
                    if sharpness < min_sharpness:
                        logger.error(f"❌ IMAGEN RECHAZADA: Imagen demasiado borrosa (nitidez={sharpness:.1f} < {min_sharpness:.1f})")
                        logger.error(f"   ⚠️  La imagen está extremadamente borrosa. Por favor, enfoca la cámara y vuelve a intentar.")
                        logger.error(f"   💡 Consejo: Espera 1-2 segundos después de enfocar antes de capturar.")
                        continue  # Skip this frame
                    
                    # Warning si está moderadamente borrosa (pero continuar)
                    if sharpness < warning_sharpness:
                        logger.warning(f"⚠️  ADVERTENCIA: Imagen moderadamente borrosa (nitidez={sharpness:.1f} < {warning_sharpness:.1f})")
                        logger.warning(f"   Continuando con el procesamiento, pero los resultados pueden ser menos precisos.")
                        logger.warning(f"   💡 Para mejores resultados, intenta enfocar mejor la cámara.")
                    else:
                        logger.info(f"✅ Calidad de imagen aceptable: nitidez={sharpness:.1f}")
                    
                    logger.info(f"\n🔄 Starting hybrid detection (SAM + Claude validation)...")
                    logger.info(f"   Frame size: {detection_frame.shape[1]}x{detection_frame.shape[0]}")
                    
                    # Step 1: Save temporary scene image for Claude
                    temp_scene_data = storage.save_scene(
                        image=detection_frame,
                        detections=current_detections if current_detections else [],
                        camera_id=f"CAM{camera_index}",
                        metadata={'temp': True}
                    )
                    
                    # Step 2: Use SAM only (hybrid_detector temporarily disabled)
                    # NOTE: hybrid_detector is disabled due to original_index issues
                    # TODO: Fix hybrid_detector to properly assign original_index
                    use_hybrid = False  # Temporarily disabled
                    hybrid_detector = None  # Not initialized
                    
                    if False and use_hybrid and hybrid_detector and analyzer:  # Disabled
                        logger.info(f"   Using hybrid detector (SAM + Claude)...")
                        logger.info(f"   ⏳ This may take 30-60 seconds...\n")
                        
                        try:
                            # Hybrid detection: validates SAM detections + finds missing objects
                            final_detections, _ = hybrid_detector.detect_with_validation(
                                image=detection_frame,
                                scene_path=temp_scene_data['raw_image']
                            )
                            
                            # Update current_detections with hybrid results
                            if final_detections:
                                # CRITICAL: Reassign original_index for all detections to ensure they're in valid range
                                # This fixes issues where hybrid_detector adds objects with invalid original_index
                                max_original_idx = len(current_detections) - 1  # Max valid original_index from SAM
                                for i, det in enumerate(final_detections):
                                    # Preserve original_index if it's valid, otherwise assign current index
                                    orig_idx = det.get('original_index', -1)
                                    if orig_idx < 0 or orig_idx > max_original_idx:
                                        # Object was added by hybrid_detector or has invalid index
                                        # Assign a safe index (use enumeration index, but mark it)
                                        det['original_index'] = i
                                        det['source'] = det.get('source', 'hybrid_fixed')
                                        logger.debug(f"   Fixed original_index for detection {i}: was {orig_idx}, now {i}")
                                current_detections = final_detections
                                logger.info(f"✅ Hybrid detection complete: {len(final_detections)} objects")
                                logger.info(f"   (SAM: {len(current_detections) - sum(1 for d in final_detections if d.get('source') == 'claude_suggestion')}, "
                                          f"Claude found: {sum(1 for d in final_detections if d.get('source', '').startswith('claude'))})")
                            else:
                                logger.warning("⚠️  Hybrid detection returned no objects, using SAM detections")
                                
                        except Exception as e:
                            logger.error(f"❌ Hybrid detection failed: {e}")
                            logger.warning("   Falling back to SAM detections only")
                    else:
                        logger.info(f"   Using SAM detections only (hybrid detector disabled for debugging)")
                    
                    # Step 3: Save scene WITHOUT crops first (crops will be generated after Claude)
                    logger.info(f"\n💾 Saving scene (crops will be generated after Claude analysis)...")
                    logger.info(f"   Objects: {len(current_detections)}")
                    
                    # Get camera info for metadata
                    camera_info = None
                    for cam in cameras:
                        if cam['index'] == camera_index:
                            camera_info = cam
                            break
                    
                    # Save scene WITHOUT crops (crops generated later)
                    scene_data = storage.save_scene(
                        image=detection_frame,
                        detections=current_detections,
                        camera_id=f"CAM{camera_index}",
                        metadata={
                            'camera_resolution': f"{camera_info['width']}x{camera_info['height']}" if camera_info else None,
                            'num_detections': len(current_detections),
                            'analyzed': len(object_analyses),
                            'frame_size': f"{detection_frame.shape[1]}x{detection_frame.shape[0]}",
                            'hybrid_detection': False  # Temporarily disabled
                        }
                    )
                    # NOTE: scene_data['crops'] will be empty - crops generated after Claude
                    
                    logger.info(f"\n✅ Scene saved: {scene_data['scene_id']}")
                    logger.info(f"   📸 Raw image: {Path(scene_data['raw_image']).absolute()}")
                    logger.info(f"   📊 Visualization: {Path(scene_data['visualization']).absolute()}")
                    if scene_data['crops']:
                        # Filter out None values before accessing
                        valid_crops = [c for c in scene_data['crops'] if c is not None] if scene_data['crops'] else []
                        if valid_crops:
                            logger.info(f"   ✂️  Crops: {len(valid_crops)} objects in {Path(valid_crops[0]).parent}")
                        else:
                            logger.warning(f"   ⚠️  No valid crops found (all were None)")
                    metadata_path = Path(scene_data['raw_image']).parent / f"scene_{scene_data['scene_id']}_meta.json"
                    logger.info(f"   📄 Metadata: {metadata_path.absolute()}")
                    logger.info(f"\n💡 Files saved to: {Path(scene_data['raw_image']).parent.absolute()}")
                    
                    # Verify files exist
                    raw_exists = os.path.exists(scene_data['raw_image'])
                    vis_exists = os.path.exists(scene_data['visualization'])
                    # Filter out None values (gaps in original_index) before checking existence
                    valid_crops = [c for c in scene_data['crops'] if c is not None] if scene_data['crops'] else []
                    crops_exist = all(os.path.exists(c) for c in valid_crops) if valid_crops else False
                    logger.info(f"   ✓ Raw image exists: {raw_exists}")
                    logger.info(f"   ✓ Visualization exists: {vis_exists}")
                    logger.info(f"   ✓ All crops exist: {crops_exist}")
                    
                    # Análisis con Claude (si está disponible) - 1 SOLA LLAMADA API
                    if analyzer and current_detections:
                        # CRITICAL: NO PRE-FILTERING - Send ALL objects to Claude
                        # Claude will decide what's useful, not us
                        # FILOSOFÍA: "SAM detecta TODO, Claude decide qué entra"
                        all_detections = current_detections
                        original_indices = []  # Maps detection_index → original_index
                        
                        for i, d in enumerate(all_detections):
                            # CRITICAL: Use original_index if available (preserves order before sorting)
                            # Otherwise fall back to enumeration index
                            original_idx = d.get('original_index', i)
                            original_indices.append(original_idx)  # Store original index
                        
                        logger.info(f"\n🔍 RESUMEN DE DETECCIÓN SAM 3:")
                        logger.info("="*60)
                        logger.info(f"   Total detectados por SAM 3: {len(all_detections)} objetos")
                        logger.info(f"   ✅ TODOS los objetos se enviarán a Claude (SIN filtros previos)")
                        logger.info("="*60)
                        logger.info("")
                        
                        logger.info(f"\n🤖 Analyzing with Claude (1 API call: validation + analysis + missing objects)...")
                        logger.info(f"   Scene image + {len(all_detections)} bounding boxes (TODOS los objetos)")
                        logger.info(f"   ⏳ Claude decidirá qué objetos son útiles...")
                        logger.info(f"   ⏳ This may take 30-60 seconds...\n")
                        
                        try:
                            # SINGLE API CALL: validation + analysis + missing objects
                            # Send ALL detections - Claude will filter intelligently
                            claude_response = analyzer.analyze_scene_with_validation(
                                scene_path=scene_data['raw_image'],
                                detections=all_detections,  # ALL objects, no filtering
                                language="spanish"
                            )
                            
                            # Extract validated objects (with full analysis) and missing objects
                            validated_analyses = claude_response.get('validated_objects', [])
                            missing_objects = claude_response.get('missing_objects', [])
                            
                            logger.info(f"✅ Claude response: {len(validated_analyses)} validated, {len(missing_objects)} missing found")
                            
                            # CRITICAL: Only use objects that Claude actually validated
                            # DO NOT create fallbacks - if Claude didn't validate it, it's likely not useful
                            if len(validated_analyses) < len(all_detections):
                                logger.info(f"📊 Claude validó {len(validated_analyses)}/{len(all_detections)} objetos")
                                logger.info(f"   ✅ Claude filtró {len(all_detections) - len(validated_analyses)} objetos como no útiles")
                                
                                # Log which objects were validated
                                validated_ns = {a.get('n') for a in validated_analyses if a.get('n')}
                                missing_ns = set(range(1, len(all_detections) + 1)) - validated_ns
                                if missing_ns:
                                    logger.debug(f"   Objetos NO validados por Claude: {sorted(missing_ns)}")
                                    logger.debug(f"   Estos objetos NO se guardarán (Claude decidió que no son útiles)")
                            
                            # Use ONLY validated_analyses - no fallbacks
                            analyses = validated_analyses
                            logger.info(f"✅ Usando {len(analyses)} objetos validados por Claude")
                            
                            # Filtrar solo objetos útiles (solo los que Claude marcó como useful='yes')
                            useful_objects = []
                            skipped_count = 0
                            timestamp = scene_data['timestamp']
                            
                            # Get image dimensions (no filtering by size - Claude decides)
                            img_height, img_width = detection_frame.shape[:2]
                            
                            # NOTE: Crops will be generated AFTER Claude analysis (only for useful objects)
                            # This eliminates mapping issues: n=1 → obj_001.jpg (always matches)
                            
                            # Log what Claude validated
                            logger.info(f"\n📋 OBJETOS VALIDADOS POR CLAUDE:")
                            logger.info("="*60)
                            if not analyses:
                                logger.warning("   ⚠️  Claude no validó ningún objeto")
                            else:
                                for analysis in analyses:
                                        n = analysis.get('n', 0)
                                        useful = analysis.get('useful', 'no').lower() == 'yes'
                                        name = analysis.get('name', 'Unknown')
                                        reason = analysis.get('reason', '')
                                        category = analysis.get('category', 'unknown')
                                        
                                        # Get original detection index for reference
                                        if n >= 1 and n <= len(all_detections):
                                            filtered_idx = n - 1
                                            if filtered_idx < len(original_indices):
                                                original_idx = original_indices[filtered_idx]
                                                det = all_detections[filtered_idx]
                                                area = det.get('area', 0)
                                                conf = det.get('confidence', 0.0)
                                                x, y, w, h = det.get('bbox', [0, 0, 0, 0])
                                                
                                                if useful:
                                                    logger.info(f"  ✅ #{n} (idx={original_idx}): {name} [{category}] - USEFUL")
                                                    logger.info(f"     área={area}px², conf={conf:.3f}, bbox=({x},{y},{w},{h})")
                                                else:
                                                    logger.info(f"  ❌ #{n} (idx={original_idx}): {name} [{category}] - NOT USEFUL")
                                                    logger.info(f"     razón: {reason}")
                                                    logger.info(f"     área={area}px², conf={conf:.3f}, bbox=({x},{y},{w},{h})")
                                            else:
                                                if useful:
                                                    logger.info(f"  ✅ #{n}: {name} [{category}] - USEFUL")
                                                else:
                                                    logger.info(f"  ❌ #{n}: {name} [{category}] - NOT USEFUL ({reason})")
                                        else:
                                            if useful:
                                                logger.info(f"  ✅ #{n}: {name} [{category}] - USEFUL")
                                            else:
                                                logger.info(f"  ❌ #{n}: {name} [{category}] - NOT USEFUL ({reason})")
                                
                                # Log objetos que Claude NO validó (no están en analyses)
                                validated_ns = {a.get('n') for a in analyses if a.get('n')}
                                missing_validations = []
                                for i in range(1, len(all_detections) + 1):
                                    if i not in validated_ns:
                                        if i-1 < len(all_detections):
                                            det = all_detections[i-1]
                                            area = det.get('area', 0)
                                            conf = det.get('confidence', 0.0)
                                            x, y, w, h = det.get('bbox', [0, 0, 0, 0])
                                            if i-1 < len(original_indices):
                                                original_idx = original_indices[i-1]
                                                missing_validations.append(f"#{i} (idx={original_idx}): área={area}px², bbox=({x},{y},{w},{h})")
                                
                                if missing_validations:
                                    logger.warning(f"\n⚠️  OBJETOS DETECTADOS POR SAM PERO NO VALIDADOS POR CLAUDE:")
                                    for mv in missing_validations:
                                        logger.warning(f"   {mv}")
                                
                                logger.info("="*60)
                                
                                # Process analyses using 'n' field to map back to correct detection/crop
                                # CRITICAL: Claude's 'n' (1-indexed) corresponds to index in all_detections (0-indexed)
                                # But crops are from original current_detections (0-indexed)
                                # Use original_indices mapping to get correct crop
                                for analysis in analyses:
                                    n = analysis.get('n', 0)
                                    
                                    # Validate n is in valid range (relative to all_detections)
                                    if n < 1 or n > len(all_detections):
                                        logger.warning(f"  ⚠️  Invalid object number n={n}, skipping")
                                        skipped_count += 1
                                        continue
                                    
                                    # Map from Claude's n (1-indexed) to filtered index (0-indexed)
                                    filtered_index = n - 1
                                    
                                    # CRITICAL: Validate filtered_index is in range
                                    if filtered_index >= len(all_detections):
                                        logger.warning(f"  ⚠️  #{n}: Filtered index {filtered_index} out of range (all_detections has {len(all_detections)} items), skipping")
                                        skipped_count += 1
                                        continue
                                    
                                    # CRITICAL FIX: ALWAYS use original_indices mapping
                                    # This is the ONLY reliable way to map from all_detections back to current_detections
                                    # The detection's own original_index may be corrupt after filtering/sorting
                                    if filtered_index >= len(original_indices):
                                        logger.error(f"  🚨 #{n}: Filtered index {filtered_index} out of range for original_indices (has {len(original_indices)} items). Skipping.")
                                        skipped_count += 1
                                        continue
                                    
                                    # Use original_indices mapping to get the correct index in current_detections
                                    current_detection_index = original_indices[filtered_index]
                                    
                                    # Validate the mapped index is in range
                                    if current_detection_index < 0 or current_detection_index >= len(current_detections):
                                        logger.error(f"  🚨 #{n}: Mapped original_index {current_detection_index} out of range (current_detections has {len(current_detections)} items). Skipping.")
                                        skipped_count += 1
                                        continue
                                    
                                    # Get detection from all_detections (we'll use its bbox for crop generation)
                                    detection = all_detections[filtered_index]
                                    
                                    # CRITICAL: Only check if Claude marked it as useful
                                    # NO OTHER FILTERS - Claude is the only judge
                                    useful = analysis.get('useful', 'no').lower() == 'yes'
                                    
                                    if not useful:
                                        skipped_count += 1
                                        reason = analysis.get('reason', 'not useful')
                                        logger.debug(f"  ⏭️  #{n}: Claude marcó como no útil - {reason}")
                                        continue
                                    
                                    # CRITICAL: Asegurar que el analysis tenga bbox válido (usar SAM como fallback si falta)
                                    claude_bbox = analysis.get('bbox')
                                    if not claude_bbox or len(claude_bbox) != 4:
                                        # Si Claude no proporcionó bbox, usar el de SAM
                                        sam_bbox = detection.get('bbox')
                                        if sam_bbox and len(sam_bbox) == 4:
                                            analysis['bbox'] = sam_bbox  # Agregar bbox de SAM al analysis
                                            logger.debug(f"     ℹ️  Usando bbox de SAM como fallback para n={n}")
                                        else:
                                            logger.warning(f"     ⚠️  No hay bbox válido (ni Claude ni SAM) para n={n}, saltando")
                                            continue
                                    
                                    # Objeto útil según Claude - agregar a lista (crop se generará después)
                                    # Si Claude dice que es útil, lo aceptamos sin más preguntas
                                    obj = {
                                        'n': n,  # Keep n for crop generation
                                        'analysis': analysis,  # Ahora tiene bbox garantizado
                                        'detection': detection,  # Bbox de SAM como respaldo
                                        'filtered_index': filtered_index
                                    }
                                    useful_objects.append(obj)
                                    logger.info(f"  ✅ #{n}: {analysis.get('name', 'Unknown object')} (Claude: útil)")
                                    logger.info(f"     Category: {analysis.get('category', 'other')}, Condition: {analysis.get('condition', 'unknown')}")
                                
                                # Process missing objects from Claude (generate crops from Claude's bbox)
                                # NOTE: Missing objects are detected by Claude but not by SAM
                                # CRITICAL: Validate bbox content BEFORE adding to avoid crops ininteligibles
                                if missing_objects:
                                    logger.info(f"\n🔍 Processing {len(missing_objects)} missing objects from Claude...")
                                    
                                    # Import validation function
                                    from image_quality import validate_bbox_content
                                    
                                    # Add missing objects to useful_objects list (they'll get crops generated)
                                    # Assign sequential n values starting from len(all_detections) + 1
                                    next_n = len(all_detections) + 1
                                    validated_missing_count = 0
                                    
                                    for missing in missing_objects:
                                        missing_bbox = missing.get('bbox')
                                        if not missing_bbox or len(missing_bbox) != 4:
                                            logger.warning(f"  ⚠️  Missing object '{missing.get('name', 'Unknown')}' has invalid bbox, skipping")
                                            continue
                                        
                                        # CRITICAL: Validate bbox content BEFORE adding
                                        # Bboxes aproximados de Claude pueden generar crops ininteligibles
                                        is_valid_bbox, bbox_metrics = validate_bbox_content(
                                            image=detection_frame,
                                            bbox=missing_bbox,
                                            min_content_ratio=0.3,  # Requiere 30% de contenido real
                                            min_sharpness=10.0  # Nitidez mínima
                                        )
                                        
                                        if not is_valid_bbox:
                                            logger.warning(f"  ⚠️  Rejecting missing object '{missing.get('name', 'Unknown')}': {bbox_metrics.get('error', 'bbox validation failed')}")
                                            logger.warning(f"     bbox={missing_bbox}, content_ratio={bbox_metrics.get('content_ratio', 0):.1%}, sharpness={bbox_metrics.get('sharpness', 0):.1f}")
                                            continue
                                        
                                        # Validate minimum area (reject objects too small)
                                        bbox_area = missing_bbox[2] * missing_bbox[3]
                                        if bbox_area < 500:  # Minimum 500px²
                                            logger.warning(f"  ⚠️  Rejecting missing object '{missing.get('name', 'Unknown')}': too small (area={bbox_area}px² < 500px²)")
                                            continue
                                        
                                        # Create a dummy detection object for crop generation
                                        missing_detection = {
                                            'bbox': missing_bbox,
                                            'confidence': 0.85,  # High confidence (Claude saw it)
                                            'area': bbox_area
                                        }
                                        
                                        # Create analysis-like object
                                        missing_analysis = {
                                            'n': next_n,
                                            'name': missing.get('name', 'Unknown object'),
                                            'category': missing.get('category', 'other'),
                                            'condition': 'unknown',
                                            'description': f"Objeto encontrado por Claude: {missing.get('name', 'Unknown')}",
                                            'estimated_value': None,
                                            'useful': 'yes',  # Claude detected it, so it's useful
                                            'bbox': missing_bbox  # Ensure bbox is in analysis for crop generation
                                        }
                                        
                                        # Add to useful_objects (will get crop generated)
                                        obj = {
                                            'n': next_n,
                                            'analysis': missing_analysis,
                                            'detection': missing_detection,
                                            'filtered_index': -1  # Not from SAM
                                        }
                                        useful_objects.append(obj)
                                        validated_missing_count += 1
                                        logger.info(f"  ✅ Added validated missing object n={next_n}: {missing.get('name', 'Unknown')} (bbox valid, area={bbox_area}px²)")
                                        next_n += 1
                                    
                                    if validated_missing_count < len(missing_objects):
                                        rejected_count = len(missing_objects) - validated_missing_count
                                        logger.warning(f"  ⚠️  Rejected {rejected_count} missing objects with invalid/approximate bboxes")
                                    
                                    # Add missing detections to all_detections for crop generation
                                    for missing in missing_objects:
                                        missing_bbox = missing.get('bbox')
                                        if missing_bbox and len(missing_bbox) == 4:
                                            missing_detection = {
                                                'bbox': missing_bbox,
                                                'confidence': 0.85,
                                                'area': missing_bbox[2] * missing_bbox[3]
                                            }
                                            all_detections.append(missing_detection)
                                
                                logger.info(f"\n📊 Results: {len(useful_objects)} useful, {skipped_count} skipped")
                                
                                # MÉTRICAS DE FILTRADO (FILOSOFÍA: "Detectar TODO, filtrar después")
                                total_sam_detections = len(current_detections) if current_detections else 0
                                total_claude_analyses = len(analyses) if analyses else 0
                                total_useful = len(useful_objects)
                                
                                logger.info(f"\n📊 MÉTRICAS DE FILTRADO:")
                                logger.info(f"="*60)
                                logger.info(f"SAM detecciones iniciales:    {total_sam_detections}")
                                logger.info(f"Enviadas a Claude:            {len(all_detections)} ({len(all_detections)/total_sam_detections*100:.1f}% de SAM)" if total_sam_detections > 0 else f"Enviadas a Claude:            {len(all_detections)}")
                                logger.info(f"Claude analizadas:            {total_claude_analyses}")
                                logger.info(f"Claude marcó útiles:          {total_useful} ({total_useful/total_claude_analyses*100:.1f}% de analizadas)" if total_claude_analyses > 0 else f"Claude marcó útiles:          {total_useful}")
                                logger.info(f"Objetos descartados por Claude: {total_claude_analyses - total_useful} ({100 - (total_useful/total_claude_analyses*100 if total_claude_analyses > 0 else 0):.1f}%)")
                                logger.info(f"="*60)
                                
                                # Log objetos descartados por Claude
                                if analyses:
                                    discarded = [a for a in analyses if a.get('useful', 'no').lower() != 'yes']
                                    if discarded:
                                        logger.info(f"\n🗑️  OBJETOS DESCARTADOS POR CLAUDE ({len(discarded)}):")
                                        for obj in discarded[:10]:  # Mostrar solo los primeros 10
                                            n = obj.get('n', 0)
                                            reason = obj.get('reason', 'No reason provided')
                                            logger.info(f"   ❌ n={n}: {reason}")
                                        if len(discarded) > 10:
                                            logger.info(f"   ... y {len(discarded) - 10} más")
                                
                                # ============================================================
                                # CRITICAL FIX: USAR BBOXES DE CLAUDE SIN VALIDAR CONTRA SAM
                                # ============================================================
                                # RAZÓN: Claude identifica objetos semánticos correctamente
                                #        SAM detecta regiones de píxeles (puede estar mal)
                                #        NO debemos rechazar bboxes de Claude solo porque
                                #        difieren de SAM - Claude tiene RAZÓN
                                
                                logger.info("\n" + "="*60)
                                logger.info("📦 GENERANDO LARGE_DETECTIONS DESDE BBOXES DE CLAUDE")
                                logger.info("="*60)
                                
                                # Rebuild large_detections from Claude's bboxes (only for useful objects)
                                # CRITICAL: NO VALIDAR contra SAM - Claude tiene razón
                                large_detections_from_claude = []
                                
                                for obj in useful_objects:
                                    analysis = obj['analysis']
                                    n = analysis.get('n', 0)
                                    
                                    # Get bbox from Claude's analysis
                                    claude_bbox = analysis.get('bbox')
                                    
                                    # FIX 3: Fallback a SAM si Claude no da bbox
                                    if not claude_bbox or len(claude_bbox) != 4:
                                        # FALLBACK: Intentar usar bbox de SAM como respaldo
                                        logger.warning(f"  ⚠️  Analysis n={n} missing Claude bbox, trying SAM fallback...")
                                        
                                        # Buscar detección correspondiente en all_detections
                                        if n >= 1 and n <= len(all_detections):
                                            sam_detection = all_detections[n-1]
                                            sam_bbox = sam_detection.get('bbox')
                                            
                                            if sam_bbox and len(sam_bbox) == 4:
                                                claude_bbox = sam_bbox  # Usar bbox de SAM
                                                logger.info(f"     ✅ Using SAM bbox as fallback: {sam_bbox}")
                                            else:
                                                logger.warning(f"     ❌ No SAM bbox available either, skipping object")
                                                continue
                                        else:
                                            logger.warning(f"     ❌ Cannot find corresponding SAM detection, skipping object")
                                            continue
                                    
                                    # Log bbox confidence if available
                                    bbox_confidence = analysis.get('bbox_confidence', 'high')
                                    if bbox_confidence == 'low':
                                        logger.info(f"     ℹ️  Low confidence bbox for n={n}, but accepting it")
                                    
                                    x, y, w, h = claude_bbox
                                    area = w * h
                                    
                                    # Validate area minimum
                                    if area < 500:  # pixels²
                                        logger.warning(f"  ⚠️  Bbox too small for n={n} (area={area}px²), skipping")
                                        continue
                                    
                                    # Validate bbox is within image bounds
                                    image_height, image_width = detection_frame.shape[:2]
                                    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                                        logger.warning(f"  ⚠️  Bbox outside image for n={n}, clipping")
                                        x = max(0, x)
                                        y = max(0, y)
                                        w = min(w, image_width - x)
                                        h = min(h, image_height - y)
                                        claude_bbox = [x, y, w, h]
                                        area = w * h  # Recalculate area after clipping
                                    
                                    # Create detection with Claude's bbox (NO validation against SAM)
                                    detection = {
                                        'bbox': claude_bbox,
                                        'confidence': 0.95,  # High confidence (Claude identified it)
                                        'area': area,
                                        'source': 'claude',
                                        'original_n': n,
                                        'name': analysis.get('name', 'Unknown')  # Add name for debugging
                                    }
                                    
                                    large_detections_from_claude.append(detection)
                                    logger.info(f"  ✅ n={n}: {analysis.get('name', 'Unknown')}, bbox={claude_bbox}, area={area}px²")
                                
                                logger.info(f"\n✅ Total detections from Claude: {len(large_detections_from_claude)}")
                                logger.info("="*60 + "\n")
                                
                                # Note: large_detections_from_claude contains Claude's bboxes for crop generation
                                # We use all_detections for mapping, but Claude's bboxes are more accurate
                                
                                # CRITICAL: Generate crops AFTER Claude analysis (only for useful objects)
                                # This eliminates mapping issues: n=1 → obj_001.jpg (always matches)
                                if useful_objects:
                                    logger.info(f"\n✂️  Generating crops for {len(useful_objects)} useful objects...")
                                    
                                    # CRITICAL FIX: Renumber useful objects consecutively (1, 2, 3, 4...)
                                    # Claude's original n may be 2, 3, 5, 9... (if some were filtered)
                                    # We need consecutive numbers: 1, 2, 3, 4... for obj_001.jpg, obj_002.jpg...
                                    analyses_for_crops = []
                                    for new_n, obj in enumerate(useful_objects, start=1):
                                        # Create a copy of the analysis with renumbered n
                                        analysis = obj['analysis'].copy()
                                        analysis['n'] = new_n  # Renumber: 1, 2, 3, 4...
                                        
                                        # CRITICAL: Ensure bbox is present before adding to analyses_for_crops
                                        # This ensures storage_v2.py always has a valid bbox to work with
                                        if not analysis.get('bbox') or len(analysis.get('bbox', [])) != 4:
                                            # Fallback to SAM detection bbox if Claude didn't provide one
                                            detection = obj.get('detection', {})
                                            sam_bbox = detection.get('bbox')
                                            if sam_bbox and len(sam_bbox) == 4:
                                                analysis['bbox'] = sam_bbox
                                                logger.debug(f"     ℹ️  Added SAM bbox to analysis for n={new_n}")
                                            else:
                                                logger.warning(f"     ⚠️  No bbox available for n={new_n}, skipping crop generation")
                                                continue  # Skip this object entirely
                                        
                                        analyses_for_crops.append(analysis)
                                        
                                        # Update obj with new_n for later use
                                        obj['new_n'] = new_n
                                    
                                    # Generate crops using renumbered n (consecutive: 1, 2, 3, 4...)
                                    # Pass useful_objects to get correct detections (not all_detections by index)
                                    n_to_crop = save_crops_for_useful_objects(
                                        image=detection_frame,
                                        analyses=analyses_for_crops,
                                        useful_objects=useful_objects,  # Pass useful_objects to get correct detections
                                        output_dir=str(storage.crops_dir),
                                        timestamp=scene_data['timestamp']
                                    )
                                    
                                    # Create final objects with crop paths
                                    final_objects = []
                                    for obj in useful_objects:
                                        new_n = obj.get('new_n')  # Use renumbered n (consecutive)
                                        if not new_n:
                                            logger.warning(f"  ⚠️  No new_n for object, skipping")
                                            continue
                                        
                                        analysis = obj['analysis']
                                        detection = obj['detection']
                                        crop_path = n_to_crop.get(new_n)  # Use new_n for crop lookup
                                        
                                        # CRITICAL: If no crop was generated, skip this object entirely
                                        # We cannot have objects in the database without crops (discrepancy between "eyes and brain")
                                        if not crop_path:
                                            logger.error(f"  ❌ No crop generated for n={new_n} (object: {analysis.get('name', 'Unknown')})")
                                            logger.error(f"     This object will NOT be saved to database (no crop = no object)")
                                            logger.error(f"     This prevents discrepancy between list and crops")
                                            continue  # Skip this object entirely - no crop = no object in database
                                        
                                        final_obj = {
                                            'id': f"obj_{timestamp}_{len(final_objects)+1:03d}",
                                            'timestamp': timestamp,
                                            'detection_number': new_n,  # Renumbered n (consecutive: 1, 2, 3...)
                                            'thumbnail': crop_path,  # Already relative path
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
                                        logger.info(f"  ✅ #{new_n}: {final_obj['name']} → {Path(crop_path).name}")
                                    
                                    # FIX AGRUPACIÓN: Fusionar objetos similares antes de guardar
                                    if final_objects:
                                        logger.info(f"\n🔗 Merging similar objects (to avoid duplicates)...")
                                        final_objects = _merge_similar_objects(final_objects)
                                        logger.info(f"   After merging: {len(final_objects)} objects (reduced from duplicates)")
                                    
                                    # Guardar solo objetos útiles en database
                                    if final_objects:
                                        database.extend(final_objects)
                                        with open(db_path, 'w') as f:
                                            json.dump(database, f, indent=2, ensure_ascii=False)
                                        logger.info(f"💾 Database saved ({len(database)} total objects)")
                                        logger.info(f"✅ Generated {len(final_objects)} crops with perfect mapping (n → obj_{n:03d}.jpg)")
                                        
                                        # MÉTRICA FINAL: Objetos guardados en web
                                        total_sam_detections = len(current_detections) if current_detections else 0
                                        logger.info(f"\n🌐 OBJETOS GUARDADOS EN WEB: {len(final_objects)}")
                                        logger.info(f"   (De {total_sam_detections} detecciones SAM → {len(final_objects)} objetos finales)")
                                    else:
                                        logger.warning("⚠️  No crops generated. Nothing saved to database.")
                                else:
                                    logger.warning("⚠️  No useful objects found. Nothing saved to database.")
                        
                        except Exception as e:
                            logger.error(f"❌ Scene analysis failed: {e}")
                            logger.exception("Full error traceback:")
                    elif not analyzer:
                        logger.info("\nℹ️  Claude analyzer not available. Objects saved but not analyzed.")
                        logger.info("   Set CLAUDE_API_KEY and restart to enable scene analysis.")
                    
                    # Testing mode: Images are kept during session for web display
                    # They will be deleted at next startup (in run_live_detection_with_claude.sh)
                    logger.info("")
                
            elif key == ord('l') or key == ord('L'):  # L - List objects
                if not current_detections:
                    logger.info("No objects detected. Press SPACE first.")
                else:
                    logger.info("\n" + "="*60)
                    logger.info("📋 DETECTED OBJECTS LIST")
                    logger.info("="*60)
                    
                    for i, detection in enumerate(current_detections[:20]):
                        obj_id = detection['id']
                        x, y, w, h = detection['bbox']
                        conf = detection['confidence']
                        
                        logger.info(f"\n#{i+1} - ID: {obj_id}")
                        logger.info(f"   BBox: ({x}, {y}, {w}, {h})")
                        logger.info(f"   Confidence: {conf:.3f}")
                        logger.info(f"   Area: {detection['area']} px²")
                        
                        if obj_id in object_analyses:
                            analysis = object_analyses[obj_id]
                            logger.info(f"   ✅ Name: {analysis.get('name', 'Unknown')}")
                            logger.info(f"   Category: {analysis.get('category', 'N/A')}")
                            logger.info(f"   Condition: {analysis.get('condition', 'N/A')}")
                            logger.info(f"   Description: {analysis.get('description', 'N/A')[:80]}...")
                            if analysis.get('estimated_value'):
                                logger.info(f"   Value: {analysis.get('estimated_value')}")
                        else:
                            logger.info(f"   ⏳ Not analyzed yet (press 'A' to analyze)")
                    
                    logger.info("\n" + "="*60 + "\n")
            
            elif key == ord('c') or key == ord('C'):  # C - Clear detections
                current_detections = []
                detection_frame = None
                object_analyses = {}
                # Clean temp files
                try:
                    for f in temp_dir.glob("*.jpg"):
                        f.unlink()
                except:
                    pass
                logger.info("Detections cleared")
                
            elif key == ord('q'):  # Q - Quit
                break
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except:
            pass
        
        logger.info("\n👋 Bye!")


if __name__ == "__main__":
    import sys
    try:
        # Print immediately to verify script is running
        print("🚀 Starting live_detection.py...", flush=True)
        sys.stdout.flush()
    live_detection()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

