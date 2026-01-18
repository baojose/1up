"""
1UP Server API - FastAPI endpoint for GPU processing
Receives 4K images, processes with SAM3 + Claude, returns results.
Max 350 lines.
"""
import base64
import logging
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yaml

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from detector import SAM3Detector
from analyzer import ClaudeAnalyzer
from storage_v2 import save_crops_for_useful_objects
from filters import filter_objects
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="1UP Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector and analyzer
detector = None
analyzer = None
config = None


class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    image_base64: str  # Base64 encoded image
    timestamp: str  # Scene timestamp
    config_override: Dict[str, Any] = {}  # Optional config overrides


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    success: bool
    detections: List[Dict[str, Any]]
    crops: Dict[str, str]  # n -> crop_path
    metadata: Dict[str, Any]
    error: str = None


def load_config():
    """Load server configuration."""
    config_path = Path(__file__).parent / "config_server.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config for RunPod GPU
        return {
            'sam3': {
                'device': 'cuda',
                'filtering': {'enabled': False},
                'enhance_image': True
            },
            'claude': {
                'api_key_env': 'CLAUDE_API_KEY',
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 16000
            },
            'storage': {
                'crops_dir': 'images/crops',
                'raw_dir': 'images/raw'
            }
        }


def initialize_models():
    """Initialize SAM3 detector and Claude analyzer."""
    global detector, analyzer, config
    
    config = load_config()
    logger.info("Loading configuration...")
    
    # Initialize SAM3 detector (GPU/CUDA)
    try:
        device = config['sam3'].get('device', 'cuda')
        logger.info(f"Initializing SAM3 detector on {device}...")
        detector = SAM3Detector(device=device)
        logger.info("✅ SAM3 detector initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SAM3: {e}")
        raise
    
    # Initialize Claude analyzer
    api_key = os.environ.get(config['claude']['api_key_env'])
    if not api_key:
        logger.warning("⚠️  Claude API key not found, analysis disabled")
        analyzer = None
    else:
        try:
            analyzer = ClaudeAnalyzer(
                api_key=api_key,
                model=config['claude'].get('model', 'claude-sonnet-4-20250514'),
                max_tokens=config['claude'].get('max_tokens', 16000)
            )
            logger.info("✅ Claude analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("🚀 Starting 1UP Detection API...")
    initialize_models()
    logger.info("✅ API ready")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "analyzer_ready": analyzer is not None
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Process image with SAM3 + Claude, return detections and crops.
    """
    global detector, analyzer, config
    
    if detector is None:
        raise HTTPException(status_code=500, detail="SAM3 detector not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        logger.info(f"📸 Processing image: {image.shape[1]}x{image.shape[0]}")
        
        # Run SAM3 detection
        filter_config = config['sam3'].get('filtering', {})
        enhance_image = config['sam3'].get('enhance_image', False)
        
        detections = detector.detect_objects(
            image,
            apply_filtering=filter_config.get('enabled', False),
            enhance_image=enhance_image,
            text_prompt=None
        )
        
        logger.info(f"🔍 SAM3 detected {len(detections)} objects")
        
        # Claude analysis (if available)
        analyses = []
        useful_objects = []
        
        if analyzer and detections:
            logger.info(f"🤖 Analyzing with Claude...")
            
            # Save temporary scene image for Claude
            temp_dir = Path(config['storage']['raw_dir'])
            temp_dir.mkdir(parents=True, exist_ok=True)
            scene_path = temp_dir / f"temp_scene_{request.timestamp}.jpg"
            cv2.imwrite(str(scene_path), image)
            
            # Analyze with Claude
            analyses = analyzer.analyze_scene_with_bboxes(
                str(scene_path),
                detections,
                language="spanish"
            )
            
            # Filter useful objects
            useful_analyses = [a for a in analyses if a.get('useful') == 'yes']
            logger.info(f"✅ Claude found {len(useful_analyses)} useful objects")
            
            # Apply post-filters
            useful_analyses = filter_objects(useful_analyses, image.shape)
            
            # Build useful_objects structure (needed for save_crops_for_useful_objects)
            # Structure: [{'detection': {...}, 'analysis': {...}, 'n': ...}]
            useful_objects = []
            for i, analysis in enumerate(useful_analyses):
                n = analysis.get('n', i + 1)
                # Find corresponding detection by n (1-indexed)
                det_idx = n - 1
                if 0 <= det_idx < len(detections):
                    detection = detections[det_idx]
                else:
                    # Fallback: use first detection if n is out of range
                    logger.warning(f"⚠️  n={n} out of range, using first detection")
                    detection = detections[0] if detections else {}
                
                useful_objects.append({
                    'detection': detection,
                    'analysis': analysis,
                    'n': n,
                    'filtered_index': i
                })
            
            # Renumber analyses consecutively (1, 2, 3, 4...)
            analyses_for_crops = []
            renumbered_analyses = []  # Store renumbered analyses for response
            for new_n, obj in enumerate(useful_objects, start=1):
                analysis = obj['analysis'].copy()
                analysis['n'] = new_n  # Renumber consecutively
                
                # Ensure bbox is present
                if not analysis.get('bbox') or len(analysis.get('bbox', [])) != 4:
                    # Fallback to SAM detection bbox
                    detection = obj.get('detection', {})
                    sam_bbox = detection.get('bbox')
                    if sam_bbox and len(sam_bbox) == 4:
                        analysis['bbox'] = sam_bbox
                    else:
                        logger.warning(f"⚠️  No bbox available for n={new_n}, skipping")
                        continue
                
                analyses_for_crops.append(analysis)
                renumbered_analyses.append(analysis)  # Store renumbered version
                obj['new_n'] = new_n
            
            # Generate crops for useful objects
            crops_dir = config['storage']['crops_dir']
            n_to_crop = save_crops_for_useful_objects(
                image,
                analyses_for_crops,  # Renumbered analyses
                useful_objects,  # Full structure with detection and analysis
                crops_dir,
                request.timestamp
            )
            
            # Convert crop paths to relative strings
            crops = {str(n): str(path) for n, path in n_to_crop.items()}
        else:
            crops = {}
            logger.warning("⚠️  Claude analyzer not available, skipping analysis")
        
        # Prepare response (use renumbered analyses if available)
        if analyzer and useful_objects and renumbered_analyses:
            response_detections = renumbered_analyses  # Use renumbered analyses (n=1,2,3...)
        else:
            response_detections = detections
        
        # Convert numpy arrays to Python lists for JSON serialization
        def convert_numpy(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        response_detections = convert_numpy(response_detections)
        
        # Prepare response
        response = DetectionResponse(
            success=True,
            detections=response_detections,
            crops=crops,
            metadata={
                'timestamp': request.timestamp,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'total_detections': len(detections),
                'useful_objects': len(useful_objects) if analyzer else len(detections)
            }
        )
        
        logger.info(f"✅ Detection complete: {len(response.detections)} objects, {len(crops)} crops")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}", exc_info=True)
        return DetectionResponse(
            success=False,
            detections=[],
            crops={},
            metadata={},
            error=str(e)
        )


@app.get("/crops/list")
async def list_crops():
    """List all available crop images."""
    crops_dir = Path(config['storage']['crops_dir'])
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    crops = []
    for crop_file in sorted(crops_dir.rglob("*.jpg")):
        relative_path = crop_file.relative_to(crops_dir.parent)
        crops.append({
            'path': str(relative_path),
            'name': crop_file.name,
            'timestamp': crop_file.parent.name
        })
    
    return {'crops': crops, 'count': len(crops)}


@app.get("/crops/{path:path}")
async def get_crop(path: str):
    """Serve crop image file."""
    crops_dir = Path(config['storage']['crops_dir'])
    crop_path = crops_dir / path
    
    if not crop_path.exists() or not crop_path.is_file():
        raise HTTPException(status_code=404, detail="Crop not found")
    
    return FileResponse(str(crop_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
