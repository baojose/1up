"""
1UP Web Application - E-commerce Display
Serves detected objects in an e-commerce format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

from flask import Flask, render_template, send_from_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Paths
BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "database" / "objects.json"
IMAGES_DIR = BASE_DIR / "images"


def load_objects() -> List[Dict[str, Any]]:
    """Load objects from database JSON file."""
    try:
        if not DATABASE_PATH.exists():
            logger.warning(f"Database not found: {DATABASE_PATH}")
            return []
        
        with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
            objects = json.load(f)
        
        logger.info(f"Loaded {len(objects)} objects from database")
        return objects
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in database: {e}")
        return []
    except Exception as e:
        logger.exception(f"Error loading database: {e}")
        return []


def format_category(category: str) -> str:
    """Format category name for display."""
    category_map = {
        'electronics': 'Electrónica',
        'furniture': 'Muebles',
        'clothing': 'Ropa',
        'kitchenware': 'Cocina',
        'decoration': 'Decoración',
        'containers': 'Contenedores',
        'other': 'Otros'
    }
    return category_map.get(category.lower(), category.capitalize())


def format_condition(condition: str) -> str:
    """Format condition for display."""
    condition_map = {
        'excellent': 'Excelente',
        'good': 'Bueno',
        'fair': 'Regular',
        'poor': 'Pobre'
    }
    return condition_map.get(condition.lower(), condition.capitalize())


@app.route('/')
def index():
    """Main page displaying all objects."""
    objects = load_objects()
    
    # Sort by timestamp (newest first)
    objects.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Format objects for display
    formatted_objects = []
    for obj in objects:
        # Fix thumbnail path: remove 'images/' prefix if present
        thumbnail = obj.get('thumbnail', '')
        if thumbnail.startswith('images/'):
            thumbnail = thumbnail[7:]  # Remove 'images/' prefix
        
        formatted_obj = {
            'id': obj.get('id', ''),
            'name': obj.get('name', 'Objeto sin nombre'),
            'description': obj.get('description', 'Sin descripción'),
            'category': format_category(obj.get('category', 'other')),
            'condition': format_condition(obj.get('condition', 'good')),
            'estimated_value': obj.get('estimated_value', 'N/A'),
            'thumbnail': thumbnail,
            'timestamp': obj.get('timestamp', ''),
            'confidence': obj.get('confidence', 0)
        }
        formatted_objects.append(formatted_obj)
    
    logger.info(f"Rendering {len(formatted_objects)} objects")
    return render_template('index.html', objects=formatted_objects)


@app.route('/images/<path:filename>')
def serve_image(filename: str):
    """Serve images from the images directory."""
    try:
        # filename should already be relative to images/ (e.g., "crops/.../obj_072.jpg")
        # Find the image file
        image_path = IMAGES_DIR / filename
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return "Image not found", 404
        
        # Get directory and filename
        directory = image_path.parent
        file_name = image_path.name
        
        return send_from_directory(str(directory), file_name)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return "Error serving image", 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'objects_count': len(load_objects())}


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("1UP Web Application - E-commerce Display")
    logger.info("=" * 60)
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info(f"Images: {IMAGES_DIR}")
    
    objects_count = len(load_objects())
    logger.info(f"Loaded {objects_count} objects")
    logger.info("")
    logger.info("Starting server on http://localhost:5001")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)

