"""
Hybrid SAM + Claude Detector - OPTIMIZED
Claude validation wrapper (no longer executes SAM).
Max 350 lines.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Claude validation wrapper for SAM detections.
    
    OPTIMIZED: No longer executes SAM (expects detections to be passed in).
    This eliminates double execution of SAM in the pipeline.
    """
    
    def __init__(self, sam_detector, claude_analyzer):
        """
        Initialize hybrid detector.
        
        Args:
            sam_detector: SAM3Detector instance (kept for compatibility, not used)
            claude_analyzer: ClaudeAnalyzer instance
        """
        self.sam = sam_detector  # Kept for compatibility, not actively used
        self.claude = claude_analyzer
        logger.info("✅ Hybrid detector initialized (Claude validation only)")
    
    def detect_with_validation(
        self,
        scene_path: str,
        sam_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Claude validation phase - validates SAM detections and finds missing objects.
        
        OPTIMIZED: No longer executes SAM (expects detections to be passed in).
        This eliminates double execution of SAM.
        
        Args:
            scene_path: Path to saved scene image for Claude
            sam_detections: Pre-computed SAM detections
            
        Returns:
            Dict with 'validated_objects' and 'missing_objects' from Claude
        """
        if not sam_detections:
            logger.warning("⚠️  No SAM detections provided")
            return {'validated_objects': [], 'missing_objects': []}
        
        logger.info(f"🔄 Claude validation + missing detection ({len(sam_detections)} SAM detections)...")
        
        try:
            # Get Claude's validation response
            claude_response = self.claude.analyze_scene_with_validation(
                scene_path=scene_path,
                detections=sam_detections,
                language="spanish"
            )
            
            validated_count = len(claude_response.get('validated_objects', []))
            missing_count = len(claude_response.get('missing_objects', []))
            
            logger.info(f"✅ Claude validation complete: {validated_count} validated, {missing_count} missing found")
            
            return claude_response
            
        except Exception as e:
            logger.error(f"❌ Claude validation failed: {e}")
            return {'validated_objects': [], 'missing_objects': []}

