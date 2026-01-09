"""
Claude Vision Analyzer
Analyzes object images using Claude Sonnet 4 vision.
Max 350 lines.
"""
import anthropic
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ClaudeAnalyzer:
    """Analyzes objects using Claude Sonnet 4 vision."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1000,
        temperature: float = 0
    ):
        """
        Initialize Claude analyzer.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 = deterministic)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"✅ Claude analyzer initialized ({model})")
    
    def analyze_scene_with_bboxes(
        self,
        scene_path: str,
        detections: List[Dict[str, Any]],
        language: str = "spanish"
    ) -> List[Dict[str, Any]]:
        """
        Analyze objects detected in complete scene.
        
        Sends ONLY the scene image + list of bounding boxes in text.
        Claude analyzes each region directly in the image.
        
        Args:
            scene_path: Path to complete scene image
            detections: List of SAM detections with bbox, area, confidence
            language: Response language
            
        Returns:
            List of analysis results (one per detection, in order)
        """
        logger.info(f"🤖 Analyzing {len(detections)} objects in scene (1 image + bboxes)...")
        
        # Encode ONLY the scene image (1 image total)
        scene_data = self._encode_image(scene_path)
        
        # Build bbox descriptions for prompt
        bbox_descriptions = []
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            area = det.get('area', w * h)
            conf = det.get('confidence', 0.0)
            bbox_descriptions.append(
                f"Objeto {i+1}: bbox [x={x}, y={y}, ancho={w}, alto={h}], "
                f"área={area}px², confianza={conf:.2f}"
            )
        
        # Create prompt with bbox list
        prompt = self._create_bbox_analysis_prompt(bbox_descriptions, len(detections), language)
        
        # Build API message (1 image + text)
        content = [
            {
                "type": "image",
                "source": scene_data
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        try:
            # Single API call with 1 image
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # Parse response
            response_text = message.content[0].text.strip()
            
            # Check if response was truncated
            if hasattr(message, 'stop_reason') and message.stop_reason == 'max_tokens':
                logger.warning(f"⚠️  Response truncated (max_tokens reached). Consider increasing max_tokens or reducing object count.")
                logger.warning(f"   Response length: {len(response_text)} chars")
            
            results = self._parse_batch_response(response_text, len(detections))
            
            logger.info(f"✅ Claude analyzed {len(results)} objects (1 API call, 1 image)")
            return results
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return self._create_fallback_batch(len(detections))
        except Exception as e:
            logger.exception("Unexpected error in scene analysis")
            return self._create_fallback_batch(len(detections))
    
    def _encode_image(self, image_path: str) -> Dict[str, str]:
        """Encode image to base64 with proper media type."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(suffix, 'image/jpeg')
        
        return {
            "type": "base64",
            "media_type": media_type,
            "data": image_data,
        }
    
    def _create_bbox_analysis_prompt(
        self,
        bbox_descriptions: List[str],
        num_objects: int,
        language: str
    ) -> str:
        """Create prompt for scene analysis with bounding boxes."""
        bboxes_text = "\n".join(bbox_descriptions)
        
        if language == "spanish":
            return f"""Analiza esta escena de un punto limpio (centro de reciclaje).

He detectado automáticamente {num_objects} objetos. Para CADA objeto, mira la región indicada por sus coordenadas en la imagen:

{bboxes_text}

Responde EXCLUSIVAMENTE con array JSON (sin markdown, sin ```json):

[
  {{"n":1, "useful":"yes", "name":"nombre específico del objeto", "category":"categoría", "condition":"excellent/good/fair/poor", "description":"descripción detallada en español (2-3 frases)", "estimated_value":"rango opcional en euros"}},
  {{"n":2, "useful":"no", "reason":"por qué no es útil"}},
  ...
]

CRITERIOS ESTRICTOS:

"useful": "yes" SOLO SI:
✅ Objeto COMPLETO y funcional (no fragmento)
✅ Tiene identidad clara y específica (NO "superficie blanca", "cosa gris", "objeto rectangular")
✅ Alguien querría llevárselo para reutilizar
✅ NO es fondo/mobiliario del punto limpio (mesa, pared, suelo, sombra)

"useful": "no" SI:
❌ Basura, papel arrugado, envoltorio
❌ Fragmento incompleto (hoja suelta, cable sin dispositivo, esquina de objeto)
❌ Fondo/mobiliario del punto limpio (mesa, pared, suelo, sombra)
❌ Partes de planta (hojas sueltas)
❌ Muy deteriorado sin posibilidad de uso
❌ Nombre genérico ("superficie", "esquina", "borde", "fragmento")

CATEGORÍAS VÁLIDAS:
furniture, electronics, books, tools, kitchenware, sports, toys, decoration, clothing, containers, other

CONDICIÓN:
- excellent: Como nuevo, sin defectos
- good: Buen estado, uso normal
- fair: Aceptable, signos de desgaste
- poor: Deteriorado, necesita reparación

IMPORTANTE:
- Responde para TODOS los {num_objects} objetos (números 1 a {num_objects})
- Sé ESTRICTO: si dudas, marca "useful": "no"
- Si useful="yes" incluye todos los campos (name, category, condition, description, estimated_value)
- Si useful="no" solo incluye n y reason
- NO incluyas markdown (```json)
- NO incluyas texto adicional fuera del array JSON
- El array debe tener exactamente {num_objects} elementos"""
        else:  # English
            return f"""Analyze this recycling center scene.

I've automatically detected {num_objects} objects. For EACH object, look at the region indicated by its coordinates in the image:

{bboxes_text}

Respond EXCLUSIVELY with JSON array (no markdown, no ```json):

[
  {{"n":1, "useful":"yes", "name":"specific object name", "category":"category", "condition":"excellent/good/fair/poor", "description":"detailed description (2-3 sentences)", "estimated_value":"optional price range"}},
  {{"n":2, "useful":"no", "reason":"why not useful"}},
  ...
]

STRICT CRITERIA:

"useful": "yes" ONLY IF:
✅ Complete and functional object (not fragment)
✅ Clear, specific identity (NOT "white surface", "gray thing", "rectangular object")
✅ Someone would want to take it for reuse
✅ NOT background/furniture of recycling center (table, wall, floor, shadow)

"useful": "no" IF:
❌ Trash, crumpled paper, wrapper
❌ Incomplete fragment (loose sheet, cable without device, object corner)
❌ Background/furniture of recycling center (table, wall, floor, shadow)
❌ Plant parts (loose leaves)
❌ Too deteriorated without possibility of use
❌ Generic name ("surface", "corner", "edge", "fragment")

VALID CATEGORIES:
furniture, electronics, books, tools, kitchenware, sports, toys, decoration, clothing, containers, other

CONDITION:
- excellent: Like new, no defects
- good: Good condition, normal wear
- fair: Acceptable, signs of wear
- poor: Deteriorated, needs repair

IMPORTANT:
- Respond for ALL {num_objects} objects (numbers 1 to {num_objects})
- Be STRICT: if in doubt, mark "useful": "no"
- If useful="yes" include all fields (name, category, condition, description, estimated_value)
- If useful="no" only include n and reason
- NO markdown (```json)
- NO additional text outside JSON array
- Array must have exactly {num_objects} elements"""
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse Claude's batch JSON response."""
        # Remove markdown code blocks if present
        text = response_text.strip()
        
        # Try to extract JSON from markdown code blocks
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    text = part[4:].strip()
                    break
                elif part.startswith('['):
                    text = part
                    break
        
        # Find JSON array (may have text before/after)
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error("No JSON array found in response")
            logger.debug(f"Response preview (first 1000 chars): {text[:1000]}")
            logger.debug(f"Response length: {len(text)} chars")
            return self._create_fallback_batch(expected_count)
        
        json_text = text[start_idx:end_idx + 1]
        
        try:
            results = json.loads(json_text)
            
            if not isinstance(results, list):
                logger.error(f"Expected list, got {type(results)}")
                return self._create_fallback_batch(expected_count)
            
            # Validate and normalize results
            normalized = []
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    logger.warning(f"Result {i} is not a dict, skipping")
                    continue
                
                # Ensure 'n' field matches index
                if 'n' not in result:
                    result['n'] = i + 1
                
                # Normalize useful field
                if 'useful' not in result:
                    result['useful'] = 'no'
                    result['reason'] = 'Missing useful field'
                
                normalized.append(result)
            
            if len(normalized) != expected_count:
                logger.warning(
                    f"Expected {expected_count} results, got {len(normalized)}. "
                    "Padding with fallback entries."
                )
                while len(normalized) < expected_count:
                    normalized.append({
                        'n': len(normalized) + 1,
                        'useful': 'no',
                        'reason': 'Missing from Claude response'
                    })
            
            return normalized[:expected_count]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Error at position {e.pos}: {e.msg}")
            logger.debug(f"JSON text around error (chars {max(0, e.pos-100)}-{min(len(json_text), e.pos+100)}): {json_text[max(0, e.pos-100):min(len(json_text), e.pos+100)]}")
            logger.debug(f"Full JSON text length: {len(json_text)} chars")
            # Try to save response for debugging
            try:
                import os
                debug_file = "claude_response_debug.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Original response:\n{response_text}\n\n")
                    f.write(f"Extracted JSON:\n{json_text}\n")
                logger.debug(f"Saved debug response to {debug_file}")
            except:
                pass
            return self._create_fallback_batch(expected_count)
    
    def _create_fallback_batch(self, count: int) -> List[Dict[str, Any]]:
        """Create fallback batch results if Claude fails."""
        return [
            {
                'n': i + 1,
                'useful': 'no',
                'reason': 'Analysis failed. Please review manually.',
                'error': True
            }
            for i in range(count)
        ]
    
    def analyze_scene_with_validation(
        self,
        scene_path: str,
        detections: List[Dict[str, Any]],
        language: str = "spanish"
    ) -> Dict[str, Any]:
        """
        Analyze scene with validation and missing object detection.
        
        Claude validates detected objects AND suggests missing objects
        that SAM didn't detect (e.g., white objects on light backgrounds).
        
        Args:
            scene_path: Path to complete scene image
            detections: List of SAM detections with bbox, area, confidence
            language: Response language
            
        Returns:
            Dict with:
            - validated_objects: List of validated analyses (same format as analyze_scene_with_bboxes)
            - missing_objects: List of missing objects with approximate bboxes
        """
        logger.info(f"🔍 Validating {len(detections)} objects + searching for missing objects...")
        
        # Encode scene image
        scene_data = self._encode_image(scene_path)
        
        # Get image dimensions for relative size calculation
        import cv2
        img = cv2.imread(scene_path)
        img_height, img_width = img.shape[:2] if img is not None else (960, 1280)
        total_pixels = img_width * img_height
        
        # Build bbox descriptions with detailed info including relative size
        bbox_descriptions = []
        logger.info(f"📤 Preparando {len(detections)} objetos para Claude:")
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            area = det.get('area', w * h)
            conf = det.get('confidence', 0.0)
            area_percent = (area / total_pixels) * 100 if total_pixels > 0 else 0
            bbox_descriptions.append(
                f"Objeto {i+1}: bbox [x={x}, y={y}, ancho={w}, alto={h}], "
                f"área={area}px² ({area_percent:.1f}% de la imagen), confianza={conf:.2f}"
            )
            logger.info(f"   Objeto {i+1}: bbox=({x},{y},{w},{h}), área={area}px² ({area_percent:.1f}%), conf={conf:.3f}")
        
        logger.info("")
        
        # Create validation prompt
        prompt = self._create_validation_prompt(bbox_descriptions, len(detections), language)
        
        # Build API message
        content = [
            {
                "type": "image",
                "source": scene_data
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            response_text = message.content[0].text.strip()
            result = self._parse_validation_response(response_text, len(detections))
            
            validated_count = len(result.get('validated_objects', []))
            missing_count = len(result.get('missing_objects', []))
            
            logger.info(f"✅ Validation complete: {validated_count} validated, {missing_count} missing found")
            
            # CRITICAL: Warn if Claude didn't validate all objects
            if validated_count < len(detections):
                logger.warning(f"⚠️  Claude only validated {validated_count}/{len(detections)} objects!")
                logger.warning(f"   Expected {len(detections)} validated objects, got {validated_count}")
                logger.warning(f"   Missing {len(detections) - validated_count} validations")
            
            return result
            
        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            return {
                'validated_objects': self._create_fallback_batch(len(detections)),
                'missing_objects': []
            }
    
    def _create_validation_prompt(
        self,
        bbox_descriptions: List[str],
        num_objects: int,
        language: str
    ) -> str:
        """Create prompt for validation + analysis + missing object detection (ALL IN ONE)."""
        bboxes_text = "\n".join(bbox_descriptions)
        
        if language == "spanish":
            return f"""Analiza esta escena de un punto limpio. El sistema detecta TODO (sombras, fragmentos, fondo). Tu trabajo: identificar TODOS los objetos útiles completos.

He detectado {num_objects} regiones con estas coordenadas:

{bboxes_text}

TAREAS:
1. VALIDAR cada región: mira el bbox [x, y, ancho, alto], identifica el objeto usando el tamaño como contexto:
   - Grandes (5-15%): muebles, electrodomésticos
   - Medianos (1-5%): libros, bolsos, ropa, contenedores
   - Pequeños (0.1-1%): frascos, botellas, juguetes, objetos decorativos
   - <0.1%: fragmentos (rechazar)
   
   Si useful="yes": name, category, condition, description, estimated_value, bbox [x,y,w,h], bbox_confidence
   Si useful="no": reason

2. BUSCAR objetos faltantes/superpuestos no detectados.

Responde SOLO JSON (sin markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "Pesa negra", "category": "sports", "condition": "good", "description": "Pesa de entrenamiento negra redonda", "estimated_value": "20-40€", "bbox": [x,y,w,h], "bbox_confidence": "high"}},
    {{"n": 2, "useful": "no", "reason": "Fragmento de fondo"}},
    ...
  ],
  "missing_objects": [
    {{"name": "frasco", "bbox": [x,y,w,h], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIOS:
✅ Útil: objeto completo funcional, grupos similares ("Especiero con 6 frascos" NO 6 separados)
✅ Útil: contenedores, botellas, frascos, objetos decorativos, plantas, utensilios de cocina
❌ No útil: fragmentos, sombras, fondo (pared/suelo/mesa), duplicados, partes del cuerpo

AGRUPACIÓN CRÍTICA (Muy importante):
- Si hay 3+ objetos similares superpuestos/cercanos → agrupa en 1 objeto grupal con cantidad
- Ejemplo: Si ves Objetos 3, 4, 5, 6, 7, 8, 9 que son 7 frascos de especias juntos:
  * Responde para TODOS: Objeto 3, Objeto 4, Objeto 5, Objeto 6, Objeto 7, Objeto 8, Objeto 9
  * Para el PRIMER objeto del grupo (ej: Objeto 3): useful="yes", name="Especiero con 7 frascos de especias", bbox=[bbox del grupo completo]
  * Para los demás del grupo (ej: Objetos 4-9): useful="no", reason="Agrupado en Especiero con 7 frascos"

BBOX PARA OBJETOS AGRUPADOS (CRÍTICO):
- Si agrupas objetos: el bbox del objeto agrupado debe cubrir TODO el grupo completo
- Ejemplo: Si agrupas Objetos 3, 4, 5, 6, 7, 8, 9 en "Especiero con 7 frascos":
  * Calcula el bbox que contiene TODOS los bboxes de esos objetos
  * min_x = mínimo x de todos los bboxes
  * min_y = mínimo y de todos los bboxes  
  * max_x = máximo (x+w) de todos los bboxes
  * max_y = máximo (y+h) de todos los bboxes
  * bbox del grupo = [min_x, min_y, max_x-min_x, max_y-min_y]
- El bbox del grupo debe ser más grande que cualquiera de los bboxes individuales

BBOX: [x,y,width,height] SIEMPRE presente si useful="yes" (OBLIGATORIO, puede ser aproximado pero debe existir).
⚠️ CRÍTICO: Si useful="yes", DEBES proporcionar bbox. Si agrupaste objetos, usa el bbox del grupo completo.

IMPORTANTE: Sé INCLUSIVO. Si un objeto es claramente visible y útil, márcalo como useful="yes", incluso si es pequeño o parcialmente oculto.

RESPONDE TODOS los {num_objects} objetos. Sé INCLUSIVO pero INTELIGENTE."""
        else:  # English
            return f"""Analyze this recycling center scene. System detects EVERYTHING (shadows, fragments, background). Your job: identify ONLY complete useful objects.

I've detected {num_objects} regions with these coordinates:

{bboxes_text}

TASKS:
1. VALIDATE each region: look at bbox [x, y, width, height], identify object using size as context:
   - Large (5-15%): furniture, appliances
   - Medium (1-5%): books, bags, clothing
   - Small (0.1-1%): jars, bottles, toys
   - <0.1%: fragments (reject)
   
   If useful="yes": name, category, condition, description, estimated_value, bbox [x,y,w,h], bbox_confidence
   If useful="no": reason

2. FIND missing/overlapping objects not detected.

Respond ONLY JSON (no markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "Black weight", "category": "sports", "condition": "good", "description": "Black round training weight", "estimated_value": "20-40€", "bbox": [x,y,w,h], "bbox_confidence": "high"}},
    {{"n": 2, "useful": "no", "reason": "Background fragment"}},
    ...
  ],
  "missing_objects": [
    {{"name": "medicine bottle", "bbox": [x,y,w,h], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIA:
✅ Useful: complete functional object, similar groups ("Spice rack with 6 jars" NOT 6 separate)
❌ Not useful: fragments, shadows, background (wall/floor/table), duplicates, body parts

GROUPING: If 3+ similar overlapping objects → 1 group object with quantity.

BBOX: [x,y,width,height] always present if useful="yes" (may be approximate).

RESPOND to ALL {num_objects} objects. Mark "useful":"no" for 60-70% (system noise). Be STRICT."""
    
    def _parse_validation_response(
        self,
        response_text: str,
        expected_count: int
    ) -> Dict[str, Any]:
        """Parse Claude's validation response with validated_objects and missing_objects."""
        text = response_text.strip()
        
        # Remove markdown
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    text = part[4:].strip()
                    break
                elif part.startswith('{'):
                    text = part
                    break
        
        # Find JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error("❌ No JSON object found in validation response")
            logger.debug(f"Response text (first 500 chars): {text[:500]}")
            return {
                'validated_objects': [],
                'missing_objects': []
            }
        
        json_text = text[start_idx:end_idx + 1]
        
        try:
            result = json.loads(json_text)
            
            # Validate structure
            validated = result.get('validated_objects', [])
            missing = result.get('missing_objects', [])
            
            # CRITICAL: Do NOT pad missing objects - only use what Claude validated
            # If Claude didn't validate an object, it means it's likely not useful
            if len(validated) != expected_count:
                logger.warning(f"⚠️  Expected {expected_count} validated objects, got {len(validated)}")
                logger.warning(f"   Claude did not validate {expected_count - len(validated)} objects")
                logger.warning(f"   These objects will be SKIPPED (not saved)")
                # DO NOT pad - only use what Claude actually validated
            
            # Validate missing_objects format
            validated_missing = []
            for missing in missing:
                if isinstance(missing, dict) and 'bbox' in missing and 'name' in missing:
                    bbox = missing['bbox']
                    if isinstance(bbox, list) and len(bbox) == 4:
                        validated_missing.append(missing)
                    else:
                        logger.warning(f"Invalid bbox format in missing object: {missing.get('name')}")
                else:
                    logger.warning(f"Invalid missing object format: {missing}")
            
            return {
                'validated_objects': validated,
                'missing_objects': validated_missing
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse validation JSON: {e}")
            logger.error(f"   JSON text (first 1000 chars): {json_text[:1000]}")
            logger.error(f"   This may indicate Claude's response format is incorrect")
            return {
                'validated_objects': [],
                'missing_objects': []
            }

