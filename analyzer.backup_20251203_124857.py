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
            return f"""Analiza esta escena de un punto limpio (centro de reciclaje).

He detectado automáticamente {num_objects} objetos con estas coordenadas:

{bboxes_text}

📋 **INSTRUCCIONES CRÍTICAS PARA IDENTIFICACIÓN:**

1. **MIRA EXACTAMENTE LA REGIÓN INDICADA POR CADA BBOX:**
   - Cada bbox [x, y, ancho, alto] define un rectángulo en la imagen
   - x, y = esquina superior izquierda del rectángulo
   - ancho, alto = dimensiones del rectángulo
   - **MIRA SOLO DENTRO DE ESE RECTÁNGULO** para identificar el objeto
   - NO confundas objetos adyacentes con el objeto dentro del bbox

2. **IDENTIFICA OBJETOS EN CADA BBOX (CRÍTICO PARA OBJETOS SUPERPUESTOS):**
   - Si el bbox contiene UN objeto completo, identifícalo correctamente
   - Si el bbox contiene MÚLTIPLES objetos superpuestos o adyacentes:
     * Identifica TODOS los objetos visibles en ese bbox
     * Proporciona un bbox separado para CADA objeto identificado
     * Si hay objetos superpuestos (uno encima del otro), identifica ambos
     * Usa el campo "bbox" para proporcionar coordenadas precisas de cada objeto
   - Si el bbox contiene solo una parte de un objeto, identifica qué parte es
   - IMPORTANTE: Si detectas múltiples objetos en una región, crea múltiples entradas en validated_objects (una por objeto)

3. **USA EL TAMAÑO COMO CONTEXTO** (CRÍTICO):
El tamaño (área y % de la imagen) te ayuda a identificar QUÉ es cada objeto:
- Objetos grandes (5-15% de imagen): muebles (sillas, mesas), electrodomésticos grandes, contenedores grandes
- Objetos medianos (1-5% de imagen): laptops, tablets, libros, bolsos, zapatos, ropa doblada, herramientas
- Objetos pequeños (0.1-1% de imagen): frascos, botellas, juguetes pequeños, accesorios
- Objetos muy pequeños (<0.1%): fragmentos, basura, partes sueltas

**EJEMPLOS DE IDENTIFICACIÓN POR TAMAÑO:**
- Objeto oscuro redondo de ~2-4% de imagen → pesa de entrenamiento o objeto redondo pesado
- Objeto grande (8-12% de imagen) con patas → silla o mueble
- Objetos medianos (1-3% de imagen) de tela/textil → ropa (camisetas, pantalones)
- Objeto grande rectangular (5-10% de imagen) con asas → contenedor/cesta de ropa
- Objeto pequeño (0.5-1% de imagen) con asa → bolso de mujer
- Objeto mediano rectangular (2-4% de imagen) con texto/portada visible → libro, cómic, revista
- Objeto muy pequeño (0.1-0.5% de imagen) con ruedas → cochecito de juguete

**USA EL TAMAÑO PARA:**
1. Identificar correctamente el objeto (pesa vs. pelota, silla vs. taburete, cómic vs. revista)
2. Distinguir objetos completos de fragmentos (objeto completo vs. esquina de objeto)
3. Validar si el bbox corresponde al objeto correcto (un bolso pequeño no puede tener un bbox gigante)
4. Identificar objetos pequeños pero completos (cochecitos de juguete, frascos, botellas)

**EJEMPLOS ESPECÍFICOS DE IDENTIFICACIÓN:**
- Si en un bbox de ~3% de imagen ves texto "¡PUTA GUERRA!" y portada de cómic → es un "Cómic/Novela gráfica '¡PUTA GUERRA!'"
- Si en un bbox de ~0.2% de imagen ves un coche pequeño con ruedas → es un "Cochecito de juguete"
- Si en un bbox de ~2% de imagen ves tela/textil doblado → es "Ropa doblada" o "Prenda de vestir"

TAREAS (TODO EN UNA SOLA RESPUESTA):

1. **VALIDAR Y ANALIZAR** cada objeto detectado:
   - **MIRA EXACTAMENTE LA REGIÓN DEL BBOX** indicado para ese objeto
   - **USA EL TAMAÑO** para identificar correctamente qué es cada objeto
   - **LEE TEXTOS VISIBLES** en la región (títulos de libros, cómics, etiquetas)
   - Compara el tamaño con otros objetos en la escena para contexto
   - ¿Es útil y reutilizable? (useful: "yes"/"no")
   - Si useful="yes": proporciona ANÁLISIS COMPLETO:
     * name: Nombre específico y detallado del objeto (ej: "Cómic '¡PUTA GUERRA!' sobre Primera Guerra Mundial", "Cochecito de juguete rojo y azul", "Ropa doblada en bolsa")
     * category: furniture, electronics, books, tools, kitchenware, sports, toys, decoration, clothing, containers, other
     * condition: excellent, good, fair, poor
     * description: Descripción detallada del objeto (2-3 frases, menciona características visibles como colores, texto, forma)
     * estimated_value: Rango estimado de valor (opcional)
     * bbox: Coordenadas precisas [x, y, w, h] del objeto identificado (CRÍTICO: usa las coordenadas del objeto real, no del bbox original si no coincide)
   - Si useful="no": solo proporciona reason

2. **BUSCAR OBJETOS FALTANTES Y SUPERPUESTOS** (CRÍTICO):
   - ¿Hay objetos útiles que NO están en la lista de arriba?
   - Busca especialmente: frascos blancos/transparentes, botellas, objetos pequeños claros sobre fondos claros
   - **OBJETOS SUPERPUESTOS**: Si ves objetos superpuestos (uno encima del otro) que no están en la lista, identifícalos
   - Si encuentras objetos faltantes o superpuestos, proporciona:
     * name: Nombre del objeto
     * bbox: Coordenadas precisas [x, y, w, h] donde x,y es esquina superior izquierda
     * category: Categoría del objeto
     * confidence: "high", "medium", o "low"

Responde EXCLUSIVAMENTE con JSON (sin markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "Pesa de entrenamiento negra", "category": "sports", "condition": "good", "description": "Pesa de entrenamiento negra con forma redonda", "estimated_value": "20-40€", "bbox": [x, y, w, h]}},
    {{"n": 2, "useful": "yes", "name": "Silla de oficina", "category": "furniture", "condition": "fair", "description": "Silla con patas oscuras y respaldo", "estimated_value": "30-60€", "bbox": [x, y, w, h]}},
    {{"n": 3, "useful": "yes", "name": "Ropa doblada (camisetas y pantalones)", "category": "clothing", "condition": "good", "description": "Pila de ropa doblada con varias prendas", "estimated_value": "5-15€", "bbox": [x, y, w, h]}},
    {{"n": 4, "useful": "no", "reason": "Fragmento de fondo"}},
    {{"n": 5, "useful": "yes", "name": "Objeto superpuesto detectado", "category": "other", "condition": "good", "description": "Objeto adicional encontrado en región superpuesta", "estimated_value": "10-20€", "bbox": [x, y, w, h]}},
    ...
  ],
  "missing_objects": [
    {{"name": "frasco de medicamento", "bbox": [150, 200, 80, 120], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIOS ESTRICTOS:
- useful="yes" SOLO si es objeto completo, funcional, reutilizable
- **OBJETOS FUNCIONALES ACEPTADOS**: pesas, sillas, mesas, ropa, bolsos, zapatos, libros, herramientas, electrodomésticos, juguetes, contenedores
- useful="no" si es fragmento, fondo, basura, mobiliario fijo del punto limpio
- **VERIFICA QUE EL BBOX CORRESPONDA AL OBJETO**: Si el tamaño no coincide con el objeto identificado, marca useful="no" y explica en reason
- missing_objects: Solo objetos claramente visibles y útiles que NO están en la lista
- bbox en missing_objects: Coordenadas aproximadas [x, y, w, h] donde x,y es esquina superior izquierda

IMPORTANTE:
- Responde para TODOS los {num_objects} objetos detectados
- **USA EL TAMAÑO PARA IDENTIFICAR CORRECTAMENTE** cada objeto
- **CRÍTICO**: Si useful="yes", DEBES incluir "bbox" con las coordenadas precisas [x, y, w, h] del objeto identificado
- El bbox debe corresponder al objeto real que identificaste, no necesariamente al bbox original si no coincide
- missing_objects puede estar vacío [] si no hay objetos faltantes
- NO incluyas markdown (```json)
- Si useful="yes" DEBES incluir todos los campos (name, category, condition, description, estimated_value, bbox)"""
        else:  # English
            return f"""Analyze this recycling center scene.

I've automatically detected {num_objects} objects with these coordinates:

{bboxes_text}

TASKS:

1. **VALIDATE** each detected object:
   - Is it useful and reusable? (useful: "yes"/"no")
   - **OVERLAPPING OBJECTS**: If a bbox contains multiple overlapping objects, identify ALL visible objects and provide separate bboxes for each
   - If useful, provide: name, category, condition, description, estimated_value, bbox (precise coordinates [x, y, w, h] of the identified object)
   - If NOT useful, provide: reason

2. **FIND MISSING AND OVERLAPPING OBJECTS** (CRITICAL):
   - Are there useful objects NOT in the list above?
   - Look especially for: white/transparent bottles, small light objects on light backgrounds
   - **OVERLAPPING OBJECTS**: If you see overlapping objects (one on top of another) not in the list, identify them
   - If you find missing or overlapping objects, provide precise coordinates [x, y, w, h]

Respond EXCLUSIVELY with JSON (no markdown):

{{
  "validated_objects": [
    {{"n": 1, "useful": "yes", "name": "...", "category": "...", "condition": "...", "description": "...", "estimated_value": "...", "bbox": [x, y, w, h]}},
    {{"n": 2, "useful": "no", "reason": "..."}},
    ...
  ],
  "missing_objects": [
    {{"name": "medicine bottle", "bbox": [x, y, w, h], "category": "containers", "confidence": "high"}},
    ...
  ]
}}

CRITERIA:
- useful="yes" ONLY if complete, functional, reusable object
- useful="no" if fragment, background, trash, furniture
- missing_objects: Only clearly visible useful objects NOT in the list
- bbox in missing_objects: Approximate coordinates [x, y, w, h] where x,y is top-left corner

IMPORTANT:
- Respond for ALL {num_objects} detected objects
- **CRITICAL**: If useful="yes", you MUST include "bbox" with precise coordinates [x, y, w, h] of the identified object
- The bbox should correspond to the actual object you identified, not necessarily the original bbox if it doesn't match
- **OVERLAPPING OBJECTS**: If a bbox contains multiple objects, create multiple entries in validated_objects (one per object with its own bbox)
- missing_objects can be empty [] if no missing objects
- NO markdown (```json)"""
    
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

