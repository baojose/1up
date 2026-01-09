"""
Test Claude API connection and basic functionality.
"""
import os
import anthropic
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_claude_api():
    """Test basic Claude API connection."""
    api_key = os.environ.get('CLAUDE_API_KEY')
    
    if not api_key:
        logger.error("❌ CLAUDE_API_KEY not set")
        logger.error("   Run: export CLAUDE_API_KEY='sk-ant-api03-...'")
        return False
    
    logger.info("✅ CLAUDE_API_KEY found")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test: ask Claude to return JSON
        logger.info("Testing Claude API connection...")
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": "Responde con un objeto JSON: {\"test\": \"ok\", \"status\": \"working\"}"
            }]
        )
        
        response = message.content[0].text.strip()
        logger.info(f"Response: {response}")
        
        # Try to parse JSON
        try:
            # Remove markdown if present
            if response.startswith('```'):
                parts = response.split('```')
                if len(parts) >= 3:
                    response = parts[1]
                    if response.startswith('json'):
                        response = response[4:]
            
            result = json.loads(response)
            logger.info(f"✅ JSON parsed successfully: {result}")
            return True
            
        except json.JSONDecodeError:
            logger.warning("⚠️  Response is not valid JSON, but API is working")
            return True
            
    except anthropic.APIError as e:
        logger.error(f"❌ Claude API error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = test_claude_api()
    if success:
        logger.info("\n✅ Claude API test passed!")
    else:
        logger.error("\n❌ Claude API test failed!")
        exit(1)

