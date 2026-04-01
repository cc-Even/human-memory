import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON from text, handling potential Markdown blocks.
    
    Args:
        text: The text to extract JSON from.
        
    Returns:
        The parsed JSON object, or None if extraction or parsing fails.
    """
    if not text:
        return None
        
    clean_text = text.strip()
    
    # Try to extract from Markdown code blocks
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}. Text: {clean_text}")
        
        # Fallback: try to find anything that looks like a JSON object or array
        # This is a bit risky but can help if there's trailing text
        match = re.search(r"(\[.*\]|\{.*\})", clean_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        return None
