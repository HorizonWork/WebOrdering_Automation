"""
Validators - Input validation utilities
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import re
from urllib.parse import urlparse

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"
    
    # Parse URL
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            return False, f"Invalid scheme: {parsed.scheme} (must be http/https)"
        
        # Check netloc
        if not parsed.netloc:
            return False, "Missing domain"
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


def validate_selector(selector: str) -> tuple[bool, Optional[str]]:
    """
    Validate CSS selector format.
    
    Args:
        selector: CSS selector to validate
        
    Returns:
        (is_valid, error_message)
    """
    if not selector:
        return False, "Selector is empty"
    
    if not isinstance(selector, str):
        return False, "Selector must be string"
    
    # Basic validation (not exhaustive)
    # Check for common invalid patterns
    invalid_patterns = [
        r'^\s*$',  # Only whitespace
        r'[<>]',   # HTML tags
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, selector):
            return False, f"Invalid selector pattern: {selector}"
    
    return True, None


def validate_action(action: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate action dict format.
    
    Args:
        action: Action dict to validate
        
    Returns:
        (is_valid, error_message)
    """
    # Must be dict
    if not isinstance(action, dict):
        return False, "Action must be dict"
    
    # Must have 'skill' key
    if 'skill' not in action:
        return False, "Action missing 'skill' key"
    
    # Skill must be string
    if not isinstance(action['skill'], str):
        return False, "Skill must be string"
    
    # Must have 'params' key
    if 'params' not in action:
        return False, "Action missing 'params' key"
    
    # Params must be dict
    if not isinstance(action['params'], dict):
        return False, "Params must be dict"
    
    return True, None


def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> tuple[bool, Optional[str]]:
    """
    Validate user query.
    
    Args:
        query: User query
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        (is_valid, error_message)
    """
    if not query:
        return False, "Query is empty"
    
    if not isinstance(query, str):
        return False, "Query must be string"
    
    query_len = len(query.strip())
    
    if query_len < min_length:
        return False, f"Query too short (min {min_length} chars)"
    
    if query_len > max_length:
        return False, f"Query too long (max {max_length} chars)"
    
    return True, None


def validate_config(config: Dict, required_keys: list) -> tuple[bool, Optional[str]]:
    """
    Validate configuration dict.
    
    Args:
        config: Config dict
        required_keys: List of required keys
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Config must be dict"
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"
    
    return True, None


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Validators - Test")
    print("=" * 70 + "\n")
    
    # Test URL validation
    print("Test 1: URL Validation")
    print("-" * 40)
    
    test_urls = [
        "https://example.com",
        "http://shopee.vn",
        "ftp://invalid.com",
        "",
        "not-a-url"
    ]
    
    for url in test_urls:
        valid, error = validate_url(url)
        status = "yes" if valid else "no"
        print(f"{status} {url or '(empty)'}: {error or 'Valid'}")
    
    # Test selector validation
    print("\n\nTest 2: Selector Validation")
    print("-" * 40)
    
    test_selectors = [
        "#search",
        ".button",
        "input[type='text']",
        "",
        "<script>",
        "button.primary"
    ]
    
    for selector in test_selectors:
        valid, error = validate_selector(selector)
        status = "yes" if valid else "no"
        print(f"{status} {selector or '(empty)'}: {error or 'Valid'}")
    
    # Test action validation
    print("\n\nTest 3: Action Validation")
    print("-" * 40)
    
    test_actions = [
        {'skill': 'click', 'params': {'selector': '#btn'}},
        {'skill': 'goto'},
        {'params': {'url': 'test'}},
        "not-a-dict"
    ]
    
    for action in test_actions:
        valid, error = validate_action(action)
        status = "yes" if valid else "no"
        print(f"{status} {action}: {error or 'Valid'}")
    
    print("\n" + "=" * 70)
    print("yes Validators test completed!")
    print("=" * 70)
