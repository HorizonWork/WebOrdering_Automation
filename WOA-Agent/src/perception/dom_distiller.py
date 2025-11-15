"""
DOM Distiller - Clean and simplify HTML for LLM consumption
Based on Agent-E's DOM distillation approach
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DOMDistiller:
    """
    Distills complex HTML into clean, LLM-friendly format.
    
    **Problem**: Raw HTML is too long and noisy for LLMs
    **Solution**: Keep only interactive elements + important structure
    
    **Distillation Modes**:
        - text_only: Remove all non-interactive elements
        - structure: Keep structure but clean attributes
        - full: Keep everything but clean
    
    **What we keep**:
        - Interactive: input, button, a, select, textarea
        - Structure: div, span with IDs/classes
        - Important attrs: id, class, name, placeholder, href
    
    **What we remove**:
        - Scripts, styles, SVGs, images
        - Inline styles and event handlers
        - Hidden elements (display:none)
        - Excessive whitespace
    """
    
    def __init__(
        self,
        max_dom_size: int = 10000,
        max_elements: int = 100
    ):
        """
        Initialize DOM distiller.
        
        Args:
            max_dom_size: Max DOM size in characters
            max_elements: Max interactive elements to extract
        """
        self.max_dom_size = max_dom_size
        self.max_elements = max_elements
        
        # Interactive elements we care about
        self.interactive_tags = {
            'input', 'button', 'a', 'select', 'textarea',
            'form', 'label'
        }
        
        # Important attributes to keep
        self.important_attrs = {
            'id', 'class', 'name', 'type', 'placeholder',
            'href', 'value', 'aria-label', 'role'
        }
        
        logger.info(f"DOMDistiller initialized (max_size={max_dom_size}, max_elements={max_elements})")
    
    def distill(
        self,
        html: str,
        mode: str = 'text_only'
    ) -> str:
        """
        Distill HTML to clean format.
        
        Args:
            html: Raw HTML
            mode: Distillation mode ('text_only', 'structure', 'full')
            
        Returns:
            Distilled HTML string
        """
        if not html:
            return ""
        
        logger.debug(f"Distilling HTML ({len(html)} chars) in mode: {mode}")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove noise
        self._remove_noise(soup)
        
        # Apply distillation based on mode
        if mode == 'text_only':
            distilled = self._distill_text_only(soup)
        elif mode == 'structure':
            distilled = self._distill_structure(soup)
        else:  # full
            distilled = self._distill_full(soup)
        
        # Truncate if too long
        if len(distilled) > self.max_dom_size:
            logger.warning(f"DOM truncated: {len(distilled)} → {self.max_dom_size} chars")
            distilled = distilled[:self.max_dom_size] + "\n<!-- TRUNCATED -->"
        
        logger.debug(f"Distilled: {len(html)} → {len(distilled)} chars ({len(distilled)/len(html)*100:.1f}%)")
        
        return distilled
    
    def _remove_noise(self, soup: BeautifulSoup):
        """Remove noisy elements"""
        # Remove scripts, styles, etc.
        for tag in soup(['script', 'style', 'noscript', 'meta', 'link', 
                        'svg', 'img', 'video', 'audio', 'iframe']):
            tag.decompose()
        
        # Remove hidden elements
        for tag in soup.find_all():
            style = tag.get('style', '')
            if 'display:none' in style or 'display: none' in style:
                tag.decompose()
    
    def _distill_text_only(self, soup: BeautifulSoup) -> str:
        """Keep only interactive elements"""
        # Remove all non-interactive elements
        for tag in soup.find_all():
            if tag.name not in self.interactive_tags:
                # Check if has interactive children
                has_interactive = bool(tag.find_all(self.interactive_tags))
                if not has_interactive:
                    tag.decompose()
        
        # Clean attributes
        for tag in soup.find_all():
            attrs_to_keep = {
                k: v for k, v in tag.attrs.items()
                if k in self.important_attrs
            }
            tag.attrs = attrs_to_keep
        
        return str(soup)
    
    def _distill_structure(self, soup: BeautifulSoup) -> str:
        """Keep structure but clean attributes"""
        # Keep structure (div, span, p, etc.)
        # But clean attributes
        for tag in soup.find_all():
            # Only keep important attrs
            attrs_to_keep = {}
            for k, v in tag.attrs.items():
                if k in self.important_attrs or k.startswith('data-'):
                    # Truncate long values
                    if isinstance(v, str) and len(v) > 100:
                        v = v[:100] + "..."
                    attrs_to_keep[k] = v
            
            tag.attrs = attrs_to_keep
        
        return str(soup)
    
    def _distill_full(self, soup: BeautifulSoup) -> str:
        """Keep everything but clean"""
        # Just clean attributes
        for tag in soup.find_all():
            # Remove inline styles and event handlers
            attrs_to_remove = [
                k for k in tag.attrs.keys()
                if k.startswith('on') or k == 'style'
            ]
            for attr in attrs_to_remove:
                del tag.attrs[attr]
        
        return str(soup)
    
    def extract_interactive_elements(self, html: str) -> List[Dict]:
        """
        Extract all interactive elements with metadata.
        
        Returns:
            List of dicts with {id, tag, text, selector, attributes}
        """
        soup = BeautifulSoup(html, 'html.parser')
        self._remove_noise(soup)
        
        elements = []
        
        for idx, tag in enumerate(soup.find_all(self.interactive_tags)):
            # Get text (truncated)
            text = tag.get_text(strip=True)
            if len(text) > 100:
                text = text[:100] + "..."
            
            # Get important attributes
            attrs = {
                k: v for k, v in tag.attrs.items()
                if k in self.important_attrs
            }
            
            # Generate selector
            selector = self._generate_selector(tag, idx)
            
            element = {
                'id': idx,
                'tag': tag.name,
                'text': text,
                'selector': selector,
                'attributes': attrs
            }
            
            elements.append(element)
            
            # Stop if max reached
            if len(elements) >= self.max_elements:
                logger.warning(f"Max elements reached: {self.max_elements}")
                break
        
        logger.info(f"✓ Extracted {len(elements)} interactive elements")
        return elements
    
    def _generate_selector(self, tag: BeautifulSoup, fallback_idx: int) -> str:
        """Generate CSS selector for element"""
        # Priority: id > name > class > nth-of-type
        if tag.get('id'):
            return f"#{tag['id']}"
        elif tag.get('name'):
            return f"{tag.name}[name='{tag['name']}']"
        elif tag.get('class'):
            classes = '.'.join(tag['class'][:2])  # Max 2 classes
            return f"{tag.name}.{classes}"
        else:
            return f"{tag.name}:nth-of-type({fallback_idx + 1})"
    
    def annotate_with_ids(self, html: str, prefix: str = "mmid") -> str:
        """
        Add unique IDs to interactive elements.
        
        Args:
            html: HTML string
            prefix: ID prefix (mmid = Multi-Modal ID)
            
        Returns:
            Annotated HTML
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        idx = 0
        for tag in soup.find_all(self.interactive_tags):
            tag[prefix] = f"{prefix}-{idx}"
            idx += 1
        
        logger.info(f"✓ Annotated {idx} elements with {prefix} IDs")
        return str(soup)


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("DOMDistiller - Test")
    print("=" * 70 + "\n")
    
    # Test HTML
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test</title>
        <script>console.log('test')</script>
        <style>.test { color: red; }</style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome</h1>
            <form id="login-form">
                <input type="text" name="username" placeholder="Username" />
                <input type="password" name="password" placeholder="Password" />
                <button type="submit">Login</button>
            </form>
            <a href="/about">About</a>
            <img src="logo.png" alt="Logo" />
            <p>Some text content</p>
        </div>
        <div style="display:none">Hidden content</div>
    </body>
    </html>
    """
    
    distiller = DOMDistiller()
    
    # Test 1: Text-only distillation
    print("Test 1: Text-Only Distillation")
    print("-" * 40)
    text_only = distiller.distill(test_html, mode='text_only')
    print(f"Original size: {len(test_html)} chars")
    print(f"Distilled size: {len(text_only)} chars")
    print(f"Reduction: {(1 - len(text_only)/len(test_html))*100:.1f}%")
    print(f"\nDistilled HTML:\n{text_only}\n")
    
    # Test 2: Extract interactive elements
    print("\nTest 2: Extract Interactive Elements")
    print("-" * 40)
    elements = distiller.extract_interactive_elements(test_html)
    print(f"Found {len(elements)} interactive elements:\n")
    for elem in elements:
        print(f"  [{elem['id']}] {elem['tag']} - {elem['selector']}")
        print(f"       Text: {elem['text']}")
        print(f"       Attrs: {elem['attributes']}\n")
    
    # Test 3: Annotate with IDs
    print("\nTest 3: Annotate with IDs")
    print("-" * 40)
    annotated = distiller.annotate_with_ids(test_html, prefix="elem")
    print(f"Annotated HTML length: {len(annotated)} chars")
    print(f"Sample: {annotated[:300]}...\n")
    
    print("=" * 70)
    print("✅ All Tests Completed!")
    print("=" * 70)
