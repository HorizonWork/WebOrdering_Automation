"""
DOM Distiller - Clean and simplify HTML for LLM consumption
Based on Agent-E's DOM distillation approach
FIXED: Added robust None-safety checks for all operations
"""

import sys
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup, Comment

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger # noqa: E402

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
        max_dom_size: int = 50000,
        max_elements: int = 300
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
            'input', 'button', 'a', 'select', 'textarea', 'form', 'label'
        }
        
        # Important attributes to keep
        self.important_attrs = {
            'id', 'class', 'name', 'type', 'placeholder', 'href', 'value', 'aria-label', 'role'
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
        if not html or not isinstance(html, str):
            return ""
        
        logger.debug(f"Distilling HTML ({len(html)} chars) in mode: {mode}")
        
        try:
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
        
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return ""
    
    def _remove_noise(self, soup: BeautifulSoup):
        """Remove noisy elements with robust None-safety"""
        if not soup:
            return
        
        # Remove scripts, styles, etc.
        noise_tags = ['script', 'style', 'noscript', 'meta', 'link', 
                     'svg', 'img', 'video', 'audio', 'iframe']
        
        for tag_name in noise_tags:
            try:
                for tag in soup.find_all(tag_name):
                    if tag and hasattr(tag, 'decompose'):
                        tag.decompose()
            except Exception as e:
                logger.debug(f"Error removing {tag_name}: {e}")
                continue
        
        # Remove comments
        try:
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                if comment and hasattr(comment, 'extract'):
                    comment.extract()
        except Exception as e:
            logger.debug(f"Error removing comments: {e}")
        
        # Remove hidden elements (ROBUST None checks)
        try:
            all_tags = list(soup.find_all(True))  # Convert to list to avoid modification during iteration
            
            for tag in all_tags:
                # Skip if None or malformed
                if tag is None or not hasattr(tag, 'attrs'):
                    continue
                
                # Skip if attrs is None
                if tag.attrs is None:
                    continue
                
                try:
                    # Safe get style
                    style = ''
                    if isinstance(tag.attrs, dict):
                        style = tag.attrs.get('style', '')
                    
                    # Check if hidden
                    if style and isinstance(style, str):
                        style_lower = style.lower().replace(' ', '')
                        if 'display:none' in style_lower or 'visibility:hidden' in style_lower:
                            if hasattr(tag, 'decompose'):
                                tag.decompose()
                            continue
                    
                    # Safe get class
                    classes = []
                    if isinstance(tag.attrs, dict):
                        classes = tag.attrs.get('class', [])
                    
                    # Check if has hidden class
                    if classes:
                        class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                        class_lower = class_str.lower()
                        
                        hidden_keywords = ['hidden', 'hide', 'd-none', 'invisible', 'display-none']
                        if any(kw in class_lower for kw in hidden_keywords):
                            if hasattr(tag, 'decompose'):
                                tag.decompose()
                            continue
                
                except (AttributeError, TypeError, KeyError) as e:
                    # Skip malformed tags
                    logger.debug(f"Error processing tag: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error removing hidden elements: {e}")
    
    def _distill_text_only(self, soup: BeautifulSoup) -> str:
        """Keep only interactive elements"""
        if not soup:
            return ""
        
        try:
            # Convert to list to avoid modification during iteration
            all_tags = list(soup.find_all(True))
            
            for tag in all_tags:
                if not tag or tag.name is None:
                    continue
                
                if tag.name not in self.interactive_tags:
                    # Check if has interactive children
                    has_interactive = bool(tag.find_all(self.interactive_tags))
                    if not has_interactive and hasattr(tag, 'decompose'):
                        tag.decompose()
            
            # Clean attributes
            for tag in soup.find_all(True):
                if not tag or not hasattr(tag, 'attrs') or tag.attrs is None:
                    continue
                
                try:
                    if isinstance(tag.attrs, dict):
                        attrs_to_keep = {
                            k: v for k, v in tag.attrs.items()
                            if k in self.important_attrs
                        }
                        tag.attrs = attrs_to_keep
                except Exception:
                    tag.attrs = {}
            
            return str(soup)
        
        except Exception as e:
            logger.error(f"Text-only distillation failed: {e}")
            return ""
    
    def _distill_structure(self, soup: BeautifulSoup) -> str:
        """Keep structure but clean attributes"""
        if not soup:
            return ""
        
        try:
            for tag in soup.find_all(True):
                if not tag or not hasattr(tag, 'attrs') or tag.attrs is None:
                    continue
                
                try:
                    # Only keep important attrs
                    attrs_to_keep = {}
                    
                    if isinstance(tag.attrs, dict):
                        for k, v in tag.attrs.items():
                            if k in self.important_attrs or k.startswith('data-'):
                                # Truncate long values
                                if isinstance(v, str) and len(v) > 100:
                                    v = v[:100] + "..."
                                attrs_to_keep[k] = v
                    
                    tag.attrs = attrs_to_keep
                
                except Exception:
                    tag.attrs = {}
            
            return str(soup)
        
        except Exception as e:
            logger.error(f"Structure distillation failed: {e}")
            return ""
    
    def _distill_full(self, soup: BeautifulSoup) -> str:
        """Keep everything but clean"""
        if not soup:
            return ""
        
        try:
            for tag in soup.find_all(True):
                if not tag or not hasattr(tag, 'attrs') or tag.attrs is None:
                    continue
                
                try:
                    if isinstance(tag.attrs, dict):
                        # Remove inline styles and event handlers
                        attrs_to_remove = [
                            k for k in list(tag.attrs.keys())
                            if k.startswith('on') or k == 'style'
                        ]
                        for attr in attrs_to_remove:
                            try:
                                del tag.attrs[attr]
                            except KeyError:
                                pass
                except Exception:
                    pass
            
            return str(soup)
        
        except Exception as e:
            logger.error(f"Full distillation failed: {e}")
            return ""
    
    def extract_interactive_elements(self, html: str) -> List[Dict]:
        """
        Extract all interactive elements with metadata.
        
        Returns:
            List of dicts with {id, tag, text, selector, attributes}
        """
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            self._remove_noise(soup)
            
            elements = []
            
            for idx, tag in enumerate(soup.find_all(self.interactive_tags)):
                if not tag:
                    continue
                
                try:
                    # Get text (truncated)
                    text = ''
                    if hasattr(tag, 'get_text'):
                        text = tag.get_text(strip=True)
                        if len(text) > 100:
                            text = text[:100] + "..."
                    
                    # Get important attributes
                    attrs = {}
                    if hasattr(tag, 'attrs') and isinstance(tag.attrs, dict):
                        attrs = {
                            k: v for k, v in tag.attrs.items()
                            if k in self.important_attrs
                        }
                    
                    # Generate selector
                    selector = self._generate_selector(tag, idx)
                    
                    element = {
                        'id': idx,
                        'tag': tag.name if hasattr(tag, 'name') else 'unknown',
                        'text': text,
                        'selector': selector,
                        'attributes': attrs
                    }
                    
                    elements.append(element)
                    
                    # Stop if max reached
                    if len(elements) >= self.max_elements:
                        logger.warning(f"Max elements reached: {self.max_elements}")
                        break
                
                except Exception as e:
                    logger.debug(f"Error extracting element {idx}: {e}")
                    continue
            
            logger.info(f"✓ Extracted {len(elements)} interactive elements")
            return elements
        
        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            return []
    
    def _generate_selector(self, tag, fallback_idx: int) -> str:
        """Generate CSS selector for element (with None-safety)"""
        if not tag or not hasattr(tag, 'name'):
            return f"unknown-{fallback_idx}"
        
        try:
            # Priority: id > name > class > nth-of-type
            if hasattr(tag, 'attrs') and isinstance(tag.attrs, dict):
                if tag.attrs.get('id'):
                    return f"#{tag.attrs['id']}"
                elif tag.attrs.get('name'):
                    return f"{tag.name}[name='{tag.attrs['name']}']"
                elif tag.attrs.get('class'):
                    classes = tag.attrs['class']
                    if isinstance(classes, list) and classes:
                        classes_str = '.'.join(classes[:2])  # Max 2 classes
                        return f"{tag.name}.{classes_str}"
            
            return f"{tag.name}:nth-of-type({fallback_idx + 1})"
        
        except Exception:
            return f"{tag.name}-{fallback_idx}"
    
    def annotate_with_ids(self, html: str, prefix: str = "mmid") -> str:
        """
        Add unique IDs to interactive elements.
        
        Args:
            html: HTML string
            prefix: ID prefix (mmid = Multi-Modal ID)
            
        Returns:
            Annotated HTML
        """
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            idx = 0
            for tag in soup.find_all(self.interactive_tags):
                if tag and hasattr(tag, '__setitem__'):
                    try:
                        tag[prefix] = f"{prefix}-{idx}"
                        idx += 1
                    except Exception:
                        pass
            
            logger.info(f"✓ Annotated {idx} elements with {prefix} IDs")
            return str(soup)
        
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            return html
    
    def find_action_buttons(self, html: str, action_type: str = 'submit') -> List[Dict]:
        """Find specific action buttons (e.g., submit, cancel, etc.)"""
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            buttons = []
            
            # Find buttons and inputs
            for tag in soup.find_all(['button', 'input']):
                if not tag:
                    continue
                
                try:
                    tag_type = ''
                    if hasattr(tag, 'attrs') and isinstance(tag.attrs, dict):
                        tag_type = tag.attrs.get('type', '').lower()
                    
                    if action_type in tag_type:
                        text = tag.get_text(strip=True) if hasattr(tag, 'get_text') else ''
                        buttons.append({
                            'tag': tag.name if hasattr(tag, 'name') else 'unknown',
                            'type': tag_type,
                            'text': text,
                            'selector': self._generate_selector(tag, len(buttons))
                        })
                except Exception:
                    continue
            
            return buttons
        
        except Exception as e:
            logger.error(f"Button search failed: {e}")
            return []


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
