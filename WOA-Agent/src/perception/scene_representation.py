"""
Scene Representation - Structured page state representation
Combines DOM, visual, and semantic information
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.perception.dom_distiller import DOMDistiller
from src.models.phobert_encoder import PhoBERTEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SceneRepresentation:
    """
    Structured representation of web page state.
    
    **Components**:
        - URL & metadata
        - Distilled DOM
        - Interactive elements
        - Semantic embeddings
        - Visual features (optional)
    
    **Use Cases**:
        - Input to planner
        - State comparison
        - Memory storage
        - Change detection
    """
    
    def __init__(
        self,
        encoder: Optional[PhoBERTEncoder] = None,
        distiller: Optional[DOMDistiller] = None
    ):
        """
        Initialize scene representation.
        
        Args:
            encoder: PhoBERT encoder for semantic features
            distiller: DOM distiller
        """
        self.encoder = encoder or PhoBERTEncoder()
        self.distiller = distiller or DOMDistiller()
        
        logger.info("SceneRepresentation initialized")
    
    def create_scene(
        self,
        url: str,
        html: str,
        screenshot: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create structured scene representation.
        
        Args:
            url: Page URL
            html: Raw HTML
            screenshot: Screenshot (base64)
            metadata: Additional metadata
            
        Returns:
            Scene dict with all components
        """
        logger.info(f"Creating scene for: {url}")
        
        # Distill DOM
        distilled_dom = self.distiller.distill(html, mode='text_only')
        
        # Extract interactive elements
        elements = self.distiller.extract_interactive_elements(html)
        
        # Generate semantic embeddings for elements
        element_texts = [
            f"{e['tag']} {e['text']}" for e in elements
        ]
        
        if element_texts:
            element_embeddings = self.encoder.encode_text(element_texts)
        else:
            element_embeddings = []
        
        # Build scene
        scene = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'dom': {
                'raw_size': len(html),
                'distilled_size': len(distilled_dom),
                'distilled_content': distilled_dom
            },
            'elements': [
                {
                    **elem,
                    'embedding': element_embeddings[i].tolist() if len(element_embeddings) > 0 else None
                }
                for i, elem in enumerate(elements)
            ],
            'visual': {
                'screenshot': screenshot,
                'has_screenshot': screenshot is not None
            },
            'metadata': metadata or {}
        }
        
        logger.info(f"✓ Scene created: {len(elements)} elements, DOM: {len(distilled_dom)} chars")
        
        return scene
    
    def compare_scenes(
        self,
        scene1: Dict,
        scene2: Dict
    ) -> Dict:
        """
        Compare two scenes to detect changes.
        
        Args:
            scene1: First scene
            scene2: Second scene
            
        Returns:
            Comparison dict with changes
        """
        changes = {
            'url_changed': scene1['url'] != scene2['url'],
            'dom_changed': scene1['dom']['distilled_content'] != scene2['dom']['distilled_content'],
            'elements_changed': {
                'added': 0,
                'removed': 0,
                'modified': 0
            }
        }
        
        # Compare element counts
        elements1 = {e['selector']: e for e in scene1['elements']}
        elements2 = {e['selector']: e for e in scene2['elements']}
        
        # Find added/removed
        selectors1 = set(elements1.keys())
        selectors2 = set(elements2.keys())
        
        added = selectors2 - selectors1
        removed = selectors1 - selectors2
        common = selectors1 & selectors2
        
        changes['elements_changed']['added'] = len(added)
        changes['elements_changed']['removed'] = len(removed)
        
        # Check modified (text changed)
        modified = 0
        for sel in common:
            if elements1[sel]['text'] != elements2[sel]['text']:
                modified += 1
        
        changes['elements_changed']['modified'] = modified
        
        # Overall change score (0-1)
        total_changes = len(added) + len(removed) + modified
        total_elements = len(elements1) + len(elements2)
        
        if total_elements > 0:
            changes['change_score'] = total_changes / total_elements
        else:
            changes['change_score'] = 0.0
        
        logger.debug(f"Scene comparison: score={changes['change_score']:.2f}")
        
        return changes
    
    def find_element_by_text(
        self,
        scene: Dict,
        text_query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Find elements by text similarity.
        
        Args:
            scene: Scene dict
            text_query: Text to search for
            top_k: Number of results
            
        Returns:
            List of matching elements
        """
        if not scene['elements']:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode_text(text_query)
        
        # Compute similarities
        results = []
        for elem in scene['elements']:
            if elem.get('embedding') is None:
                continue
            
            elem_embedding = elem['embedding']
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, elem_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(elem_embedding) + 1e-8
            )
            
            results.append({
                'element': elem,
                'similarity': float(similarity)
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def summarize_scene(self, scene: Dict) -> str:
        """Generate human-readable scene summary"""
        summary = f"""
Scene Summary:
  URL: {scene['url']}
  Timestamp: {scene['timestamp']}
  
  DOM:
    Raw size: {scene['dom']['raw_size']} chars
    Distilled size: {scene['dom']['distilled_size']} chars
    Reduction: {(1 - scene['dom']['distilled_size']/scene['dom']['raw_size'])*100:.1f}%
  
  Interactive Elements: {len(scene['elements'])}
"""
        
        # Element breakdown
        if scene['elements']:
            tags = {}
            for elem in scene['elements']:
                tag = elem['tag']
                tags[tag] = tags.get(tag, 0) + 1
            
            summary += "    Element types:\n"
            for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True):
                summary += f"      - {tag}: {count}\n"
        
        summary += f"\n  Screenshot: {'Available' if scene['visual']['has_screenshot'] else 'Not available'}"
        
        return summary.strip()
    
    def save_scene(self, scene: Dict, path: str):
        """Save scene to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove embeddings for storage (too large)
        scene_copy = scene.copy()
        for elem in scene_copy['elements']:
            if 'embedding' in elem:
                del elem['embedding']
        
        with open(path, 'w') as f:
            json.dump(scene_copy, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Scene saved to {path}")
    
    def load_scene(self, path: str) -> Dict:
        """Load scene from JSON file"""
        with open(path, 'r') as f:
            scene = json.load(f)
        
        logger.info(f"📂 Scene loaded from {path}")
        return scene


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("SceneRepresentation - Test")
    print("=" * 70 + "\n")
    
    # Test HTML
    test_html = """
    <html>
    <body>
        <h1>E-commerce Site</h1>
        <form id="search">
            <input name="q" placeholder="Search products" />
            <button type="submit">Search</button>
        </form>
        <div class="products">
            <a href="/product/1">Product 1</a>
            <a href="/product/2">Product 2</a>
            <button class="add-to-cart">Add to cart</button>
        </div>
    </body>
    </html>
    """
    
    scene_rep = SceneRepresentation()
    
    # Test 1: Create scene
    print("Test 1: Create Scene")
    print("-" * 40)
    
    scene = scene_rep.create_scene(
        url="https://example.com",
        html=test_html,
        metadata={'page_type': 'search'}
    )
    
    print(f"✓ Scene created")
    print(f"  Elements: {len(scene['elements'])}")
    print(f"  DOM size: {scene['dom']['distilled_size']} chars\n")
    
    # Test 2: Scene summary
    print("\nTest 2: Scene Summary")
    print("-" * 40)
    summary = scene_rep.summarize_scene(scene)
    print(summary)
    
    # Test 3: Find elements by text
    print("\n\nTest 3: Find Elements by Text")
    print("-" * 40)
    
    results = scene_rep.find_element_by_text(scene, "search button", top_k=3)
    print(f"Query: 'search button'\n")
    for i, result in enumerate(results, 1):
        elem = result['element']
        print(f"{i}. {elem['tag']} - {elem['text']}")
        print(f"   Selector: {elem['selector']}")
        print(f"   Similarity: {result['similarity']:.4f}\n")
    
    # Test 4: Compare scenes
    print("\nTest 4: Compare Scenes")
    print("-" * 40)
    
    # Create modified scene
    modified_html = test_html.replace("Product 1", "Product A")
    scene2 = scene_rep.create_scene(
        url="https://example.com",
        html=modified_html
    )
    
    changes = scene_rep.compare_scenes(scene, scene2)
    print(f"URL changed: {changes['url_changed']}")
    print(f"DOM changed: {changes['dom_changed']}")
    print(f"Change score: {changes['change_score']:.2f}")
    print(f"Elements: +{changes['elements_changed']['added']}, -{changes['elements_changed']['removed']}, ~{changes['elements_changed']['modified']}")
    
    print("\n" + "=" * 70)
    print("✅ All Tests Completed!")
    print("=" * 70)
