
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

def compact_elements(elements: List[Dict[str, Any]]) -> str:
    """
    Convert list of element dicts to a compact markdown-like string.
    Format: [id] <tag> text (attrs)
    Aggressively filters empty/useless elements.
    """
    lines = []
    seen_texts = set()
    
    # Priority attributes that make an element worth keeping even if empty text
    meaningful_attrs = {'placeholder', 'aria-label', 'name', 'title', 'alt', 'value', 'role'}
    
    for el in elements:
        # Get essential data
        el_id = el.get("id", "")
        tag = el.get("tag", "unknown")
        text = el.get("text", "").strip().replace("\n", " ")
        attrs = el.get("attributes", {})
        
        # 1. FILTER: Skip if no text AND no meaningful attributes
        has_meaningful_attr = any(k in attrs for k in meaningful_attrs)
        if not text and not has_meaningful_attr:
            continue
            
        # 2. DEDUPLICATION: Skip if exact text+tag already seen (unless it's an input)
        # We allow inputs to duplicate because they might be different fields
        if text and tag not in ['input', 'select', 'textarea']:
            key = f"{tag}:{text}"
            if key in seen_texts:
                continue
            seen_texts.add(key)
        
        # Format: [id] <tag> text
        line = f"[{el_id}] <{tag}> {text}"
        
        # Add key attributes
        attr_parts = []
        for k, v in attrs.items():
            if k in meaningful_attrs or (k == 'class' and 'btn' in str(v)):
                attr_parts.append(f"{k}='{v}'")
                
        if attr_parts:
            line += f" ({' '.join(attr_parts)})"
        
        lines.append(line)
        
    return "\n".join(lines)

def compact_episode(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact the episode structure.
    Replaces 'elements' list with 'dom_text' string.
    """
    compacted = {
        "episode_id": data.get("episode_id"),
        "goal": data.get("goal"),
        "metadata": data.get("metadata", {}),
        "steps": []
    }
    
    for step in data.get("steps", []):
        # Get elements from page_state
        page_state = step.get("page_state", {})
        dom_state = page_state.get("dom_state", {})
        elements = dom_state.get("elements", [])
        
        # Convert to compact text
        dom_text = compact_elements(elements)
        
        compact_step = {
            "step": step.get("step"),
            "action": step.get("action"),
            "thought": step.get("thought"),
            "dom_text": dom_text, # Replaces complex page_state
            "screenshot": page_state.get("vision_state", {}).get("screenshot_id")
        }
        
        compacted["steps"].append(compact_step)
        
    return compacted

def main():
    parser = argparse.ArgumentParser(description="Compact episodes for efficient LLM/ViT5 usage")
    parser.add_argument("input_dir", help="Input directory (cleaned episodes)")
    parser.add_argument("output_dir", help="Output directory for compacted episodes")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in in_dir.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            compacted = compact_episode(data)
            
            out_path = out_dir / file_path.name
            out_path.write_text(json.dumps(compacted, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] Compacted {file_path.name} -> {out_path}")
            
        except Exception as e:
            print(f"[error] Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    main()
