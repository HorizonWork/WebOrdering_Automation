
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.perception.dom_distiller import DOMDistiller

def clean_episode(data: Dict[str, Any], dom_snapshots_dir: Path = None) -> Dict[str, Any]:
    """
    Prune unnecessary fields to reduce file size and token usage.
    Keeps only what's needed for labeling and training.
    If elements are missing, tries to hydrate from HTML snapshots.
    """
    cleaned = {
        "episode_id": data.get("episode_id"),
        "goal": data.get("goal"),
        "start_url": data.get("start_url"),
        "created_at": data.get("created_at"),
        "final_status": data.get("final_status"),
        "metadata": data.get("metadata", {}), # Preserve metadata for platform info
        "steps": []
    }
    
    distiller = DOMDistiller()
    episode_id = data.get("episode_id")
    
    for step in data.get("steps", []):
        clean_step = {
            "step": step.get("step"),
            "timestamp": step.get("timestamp"),
            "action": step.get("action"), # Keep original action
            "thought": step.get("thought"), # Keep original thought if present
            "page_state": {}
        }
        
        # Prune page_state
        raw_state = step.get("page_state", {}) or {}
        
        # Keep simplified DOM state (distilled elements)
        dom_state = raw_state.get("dom_state", {}) or {}
        
        elements = dom_state.get("elements", [])
        
        # HYDRATION LOGIC: If elements are empty, try to load from HTML snapshot
        if not elements and dom_snapshots_dir:
            step_num = step.get("step")
            html_path = dom_snapshots_dir / episode_id / f"step_{step_num}.html"
            if html_path.exists():
                try:
                    html_content = html_path.read_text(encoding="utf-8")
                    elements = distiller.extract_interactive_elements(html_content)
                    print(f"  [info] Hydrated {len(elements)} elements from {html_path.name}")
                except Exception as e:
                    print(f"  [warn] Failed to hydrate from {html_path.name}: {e}")
        
        clean_dom = {
            "selector_map": dom_state.get("selector_map", {}),
            "elements": []
        }
        
        # Keep only first 150 elements to reduce file size
        # And strip unnecessary attributes
        clean_elements = []
        allowed_attrs = {'mmid', 'class', 'name', 'type', 'aria-label', 'role', 'placeholder', 'value'}
        
        for el in elements[:150]:
            # Filter attributes
            raw_attrs = el.get("attributes", {})
            clean_attrs = {k: v for k, v in raw_attrs.items() if k in allowed_attrs}
            
            clean_el = {
                "id": el.get("id"),
                "tag": el.get("tag"),
                "text": (el.get("text") or "")[:50], # Truncate text
                "selector": el.get("selector"),
                "attributes": clean_attrs
            }
            clean_elements.append(clean_el)
            
        clean_dom["elements"] = clean_elements
        
        clean_step["page_state"] = {
            "page_type": raw_state.get("page_type"),
            "dom_state": clean_dom,
            "vision_state": {
                "screenshot_id": raw_state.get("vision_state", {}).get("screenshot_id")
            }
        }
        
        cleaned["steps"].append(clean_step)
        
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="Clean and prune episodes for labeling")
    parser.add_argument("input_dir", help="Input directory containing raw JSONs")
    parser.add_argument("output_dir", help="Output directory for cleaned JSONs")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer dom_snapshots dir (sibling to input_dir's parent usually, or sibling to input_dir)
    # Assuming structure: .../episodes/*.json  AND .../dom_snapshots/episode_id/*.html
    # If input_dir is .../episodes, then dom_snapshots is .../dom_snapshots
    dom_snapshots_dir = in_dir.parent / "dom_snapshots"
    if not dom_snapshots_dir.exists():
        # Try sibling of input_dir (if input_dir is not named episodes)
        dom_snapshots_dir = in_dir.parent / "dom_snapshots"
        
    print(f"Looking for DOM snapshots in: {dom_snapshots_dir}")
    
    for file_path in in_dir.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            cleaned = clean_episode(data, dom_snapshots_dir=dom_snapshots_dir)
            
            out_path = out_dir / file_path.name
            out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] Cleaned {file_path.name} -> {out_path}")
            
        except Exception as e:
            print(f"[error] Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    main()
