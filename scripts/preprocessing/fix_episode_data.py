# -*- coding: utf-8 -*-
"""
Fix & normalize raw episode data from policy collection.
"""

import json
import glob
from pathlib import Path
from typing import Dict, Any

def fix_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Apply normalization rules"""
    
    goal = episode.get("goal", "")
    
    for step in episode.get("steps", []):
        # Fix 1: Correct page_type based on URL
        page_state = step.get("page_state", {})
        url = page_state.get("url", "").lower()
        
        if "catalog" in url or "q=" in url:
            page_state["page_type"] = "search_results"
        elif "product" in url or "-i" in url:
            page_state["page_type"] = "product_detail"
        elif "cart" in url:
            page_state["page_type"] = "cart"
        else:
            page_state["page_type"] = "home"
        
        # Fix 2: Normalize search text (extract product name from goal)
        action = step.get("action", {})
        if action.get("skill") == "fill":
            params = action.get("params", {})
            
            # If text is the full goal, extract just product name
            text = params.get("text", "")
            if text == goal:
                # Try to extract just the product part
                # Heuristic: "Tìm X màu Y giá Z" → extract X
                if "màu" in text:
                    product = text.split(" màu")[0].replace("Tìm ", "").strip()
                    params["text"] = product
                elif "giá" in text:
                    product = text.split(" giá")[0].replace("Tìm ", "").strip()
                    params["text"] = product
        
        # Fix 3: Ensure fallback_selectors exist
        if "fallback_selectors" not in params and action.get("skill") in ["fill", "click"]:
            selector = params.get("selector", "")
            if selector:
                # Generate common fallbacks based on selector
                fallbacks = []
                if "input" in selector.lower():
                    fallbacks = [
                        "input[type='search']",
                        "input[placeholder*='search']",
                        "//input[@type='search']"
                    ]
                elif "button" in selector.lower():
                    fallbacks = [
                        "button[type='submit']",
                        "button:has-text('Search')",
                        "//button[contains(@class, 'search')]"
                    ]
                
                if fallbacks:
                    params["fallback_selectors"] = fallbacks
    
    return episode

def normalize_episodes(input_dir="data/raw/lazada/episodes", output_dir="data/raw/lazada/episodes_fixed"):
    """Process all episodes"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    files = glob.glob(f"{input_dir}/ep_*.json")
    print(f"🔍 Found {len(files)} episodes to fix.")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                episode = json.load(f)
            
            # Apply fixes
            fixed = fix_episode(episode)
            
            # Save fixed version
            output_path = Path(output_dir) / Path(file_path).name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fixed, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Fixed: {Path(file_path).name}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ All episodes fixed! Saved to: {output_dir}")

if __name__ == "__main__":
    normalize_episodes()
