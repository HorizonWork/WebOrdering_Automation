"""
Enhanced Gemini-based annotator for human teleoperation episodes.

Analyzes screenshots and DOM changes to infer:
- Action type (click, type, scroll, select, navigate)
- Target element selector
- Input values (for type actions)
- Reason/intent
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def load_screenshot_base64(screenshot_path: Path) -> Optional[str]:
    """Load screenshot and convert to base64 for Gemini."""
    if not screenshot_path.exists():
        return None
    try:
        img_bytes = screenshot_path.read_bytes()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"[warn] Failed to load screenshot {screenshot_path}: {e}")
        return None


def load_dom_html(dom_path: Path) -> Optional[str]:
    """Load DOM snapshot."""
    if not dom_path.exists():
        return None
    try:
        return dom_path.read_text(encoding='utf-8')[:5000]  # Limit to 5k chars
    except Exception as e:
        print(f"[warn] Failed to load DOM {dom_path}: {e}")
        return None


async def annotate_step_with_gemini(
    model: Any,
    goal: str,
    prev_screenshot_b64: Optional[str],
    curr_screenshot_b64: Optional[str],
    curr_dom: Optional[str],
    prev_action_desc: str = "none"
) -> Dict[str, Any]:
    """
    Use Gemini to infer what action was performed between prev and curr states.
    
    Returns:
        {
            "action_type": "click" | "type" | "scroll" | "select" | "navigate" | "wait",
            "target_element": "CSS selector or description",
            "input_value": "text typed" (for type actions),
            "reason": "Why this action was taken"
        }
    """
    
    prompt_parts = [
        "You are an expert in analyzing web automation traces.\n\n",
        f"User Goal: {goal}\n",
        f"Previous Action: {prev_action_desc}\n\n",
        "Below are two screenshots and the current page DOM.\n",
        "Analyze the visual and DOM differences to determine:\n",
        "1. What action was performed (click, type, scroll, select_dropdown, navigate, wait)\n",
        "2. Which element was targeted (CSS selector or description)\n",
        "3. If typing, what text was entered\n",
        "4. Why this action makes sense for the goal\n\n",
        "Return ONLY a JSON object with this structure:\n",
        "{\n",
        '  "action_type": "click | type | scroll | select_dropdown | navigate | wait",\n',
        '  "target_element": "CSS selector or element description",\n',
        '  "input_value": "typed text (if action_type=type, else empty)",\n',
        '  "reason": "Brief explanation of intent"\n',
        "}\n\n"
    ]
    
    # Add screenshots if available
    if prev_screenshot_b64:
        prompt_parts.append(f"Previous Screenshot (base64): {prev_screenshot_b64[:100]}...\n\n")
    if curr_screenshot_b64:
        prompt_parts.append(f"Current Screenshot (base64): {curr_screenshot_b64[:100]}...\n\n")
    
    # Add DOM snippet
    if curr_dom:
        prompt_parts.append(f"Current DOM (first 1000 chars):\n{curr_dom[:1000]}\n")
    
    prompt_text = "".join(prompt_parts)
    
    try:
        loop = asyncio.get_running_loop()
        
        def _call_gemini():
            time.sleep(2)  # Rate limiting
            return model.generate_content(prompt_text)
        
        resp = await loop.run_in_executor(None, _call_gemini)
        raw = resp.text or ""
        
        # Try to parse JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"[warn] Failed to parse Gemini response as JSON: {raw[:200]}")
            return {
                "action_type": "unknown",
                "target_element": "unknown",
                "input_value": "",
                "reason": "Failed to parse Gemini response"
            }
            
    except Exception as exc:
        print(f"[error] Gemini call failed: {exc}")
        return {
            "action_type": "unknown",
            "target_element": "unknown",
            "input_value": "",
            "reason": f"Error: {exc}"
        }


async def annotate_episode_async(
    raw_episode: Dict[str, Any],
    model: Any,
    base_path: Path
) -> Dict[str, Any]:
    """
    Annotate all steps in an episode by analyzing screenshots and DOM.
    
    Args:
        raw_episode: Episode JSON dict
        model: Gemini model instance
        base_path: Base path to find screenshots/DOM (e.g., data/raw/lazada/)
    """
    episode_id = raw_episode.get("episode_id", "unknown")
    goal = raw_episode.get("goal", "unknown goal")
    steps = raw_episode.get("steps", [])
    
    screenshot_base = base_path / "screenshots" / episode_id
    dom_base = base_path / "dom_snapshots" / episode_id
    
    prev_screenshot_b64 = None
    prev_action_desc = "Start of episode"
    
    for step in steps:
        step_num = step.get("step", 0)
        
        # Load current screenshot and DOM
        curr_screenshot_path = screenshot_base / f"step_{step_num}.png"
        curr_dom_path = dom_base / f"step_{step_num}.html"
        
        curr_screenshot_b64 = load_screenshot_base64(curr_screenshot_path)
        curr_dom = load_dom_html(curr_dom_path)
        
        # Call Gemini to infer action
        inferred = await annotate_step_with_gemini(
            model,
            goal,
            prev_screenshot_b64,
            curr_screenshot_b64,
            curr_dom,
            prev_action_desc
        )
        
        # Update step action with inferred data
        step["action"] = {
            "skill": inferred["action_type"],
            "selector": inferred["target_element"],
            "value": inferred["input_value"],
            "reason": inferred["reason"]
        }
        step["thought"] = inferred["reason"]
        
        # Update for next iteration
        prev_screenshot_b64 = curr_screenshot_b64
        prev_action_desc = f'{inferred["action_type"]} on {inferred["target_element"]}'
        
        print(f"[ok] Step {step_num}: {inferred['action_type']} - {inferred['reason'][:50]}")
    
    return raw_episode


def annotate_episode(
    raw_episode: Dict[str, Any],
    api_key: Optional[str] = None,
    base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for annotate_episode_async.
    
    Args:
        raw_episode: Episode JSON dict
        api_key: Gemini API key (or from env GEMINI_API_KEY)
        base_path: Base path to find assets (default: data/raw/lazada)
    """
    if genai is None:
        print("[error] google-generativeai not installed. Run: pip install google-generativeai")
        return raw_episode
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[error] No GEMINI_API_KEY found. Set env variable or pass api_key argument.")
        return raw_episode
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    
    # Determine base path from episode_id if not provided
    if base_path is None:
        # Try to infer from episode metadata or default to lazada
        platform = raw_episode.get("metadata", {}).get("platform", "lazada")
        base_path = Path(f"data/raw/{platform}")
    
    # Run async annotation
    return asyncio.run(annotate_episode_async(raw_episode, model, base_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a raw human_teleop episode with Gemini labels")
    parser.add_argument("input", help="Path to raw episode JSON")
    parser.add_argument("output", help="Destination JSON path")
    parser.add_argument("--base_path", help="Base path to find screenshots/DOM (default: auto-detect)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    
    base_path = Path(args.base_path) if args.base_path else None
    labeled = annotate_episode(data, base_path=base_path)
    
    Path(args.output).write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Annotated episode saved to {args.output}")


if __name__ == "__main__":
    main()
