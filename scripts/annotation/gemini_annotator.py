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


async def annotate_full_episode(
    model: Any,
    goal: str,
    steps_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Annotate the entire episode in ONE LLM call.
    steps_data: List of dicts with {step_num, screenshot_b64, dom_text}
    """
    intro_parts = [
        "Analyze the following sequence of web interactions to achieve the Goal.\n",
        f"Goal: {goal}\n\n",
        "For EACH step, you are given the Original Action (recorded) and the Visual/DOM context.\n",
        "Explain WHY the model would perform this action next (model's reasoning, not user speech).\n",
        "Write thoughts in Vietnamese, ngôi thứ nhất của mô hình (e.g. \"Tôi cần mở kết quả tìm kiếm...\").\n",
        "Return a JSON LIST of objects, each with exactly these keys:\n",
        '  - step: integer (matching the step number)\n',
        '  - thought: string (model reasoning, Vietnamese, ngôi Tôi)\n',
        '  - description: string (short action summary, e.g. "Click vào Anker Flagship Store")\n\n',
        "Example Output:\n",
        "[\n",
        "  {\n",
        '    \"step\": 1,\n',
        '    \"thought\": \"User clicked search to find the product\",\n',
        '    \"description\": \"Search for item\"\n',
        "  }\n",
        "]\n\n",
        "Use Vietnamese in your explanations.\n"
    ]

    final_content: List[Any] = ["".join(intro_parts)]
    
    for step in steps_data:
        s_num = step["step_num"]
        orig_act = step.get("original_action", "unknown")
        final_content.append(f"\n--- Step {s_num} ---\nOriginal Action: {orig_act}\n")
        
        # Add Image (as binary part so Gemini can see it)
        if step.get("screenshot_b64"):
            img_bytes = base64.b64decode(step["screenshot_b64"])
            final_content.append({"mime_type": "image/png", "data": img_bytes})
        
        # Add DOM snippet
        if step.get("dom_text"):
            final_content.append(f"DOM:\n{step['dom_text'][:800]}\n")

    final_content.append("\nReturn ONLY the JSON list now:")
    
    # Call Model
    try:
        # Use the running loop directly
        loop = asyncio.get_running_loop()
        
        def _call():
            return model.generate_content(final_content)
        
        resp = await loop.run_in_executor(None, _call)
        raw = resp.text
        
        print(f"  [debug] Gemini Raw Response (first 100 chars): {raw[:100]}...")
        
        # Parse JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(raw)
        print(f"  [debug] Parsed {len(parsed)} items.")
        return parsed
        
    except Exception as e:
        print(f"[error] Batch annotation failed: {e}")
        return []

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
    prev_action_desc: str = "none",
    orig_action_desc: str = "unknown action"
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
        "Analyze the web interaction trace below.\n",
        f"Goal: {goal}\n",
        f"Last Action: {prev_action_desc}\n\n",
        f"Original recorded action: {orig_action_desc}\n",
        "Your output MUST stay consistent with the recorded action. If unsure, mirror the recorded action.\n",
        "If the recorded action is human_action, treat the raw_input as an instruction and reason how to satisfy it using the current screen. If it requires user choice (e.g., chọn màu), mention needing to pick or ask the user.\n",
        "You are the model thinking about YOUR next move (not the user’s words).\n",
        "Write the reason as the model’s chain-of-thought in Vietnamese, e.g. \"Tôi đã mở trang sản phẩm, cần chọn màu...\".\n",
        "Return JSON ONLY:\n",
        "{\n",
        '  "action_type": "click" | "type" | "scroll" | "wait" | "request_help",\n',
        '  "target_element": "CSS selector or element description",\n',
        '  "input_value": "text if typing, else empty",\n',
        '  "reason": "Model reasoning about next action (chain-of-thought, Vietnamese, ngôi Tôi)",\n',
        '  "description": "Concrete action summary in Vietnamese, e.g. \'Click vào Anker Flagship Store\'"\n',
        "}\n"
    ]
    
    final_content: List[Any] = ["".join(prompt_parts)]

    # Add screenshots if available (as binary parts)
    if prev_screenshot_b64:
        final_content.append("Previous Screenshot:")
        final_content.append({"mime_type": "image/png", "data": base64.b64decode(prev_screenshot_b64)})
    if curr_screenshot_b64:
        final_content.append("Current Screenshot:")
        final_content.append({"mime_type": "image/png", "data": base64.b64decode(curr_screenshot_b64)})
    
    # Add DOM snippet (Distilled)
    if curr_dom:
        # If curr_dom is a string (legacy), use it
        if isinstance(curr_dom, str):
            final_content.append(f"Current DOM (snippet):\n{curr_dom[:1000]}\n")
        # If it's a dict (distilled state), format it
        elif isinstance(curr_dom, dict):
            elements = curr_dom.get("elements", [])
            dom_str = "Interactive Elements:\n"
            for el in elements[:50]: # Top 50 elements
                text = el.get("text", "").strip()
                selector = el.get("selector", "")
                if text or selector:
                    dom_str += f"- [{el.get('tag')}] {text} (Selector: {selector})\n"
            final_content.append(f"Current DOM State:\n{dom_str}\n")
    else:
        final_content.append("Current DOM State: (not available)")
    
    try:
        loop = asyncio.get_running_loop()
        
        def _call_gemini():
            time.sleep(2)  # Rate limiting
            return model.generate_content(final_content)
        
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
    base_path: Path,
    human_only: bool = True
) -> Dict[str, Any]:
    """
    Annotate all steps in an episode by analyzing screenshots and DOM.
    
    Args:
        raw_episode: Episode JSON dict
        model: Gemini model instance
        base_path: Base path to find screenshots/DOM (e.g., data/raw/lazada/)
        human_only: If True, only annotate steps marked as human teleop, leave others untouched
    """
    episode_id = raw_episode.get("episode_id", "unknown")
    goal = raw_episode.get("goal", "unknown goal")
    steps = raw_episode.get("steps", [])
    
    screenshot_base = base_path / "screenshots" / episode_id
    dom_base = base_path / "dom_snapshots" / episode_id
    
    prev_screenshot_b64 = None
    prev_action_desc = "Start of episode"
    
    def summarize_recorded_action(action: Dict[str, Any]) -> str:
        skill = action.get("skill")
        params = (action.get("params") or {})
        selector = params.get("selector") or action.get("selector") or params.get("element")
        text = params.get("text") or params.get("value") or params.get("raw_input")
        
        if skill in ("fill", "type"):
            target = selector or "ô nhập"
            return f"Điền \"{text}\" vào {target}" if text else f"Điền vào {target}"
        if skill == "click":
            return f"Click vào {selector or 'nút/điểm chọn'}"
        if skill == "scroll":
            return "Cuộn trang"
        if skill == "request_help":
            return "Gọi hỗ trợ (captcha/khó khăn)"
        if skill == "human_action":
            return f"Hành động người: {text}" if text else "Hành động người"
        return skill or "Hành động"

    for step in steps:
        step_num = step.get("step", 0)
        
        # Load current screenshot and DOM
        curr_screenshot_path = screenshot_base / f"step_{step_num}.png"
        
        # Use distilled DOM from step data if available, else fallback to HTML file
        curr_dom = step.get("page_state", {}).get("dom_state")
        if not curr_dom:
             curr_dom_path = dom_base / f"step_{step_num}.html"
             curr_dom = load_dom_html(curr_dom_path)
        
        curr_screenshot_b64 = load_screenshot_base64(curr_screenshot_path)

        # Determine whether this step should be annotated
        action_info = step.get("action", {}) or {}
        action_skill = action_info.get("skill")
        thought_text = step.get("thought", "")
        is_human = action_skill == "human_action" or "(human teleop)" in thought_text.lower()
        should_annotate = (not human_only) or is_human or action_skill == "request_help"
        recorded_summary = summarize_recorded_action(action_info)
        recorded_params = (action_info.get("params") or {})
        
        if should_annotate:
            inferred = await annotate_step_with_gemini(
                model,
                goal,
                prev_screenshot_b64,
                curr_screenshot_b64,
                curr_dom,
                prev_action_desc,
                orig_action_desc=recorded_summary
            )
            
            # Only enrich metadata; do NOT overwrite recorded action
            inferred_reason = inferred.get("reason", "") or ""
            inferred_desc = inferred.get("description", "") or inferred_reason

            # Special handling for request_help actions (e.g., captcha)
            if action_skill == "request_help":
                default_help_text = "Tôi gặp captcha hoặc tình huống khó, cần người dùng hỗ trợ."
                inferred_reason = inferred_reason or default_help_text
                inferred_desc = inferred_desc or default_help_text

            # If model action disagrees with recorded skill, fall back to recorded summary
            inferred_skill = inferred.get("action_type")
            if inferred_skill and action_skill and inferred_skill != action_skill:
                inferred_desc = recorded_summary
                inferred_reason = inferred_reason or recorded_summary

            # Sanitize mismatched content (e.g., click but reasoning says 'nhập')
            def sanitize_mismatch(skill: Optional[str], reason: str, desc: str) -> (str, str):
                skill = skill or ""
                reason_l = reason.lower()
                desc_l = desc.lower()
                
                def mention_type(text: str) -> bool:
                    return any(k in text for k in ["nhập", "gõ", "type", "fill", "input text"])
                
                def mention_click(text: str) -> bool:
                    return any(k in text for k in ["click", "nhấp", "bấm"])
                
                if skill == "click":
                    if mention_type(reason_l) or mention_type(desc_l):
                        # Click action should not mention typing
                        return recorded_summary, recorded_summary
                if skill in ("fill", "type"):
                    text_val = recorded_params.get("text") or recorded_params.get("value")
                    selector_val = recorded_params.get("selector") or recorded_params.get("element") or "ô nhập"
                    fill_phrase = f'Nhập "{text_val}" vào {selector_val}' if text_val else f"Nhập vào {selector_val}"
                    if text_val and text_val.lower() not in reason_l and text_val.lower() not in desc_l:
                        return fill_phrase, fill_phrase
                return reason, desc

            inferred_reason, inferred_desc = sanitize_mismatch(action_skill, inferred_reason, inferred_desc)

            # Enforce description coherence: always anchor to recorded action
            final_desc = recorded_summary
            # Prefer model description only if same skill and non-empty
            if inferred_skill == action_skill and inferred_desc:
                final_desc = inferred_desc

            # Thought must also stay consistent; if empty or skill mismatch, use recorded summary
            final_reason = inferred_reason if (inferred_reason and (inferred_skill == action_skill or not inferred_skill)) else recorded_summary
            if not final_reason:
                final_reason = recorded_summary

            step["thought"] = final_reason
            step["description"] = final_desc
            
            print(f"[ok] Step {step_num}: {inferred.get('action_type')} - {inferred.get('reason', '')[:50]}")
            prev_action_desc = f'{inferred.get("action_type")} on {inferred.get("target_element")}'
        else:
            # Keep original action; still advance prev_action_desc for context
            selector = action_info.get("selector") or action_info.get("params", {}).get("selector")
            prev_action_desc = f'{action_skill} on {selector}'
        
        # Update for next iteration
        prev_screenshot_b64 = curr_screenshot_b64
    
    return raw_episode


def annotate_episode(
    episode_data: Dict[str, Any],
    base_path: Path = None,
    api_key: str = None,
    human_only: bool = True
) -> Dict[str, Any]:
    """
    Annotate a single episode with Gemini.
    """
    if genai is None:
        print("[error] google-generativeai not installed. Run: pip install google-generativeai")
        return episode_data
        
    # Configure API key if provided
    if api_key:
        genai.configure(api_key=api_key)
    elif not os.getenv("GEMINI_API_KEY"):
        print("[error] No GEMINI_API_KEY found. Set env variable or pass api_key argument.")
        return episode_data
    
    # Configure Gemini
    # If api_key was provided, it's already configured. If not, it will use the env var.
    # We need to ensure genai.configure is called with the *actual* key being used.
    final_api_key = api_key or os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=final_api_key)
    
    model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config={"response_mime_type": "application/json"}
    )
    
    # Determine base path from episode_id if not provided
    if base_path is None:
        # Try to infer from episode metadata or default to lazada
        platform = episode_data.get("metadata", {}).get("platform", "lazada")
        base_path = Path(f"data/raw/{platform}")
    
    # Run async annotation
    return asyncio.run(annotate_episode_async(episode_data, model, base_path, human_only=human_only))


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a raw human_teleop episode with Gemini labels")
    parser.add_argument("input", help="Path to raw episode JSON")
    parser.add_argument("output", help="Destination JSON path")
    parser.add_argument("--base_path", help="Base path to find screenshots/DOM (default: auto-detect)")
    parser.add_argument(
        "--all_steps",
        action="store_true",
        help="Annotate every step (default: only steps marked human teleop)"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    
    base_path = Path(args.base_path) if args.base_path else None
    labeled = annotate_episode(data, base_path=base_path, human_only=not args.all_steps)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Annotated episode saved to {args.output}")


if __name__ == "__main__":
    main()
