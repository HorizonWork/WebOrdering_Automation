
import argparse
import json
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    import google.generativeai as genai
    from google.api_core import exceptions
except ImportError:
    print("Please install google-generativeai")
    sys.exit(1)
# "gemini-2.0-flash"
from scripts.annotation.gemini_annotator import annotate_step_with_gemini, annotate_full_episode, load_screenshot_base64

class KeyManager:
    def __init__(self, api_keys: List[str], max_requests_per_minute: int = 10):
        self.api_keys = api_keys
        self.current_index = 0
        self.models = ["gemini-2.5-flash-lite","gemini-2.0-flash-lite"] # Fallback models
        self.model_index = 0
        self.max_requests_per_minute = max_requests_per_minute
        # Track recent request timestamps per key to rate-limit locally
        self.request_times = {i: [] for i in range(len(api_keys))}
        
    def get_client(self):
        if not self.api_keys:
            raise ValueError("No API keys provided")
        
        key = self.api_keys[self.current_index]
        genai.configure(api_key=key)
        
        model_name = self.models[self.model_index]
        return genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})

    def rotate_key(self):
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        print(f"[system] Rotating to API Key #{self.current_index + 1}")
        # Reset local window counter for the new key
        self.request_times.setdefault(self.current_index, [])
        
    def rotate_model(self):
        self.model_index = (self.model_index + 1) % len(self.models)
        print(f"[system] Switching to Model: {self.models[self.model_index]}")

    def record_request(self):
        """Record a request and rotate if exceeding local per-minute threshold."""
        now = time.time()
        times = self.request_times.setdefault(self.current_index, [])
        times.append(now)
        # Keep only last 60s
        self.request_times[self.current_index] = [t for t in times if now - t < 60]
        if len(self.request_times[self.current_index]) >= self.max_requests_per_minute:
            print(f"[system] Local rate limit hit ({len(self.request_times[self.current_index])} req/min). Rotating key.")
            self.rotate_key()

def chunk_steps(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i+size] for i in range(0, len(items), size)]


async def process_episode(
    episode_path: Path, 
    output_path: Path, 
    key_manager: KeyManager,
    base_path: Path,
    human_only: bool = True,
    chunk_size: int = 5
):
    if output_path.exists():
        print(f"[skip] {episode_path.name} already labeled.")
        return

    try:
        data = json.loads(episode_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[error] Failed to read {episode_path.name}: {e}")
        return

    episode_id = data.get("episode_id")
    steps = data.get("steps", [])
    goal = data.get("goal", "")
    
    screenshot_base = base_path / "screenshots" / episode_id
    
    print(f"Processing {episode_path.name} ({len(steps)} steps)...")
    
    prev_screenshot_b64 = None
    prev_action_desc = "Start of episode"
    
    from scripts.annotation.gemini_annotator import annotate_full_episode, load_screenshot_base64

    # Prepare batch data
    steps_data = []
    steps_to_label = []
    
    for step in steps:
        # HYBRID: Skip if already labeled OR not human when human_only is True
        existing_thought = step.get("thought")
        action_skill = (step.get("action") or {}).get("skill")
        has_teleop_flag = "(human teleop)" in (existing_thought or "").lower()

        # Only label teleop-style steps when human_only is True
        if human_only and not (action_skill in ("human_action", "request_help") or has_teleop_flag):
            continue

        # Skip if already labeled (non-teleop content present)
        if existing_thought and "(human teleop)" not in (existing_thought or "").lower() and len(existing_thought) > 10:
            continue
            
        step_num = step.get("step")
        curr_screenshot_path = screenshot_base / f"step_{step_num}.png"
        curr_screenshot_b64 = load_screenshot_base64(curr_screenshot_path)
        curr_dom = step.get("dom_text")
        
        # Format original action for context
        orig_act = step.get("action", {})
        if isinstance(orig_act, dict):
            act_str = f"{orig_act.get('skill')} on {orig_act.get('selector') or orig_act.get('params', {}).get('selector')}"
            if orig_act.get('value') or orig_act.get('text'):
                act_str += f" value='{orig_act.get('value') or orig_act.get('text')}'"
        else:
            act_str = str(orig_act)

        steps_data.append({
            "step_num": step_num,
            "screenshot_b64": curr_screenshot_b64,
            "dom_text": curr_dom,
            "original_action": act_str
        })
        steps_to_label.append(step)
        
    if not steps_data:
        print(f"  [skip] All steps already labeled.")
        return

    print(f"  Sending batch of {len(steps_data)} steps to Gemini...")
    
    # Process in smaller chunks to reduce failure impact
    step_chunks = chunk_steps(steps_data, chunk_size)
    for chunk_idx, chunk in enumerate(step_chunks, 1):
        max_retries = 3
        pending = list(chunk)
        for attempt in range(max_retries):
            try:
                model = key_manager.get_client()
                await asyncio.sleep(1)
                
                results = await annotate_full_episode(model, goal, pending)
                key_manager.record_request()  # count this request
                
                success_count = 0
                returned_steps = set()
                for res in results:
                    s_idx = res.get("step")
                    returned_steps.add(s_idx)
                    target_step = next((s for s in steps_to_label if s.get("step") == s_idx), None)
                    if target_step:
                        target_step["thought"] = res.get("thought", "")
                        target_step["description"] = res.get("description", "")
                        success_count += 1
                
                pending = [s for s in pending if s.get("step") not in returned_steps]
                if pending:
                    missing_ids = [s.get("step") for s in pending]
                    print(f"  [warn] Chunk {chunk_idx}/{len(step_chunks)} missing {len(pending)} steps: {missing_ids}. Retrying (attempt {attempt+1}/{max_retries}).")
                    key_manager.rotate_key()
                    key_manager.rotate_model()
                    await asyncio.sleep(5)
                    continue
                
                if success_count < len(chunk):
                    print(f"  [warn] Chunk {chunk_idx}/{len(step_chunks)} labeled {success_count}/{len(chunk)} steps.")
                else:
                    print(f"  [ok] Chunk {chunk_idx}/{len(step_chunks)} labeled {success_count}/{len(chunk)} steps.")
                break
                
            except exceptions.ResourceExhausted:
                print(f"  [429] Quota exceeded. Rotating key/model and retrying chunk {chunk_idx} (attempt {attempt+1}/{max_retries})...")
                key_manager.rotate_key()
                key_manager.rotate_model()
                await asyncio.sleep(5)
            except Exception as e:
                print(f"  [error] Chunk {chunk_idx} failed: {e}")
                break
        
    # Save result
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Saved {output_path.name}")
    # Rotate key after each episode to spread quota usage
    key_manager.rotate_key()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--base_path")
    parser.add_argument("--api_keys")
    parser.add_argument("--all_steps", action="store_true", help="Annotate every step (default: only human teleop steps)")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load keys
    keys = []
    if args.api_keys:
        if Path(args.api_keys).exists():
            keys = [k.strip() for k in Path(args.api_keys).read_text().splitlines() if k.strip()]
        else:
            keys = args.api_keys.split(",")
            
    if not keys:
        print("No keys provided")
        return

    manager = KeyManager(keys)
    base_path = Path(args.base_path)
    
    tasks = []
    # Process sequentially to avoid hammering rate limits too hard even with rotation
    # But we could do small batches if we had many keys. For now, sequential is safer for free tier.
    for ep_file in in_dir.glob("*.json"):
        out_file = out_dir / f"{ep_file.stem}_labeled.json"
        await process_episode(ep_file, out_file, manager, base_path, human_only=not args.all_steps)

if __name__ == "__main__":
    asyncio.run(main())
