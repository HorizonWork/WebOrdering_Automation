"""Batch annotator orchestrating Gemini calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.annotation.gemini_annotator import annotate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch annotate raw episodes")
    parser.add_argument("input_dir", help="Folder with raw episodes")
    parser.add_argument("output_dir", help="Folder to store annotated episodes")
    parser.add_argument("--base_path", help="Base path for screenshots (e.g. data/manual/shopee)")
    parser.add_argument("--api_keys", help="Comma-separated list of Gemini API keys (or path to file with keys)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = Path(args.base_path) if args.base_path else None
    
    # Load API keys
    api_keys = []
    if args.api_keys:
        if Path(args.api_keys).exists():
            api_keys = [k.strip() for k in Path(args.api_keys).read_text().splitlines() if k.strip()]
        else:
            api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    
    if not api_keys and not os.getenv("GEMINI_API_KEY"):
        print("[error] No API keys provided. Use --api_keys or set GEMINI_API_KEY env var.")
        return

    print(f"Loaded {len(api_keys)} API keys for rotation.")

    for idx, episode_file in enumerate(in_dir.glob("*.json")):
        # Rotate keys
        current_key = None
        if api_keys:
            current_key = api_keys[idx % len(api_keys)]
            
        print(f"Processing {episode_file.name} with key ...{current_key[-4:] if current_key else 'ENV'}")
        
        labeled = annotate_episode(
            json.loads(episode_file.read_text(encoding="utf-8")), 
            base_path=base_path,
            api_key=current_key
        )
        target = out_dir / f"{episode_file.stem}_labeled.json"
        target.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {episode_file.name} -> {target.name}")


if __name__ == "__main__":
    main()
