# -*- coding: utf-8 -*-
"""
Normalize manual episodes -> simple planner/controller JSONL.

- Scans input_dir recursively for *.json episodes.
- Keeps original thought/action/observation/page_state if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_manual_data(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episode_paths = sorted(input_path.rglob("*.json"))
    print(f"[info] Found {len(episode_paths)} episodes in {input_path}")

    planner_data: List[Dict[str, Any]] = []
    controller_data: List[Dict[str, Any]] = []

    for ep_path in episode_paths:
        try:
            episode = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] Cannot load {ep_path}: {exc}")
            continue

        goal = episode.get("goal", "")
        steps = episode.get("steps", []) or []

        for i, step in enumerate(steps, start=1):
            page_state = step.get("page_state") or {}
            observation = step.get("observation") or {}
            thought = step.get("thought") or ""
            action = step.get("action") or {}
            teacher = step.get("teacher_labels") or {}

            # Planner sample (fallback if planner missing)
            planner_out = (
                (step.get("planner") or {}).get("next_plan_step")
                or (teacher.get("planner") or {}).get("next_plan_step")
                or {
                    "step_id": f"STEP_{i}",
                    "type": page_state.get("page_type", "unknown"),
                    "description": thought or action.get("skill", ""),
                }
            )

            planner_data.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "step": i,
                    "input": f"Goal: {goal}\nPage type: {page_state.get('page_type', 'unknown')}\nCurrent Step: {i}",
                    "output": json.dumps(planner_out, ensure_ascii=False),
                }
            )

            # Controller sample (fallback to executed action)
            controller_out = (
                (step.get("controller") or {}).get("chosen_action")
                or (teacher.get("controller") or {}).get("chosen_action")
                or action
            )

            ctrl_plan = (step.get("planner") or {}).get("next_plan_step", {})
            ctrl_desc = ctrl_plan.get("description", "")
            obs_title = observation.get("title", "")

            controller_data.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "step": i,
                    "input": f"Plan: {ctrl_desc}\nObservation: {obs_title}\nPage type: {page_state.get('page_type', 'unknown')}",
                    "output": json.dumps(controller_out, ensure_ascii=False),
                }
            )

    # Save JSONL
    planner_out = output_path / "planner_train.jsonl"
    with planner_out.open("w", encoding="utf-8") as f:
        for item in planner_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    controller_out = output_path / "controller_train.jsonl"
    with controller_out.open("w", encoding="utf-8") as f:
        for item in controller_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[ok] Processed {len(planner_data)} planner samples -> {planner_out}")
    print(f"[ok] Processed {len(controller_data)} controller samples -> {controller_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize manual episodes to planner/controller JSONL")
    ap.add_argument("--input_dir", default="data/manual/episodes", help="Root dir containing episode JSON files (recursive).")
    ap.add_argument("--output_dir", default="data/processed", help="Output directory for processed JSONL.")
    args = ap.parse_args()
    normalize_manual_data(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
