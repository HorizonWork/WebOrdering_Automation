# -*- coding: utf-8 -*-
"""
Generate schema-aligned samples for Planner and Controller from collected episodes.

This script reads `ep_*.json` files produced by `scripts/collect_trajectories.py`
and emits two JSONL files:

  - `data/schema_samples/planner_samples.jsonl`
  - `data/schema_samples/controller_samples.jsonl`

Each line is a training/example row:
  - Planner:   {episode_id, step, planner_input, planner_output, ...}
  - Controller:{episode_id, step, controller_input, controller_output, ...}

If an episode step contains `teacher_labels` (from Gemini Teacher),
real labels are used to fill `planner_output` / `controller_output`.
Otherwise, sensible placeholders are used so you can still inspect schema
without hand-labeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_episodes(episodes_dir: Path, max_episodes: int | None = None) -> List[Dict[str, Any]]:
    """Load ep_*.json episodes from a directory."""
    files = sorted(p for p in episodes_dir.glob("*.json") if p.is_file())
    if max_episodes is not None:
        files = files[:max_episodes]

    episodes: List[Dict[str, Any]] = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                episodes.append(data)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Cannot read episode {p}: {exc}")
    return episodes


def infer_high_level_filters(dom_state: Dict[str, Any]) -> Tuple[bool, bool]:
    """Heuristics: detect presence of price / official-store filters."""
    filters = dom_state.get("filters") or []
    has_price = False
    has_official = False

    for f in filters:
        label = (f.get("label") or "").lower()
        fid = (f.get("id") or "").lower()
        text = f"{fid} {label}"
        if any(k in text for k in ["price", "giá", "khoảng giá"]):
            has_price = True
        if any(k in text for k in ["mall", "chính hãng", "official"]):
            has_official = True

    return has_price, has_official


def build_history_summary(steps: List[Dict[str, Any]], upto_step_num: int, window: int = 3) -> List[str]:
    """Create short textual history summary for Planner."""
    hist: List[str] = []
    # steps are 1-based; we select steps with step < upto_step_num
    for s in steps:
        step_num = int(s.get("step") or 0)
        if step_num <= 0 or step_num >= upto_step_num:
            continue
        if upto_step_num - step_num > window:
            continue
        action = s.get("action") or {}
        skill = action.get("skill")
        params = action.get("params")
        if skill:
            hist.append(f"Step {step_num}: {skill}({params})")
    return hist


def build_short_history_for_controller(steps: List[Dict[str, Any]], upto_step_num: int, window: int = 3) -> List[str]:
    """Create short textual history for Controller."""
    hist: List[str] = []
    for s in steps:
        step_num = int(s.get("step") or 0)
        if step_num <= 0 or step_num >= upto_step_num:
            continue
        if upto_step_num - step_num > window:
            continue
        action = s.get("action") or {}
        skill = action.get("skill")
        params = action.get("params")
        if skill:
            hist.append(f"Step {step_num}: {skill}({params})")
    return hist


def heuristic_current_step(page_type: str) -> str:
    """Very rough mapping from page_type to coarse-grained progress."""
    page_type_l = (page_type or "").lower()
    if "search" in page_type_l:
        return "SEARCH"
    if "product" in page_type_l:
        return "PRODUCT_SELECTED"
    if "cart" in page_type_l:
        return "IN_CART"
    if "checkout" in page_type_l:
        return "CHECKOUT"
    return "UNKNOWN"


def build_available_actions_flat_from_filters(dom_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a flat list of candidate DSL actions from dom_state.filters[*].actions.

    This is only for schema/debugging; real DSL mapping will later come from
    hand-crafted configs per site.
    """
    filters = dom_state.get("filters") or []
    actions: Dict[str, Dict[str, Any]] = {}

    for f in filters:
        for aid in f.get("actions") or []:
            if not isinstance(aid, str):
                continue
            if aid in actions:
                continue
            aid_lower = aid.lower()
            if "input" in aid_lower or "text" in aid_lower:
                atype = "FILL"
            elif any(k in aid_lower for k in ["btn", "button", "checkbox", "check", "click", "apply"]):
                atype = "CLICK"
            else:
                atype = "CLICK"

            actions[aid] = {
                "id": aid,
                "type": atype,
                "description": f"TODO: describe {aid}",
                "dom_selector": None,
                "vision_ref": None,
            }

    return list(actions.values())


def build_last_action_result(step: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight last_action_result from step['result']."""
    res = step.get("result") or {}
    status = (res.get("status") or "").lower()
    success = status == "success"
    err = None if success else res.get("message")
    return {"success": bool(success), "error": err}


def extract_teacher_labels(step: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract planner/controller labels from teacher_labels if present.

    Expected structure (per collect_trajectories.Teacher prompt):
    {
      "planner": { "next_plan_step": {...}, ... },
      "controller": {
          "chosen_action": {...},
          "reason": "..."
      }
    }
    """
    labels = step.get("teacher_labels") or {}
    planner = labels.get("planner") or {}
    controller = labels.get("controller") or {}
    return planner, controller


def build_planner_sample(ep: Dict[str, Any], step: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create one PlannerInput + PlannerOutput sample."""
    goal_text = ep.get("goal", "")
    page_state = step.get("page_state") or {}
    dom_state = page_state.get("dom_state") or {}
    page_type = page_state.get("page_type", "unknown")
    has_price, has_official = infer_high_level_filters(dom_state)

    step_num = int(step.get("step") or 0)

    planner_input = {
        "goal": {
            "raw_user_goal": goal_text,
            "constraints": {
                # leave empty; can be filled by separate goal parser later
            },
        },
        "high_level_state": {
            "page_type": page_type,
            "current_step": heuristic_current_step(page_type),
            "has_price_filter": has_price,
            "has_official_store_filter": has_official,
        },
        "history_summary": build_history_summary(steps, step_num),
        "visual_summary": "",
        "detected_obstacles": [],
        "available_high_level_actions": [
            "SEARCH_PRODUCT",
            "APPLY_FILTER",
            "SELECT_PRODUCT",
            "GO_TO_CART",
            "GO_TO_CHECKOUT",
            "FILL_CHECKOUT_INFO",
            "REVIEW_ORDER",
            "TERMINATE",
        ],
    }

    # Default placeholder PlannerOutput
    planner_output_placeholder = {
        "plan_version": 1,
        "overall_strategy": "TODO: mô tả chiến lược tổng quan.",
        "next_plan_step": {
            "step_id": "STEP_TODO",
            "type": "SEARCH_PRODUCT",
            "description": "TODO: mô tả bước tiếp theo.",
            "constraints": {},
        },
    }

    planner_labels, _ = extract_teacher_labels(step)
    teacher_next = planner_labels.get("next_plan_step") if isinstance(planner_labels, dict) else None

    if isinstance(teacher_next, dict):
        planner_output = {
            "plan_version": 1,
            "overall_strategy": planner_labels.get("overall_strategy", ""),
            "next_plan_step": teacher_next,
        }
    else:
        planner_output = planner_output_placeholder

    return {
        "episode_id": ep.get("episode_id"),
        "step": step.get("step"),
        "planner_input": planner_input,
        "planner_output": planner_output,
        "planner_output_placeholder": planner_output_placeholder,
    }


def build_controller_sample(ep: Dict[str, Any], step: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create one ControllerInput + ControllerOutput sample."""
    goal_text = ep.get("goal", "")
    goal_summary = goal_text[:120]

    page_state_full = step.get("page_state") or {}
    page_type = page_state_full.get("page_type", "unknown")
    dom_state = page_state_full.get("dom_state") or {}
    vision_state = page_state_full.get("vision_state") or {}

    has_price, has_official = infer_high_level_filters(dom_state)

    # Heuristic current_plan_step.type for debug
    if has_price or has_official:
        step_type = "APPLY_FILTER"
        desc = "Lọc theo các filter đã có trên trang (tạm thời)."
    else:
        step_type = "SEARCH_PRODUCT"
        desc = "Tiếp tục thao tác tìm kiếm / duyệt sản phẩm (tạm thời)."

    step_num = int(step.get("step") or 0)

    controller_input = {
        "goal": {
            "summary": goal_summary,
        },
        "current_plan_step": {
            "type": step_type,
            "description": desc,
            "constraints": {},
        },
        "page_state": {
            "page_type": page_type,
            "dom_state": dom_state,
            "vision_state": vision_state,
        },
        "available_actions_flat": build_available_actions_flat_from_filters(dom_state),
        "last_action_result": build_last_action_result(step),
        "short_history": build_short_history_for_controller(steps, step_num),
    }

    controller_output_placeholder = {
        "chosen_action": {
            "action_id": "",
            "type": "",
            "reason": "TODO: lý do chọn action này.",
        },
        "reason": "TODO: giải thích vì sao action giúp tiến gần current_plan_step.",
    }

    _, controller_labels = extract_teacher_labels(step)
    chosen = controller_labels.get("chosen_action") if isinstance(controller_labels, dict) else None

    if isinstance(chosen, dict):
        controller_output = {
            "chosen_action": chosen,
            "reason": controller_labels.get("reason", chosen.get("reason", "")),
        }
    else:
        controller_output = controller_output_placeholder

    return {
        "episode_id": ep.get("episode_id"),
        "step": step.get("step"),
        "controller_input": controller_input,
        "controller_output": controller_output,
        "controller_output_placeholder": controller_output_placeholder,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Planner/Controller schema samples from collected episodes."
    )
    ap.add_argument(
        "--episodes_dir",
        default="data/trajectories/shopping/episodes",
        help="Folder containing ep_*.json produced by collect_trajectories.py.",
    )
    ap.add_argument(
        "--out_dir",
        default="data/schema_samples",
        help="Output folder for planner_samples.jsonl and controller_samples.jsonl.",
    )
    ap.add_argument(
        "--max_episodes",
        type=int,
        default=5,
        help="Max episodes to load (0 or negative = no limit).",
    )
    ap.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=4,
        help="Max steps per episode to sample (0 or negative = all).",
    )
    args = ap.parse_args()

    episodes_dir = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)

    max_eps = args.max_episodes if args.max_episodes > 0 else None
    episodes = load_episodes(episodes_dir, max_eps)
    if not episodes:
        print(f"[error] No episode JSON found in {episodes_dir}")
        return

    planner_rows: List[Dict[str, Any]] = []
    controller_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        steps = ep.get("steps") or []
        if args.max_steps_per_episode > 0:
            steps_subset = steps[: args.max_steps_per_episode]
        else:
            steps_subset = steps

        for s in steps_subset:
            if not s.get("page_state"):
                continue
            planner_rows.append(build_planner_sample(ep, s, steps))
            controller_rows.append(build_controller_sample(ep, s, steps))

    if not planner_rows:
        print("[warn] No Planner samples generated (no steps with page_state).")
    if not controller_rows:
        print("[warn] No Controller samples generated (no steps with page_state).")

    write_jsonl(out_dir / "planner_samples.jsonl", planner_rows)
    write_jsonl(out_dir / "controller_samples.jsonl", controller_rows)

    print(f"[ok] Wrote {len(planner_rows)} planner_samples to {out_dir / 'planner_samples.jsonl'}")
    print(f"[ok] Wrote {len(controller_rows)} controller_samples to {out_dir / 'controller_samples.jsonl'}")


if __name__ == "__main__":
    main()

