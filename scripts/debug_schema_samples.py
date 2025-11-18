# -*- coding: utf-8 -*-
"""
Debug / bootstrap schema samples for Planner & Controller.

Mục tiêu:
    - Đọc các episode đã thu thập từ scripts/collect_trajectories.py.
    - Với một số bước/step tiêu biểu, sinh ra:
        * PlannerInput skeleton (theo config/schemas/planner_input.schema.json).
        * ControllerInput skeleton (theo config/schemas/controller_input.schema.json).
    - Ghi ra 2 file JSONL để bạn mở trong IDE và chỉnh tay:
        * data/schema_samples/planner_samples.jsonl
        * data/schema_samples/controller_samples.jsonl

Cách chạy (ví dụ):

    python scripts/debug_schema_samples.py ^
        --episodes_dir data/trajectories/shopping/episodes ^
        --out_dir data/schema_samples ^
        --max_episodes 5 ^
        --max_steps_per_episode 4

Sau đó:
    - Mở 2 file JSONL output, kiểm tra xem schema có “đủ chỗ chứa” cho các case thực tế.
    - Bổ sung / chỉnh tay các trường còn thiếu (constraints, description chi tiết, v.v.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_episodes(episodes_dir: Path, max_episodes: int | None = None) -> List[Dict[str, Any]]:
    files = sorted(p for p in episodes_dir.glob("*.json") if p.is_file())
    if max_episodes is not None:
        files = files[:max_episodes]

    episodes: List[Dict[str, Any]] = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                episodes.append(data)
        except Exception as exc:
            print(f"[warn] Không đọc được episode {p}: {exc}")
    return episodes


def infer_high_level_filters(dom_state: Dict[str, Any]) -> Tuple[bool, bool]:
    """Heuristics: có filter giá / filter official store hay không."""
    filters = dom_state.get("filters") or []
    has_price = False
    has_official = False
    for f in filters:
        label = (f.get("label") or "").lower()
        fid = (f.get("id") or "").lower()
        text = fid + " " + label
        if any(k in text for k in ["price", "giá", "khoảng giá"]):
            has_price = True
        if any(k in text for k in ["mall", "chính hãng", "official"]):
            has_official = True
    return has_price, has_official


def build_history_summary(steps: List[Dict[str, Any]], upto_idx: int, window: int = 3) -> List[str]:
    """Tạo history_summary dạng text ngắn gọn cho Planner."""
    start = max(0, upto_idx - window)
    hist: List[str] = []
    for s in steps[start:upto_idx]:
        step_num = s.get("step")
        action = s.get("action") or {}
        skill = action.get("skill")
        params = action.get("params")
        if skill:
            hist.append(f"Step {step_num}: {skill}({params})")
    return hist


def build_short_history_for_controller(steps: List[Dict[str, Any]], upto_idx: int, window: int = 3) -> List[str]:
    """Tạo short_history text cho Controller."""
    start = max(0, upto_idx - window)
    hist: List[str] = []
    for s in steps[start:upto_idx]:
        step_num = s.get("step")
        action = s.get("action") or {}
        skill = action.get("skill")
        params = action.get("params")
        if skill:
            hist.append(f"Step {step_num}: {skill}({params})")
    return hist


def heuristic_current_step(page_type: str) -> str:
    """Đoán current_step đơn giản từ page_type (chỉ để debug, không dùng train)."""
    page_type = (page_type or "").lower()
    if "search" in page_type:
        return "SEARCH"
    if "product" in page_type:
        return "PRODUCT_SELECTED"
    if "cart" in page_type:
        return "IN_CART"
    if "checkout" in page_type:
        return "CHECKOUT"
    return "UNKNOWN"


def build_available_actions_flat_from_filters(dom_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Sinh action skeleton từ dom_state.filters[*].actions.
    - Chỉ để debug schema, không phải DSL chính thức.
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
    res = step.get("result") or {}
    status = (res.get("status") or "").lower()
    success = status == "success"
    err = None if success else res.get("message")
    return {"success": bool(success), "error": err}


def build_planner_sample(ep: Dict[str, Any], step: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tạo 1 sample PlannerInput skeleton + placeholder PlannerOutput."""
    goal_text = ep.get("goal", "")
    page_state = step.get("page_state") or {}
    dom_state = page_state.get("dom_state") or {}
    page_type = page_state.get("page_type", "unknown")
    has_price, has_official = infer_high_level_filters(dom_state)

    planner_input = {
        "goal": {
            "raw_user_goal": goal_text,
            "constraints": {
                # Bạn tự điền / refine sau
            },
        },
        "high_level_state": {
            "page_type": page_type,
            "current_step": heuristic_current_step(page_type),
            "has_price_filter": has_price,
            "has_official_store_filter": has_official,
        },
        "history_summary": build_history_summary(steps, step.get("step", 0)),
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

    # Placeholder PlannerOutput để bạn điền tay
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

    return {
        "episode_id": ep.get("episode_id"),
        "step": step.get("step"),
        "planner_input": planner_input,
        "planner_output_placeholder": planner_output_placeholder,
    }


def build_controller_sample(ep: Dict[str, Any], step: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tạo 1 sample ControllerInput skeleton + placeholder ControllerOutput."""
    goal_text = ep.get("goal", "")
    goal_summary = goal_text[:120]

    page_state_full = step.get("page_state") or {}
    page_type = page_state_full.get("page_type", "unknown")
    dom_state = page_state_full.get("dom_state") or {}
    vision_state = page_state_full.get("vision_state") or {}

    has_price, has_official = infer_high_level_filters(dom_state)

    # Heuristic current_plan_step.type đơn giản để debug
    if has_price or has_official:
        step_type = "APPLY_FILTER"
        desc = "Lọc theo các filter có sẵn trên trang (tạm thời)."
    else:
        step_type = "SEARCH_PRODUCT"
        desc = "Tiếp tục thao tác tìm kiếm / duyệt sản phẩm (tạm thời)."

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
        "short_history": build_short_history_for_controller(steps, step.get("step", 0)),
    }

    controller_output_placeholder = {
        "chosen_action": {
            "action_id": "",
            "type": "",
            "reason": "TODO: lý do chọn action này.",
        },
        "reason": "TODO: giải thích vì sao action giúp tiến gần current_plan_step.",
    }

    return {
        "episode_id": ep.get("episode_id"),
        "step": step.get("step"),
        "controller_input": controller_input,
        "controller_output_placeholder": controller_output_placeholder,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Planner/Controller schema samples from collected episodes.")
    ap.add_argument(
        "--episodes_dir",
        default="data/trajectories/shopping/episodes",
        help="Thư mục chứa ep_*.json do collect_trajectories.py sinh ra.",
    )
    ap.add_argument(
        "--out_dir",
        default="data/schema_samples",
        help="Thư mục ghi planner_samples.jsonl và controller_samples.jsonl.",
    )
    ap.add_argument(
        "--max_episodes",
        type=int,
        default=5,
        help="Số episode tối đa để lấy mẫu (0 hoặc âm = không giới hạn).",
    )
    ap.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=4,
        help="Số step tối đa trên mỗi episode để sinh sample (chọn step đầu tiên).",
    )
    args = ap.parse_args()

    episodes_dir = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)

    max_eps = args.max_episodes if args.max_episodes > 0 else None
    episodes = load_episodes(episodes_dir, max_eps)
    if not episodes:
        print(f"[error] Không tìm thấy episode JSON nào trong {episodes_dir}")
        return

    planner_rows: List[Dict[str, Any]] = []
    controller_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        steps = ep.get("steps") or []
        # lấy các bước đầu tiên (hoặc toàn bộ nếu ít hơn)
        steps_subset = steps[: args.max_steps_per_episode] if args.max_steps_per_episode > 0 else steps
        for s in steps_subset:
            # Bỏ qua step không có page_state
            if not s.get("page_state"):
                continue
            planner_rows.append(build_planner_sample(ep, s, steps))
            controller_rows.append(build_controller_sample(ep, s, steps))

    if not planner_rows:
        print("[warn] Không sinh được sample Planner nào (không có step với page_state).")
    if not controller_rows:
        print("[warn] Không sinh được sample Controller nào (không có step với page_state).")

    write_jsonl(out_dir / "planner_samples.jsonl", planner_rows)
    write_jsonl(out_dir / "controller_samples.jsonl", controller_rows)

    print(f"[ok] Đã ghi {len(planner_rows)} planner_samples vào {out_dir / 'planner_samples.jsonl'}")
    print(f"[ok] Đã ghi {len(controller_rows)} controller_samples vào {out_dir / 'controller_samples.jsonl'}")


if __name__ == "__main__":
    main()

