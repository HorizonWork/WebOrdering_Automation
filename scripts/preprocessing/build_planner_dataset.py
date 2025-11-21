# -*- coding: utf-8 -*-
"""
Generate schema-aligned samples for Planner and Controller from collected episodes.

This script reads `ep_*.json` files (or synthetic samples if none are present) and
emits JSONL datasets:

  - `data/processed/planner/train.jsonl` / `test.jsonl`
  - `data/processed/controller/train.jsonl` / `test.jsonl`

Each line is a training/example row:
  - Planner:    {episode_id, step, planner_input, planner_output, ...}
  - Controller: {episode_id, step, controller_input, controller_output, ...}

If an episode step contains `teacher_labels`, real labels are used. Otherwise we
fill reasonable defaults based on page_state, goal and constraints (no more
placeholder "TODO" strings).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_SYNTHETIC_DIR = Path("data/raw/misc/synthetic_episodes")

# ------------------------------------------------------------------------------
# Data loading utilities
# ------------------------------------------------------------------------------


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


def load_synthetic_episodes(dir_path: Path = DEFAULT_SYNTHETIC_DIR) -> List[Dict[str, Any]]:
    """Load synthetic example episodes used when no real data is available."""
    if not dir_path.exists():
        return []
    return load_episodes(dir_path)


# ------------------------------------------------------------------------------
# Heuristics
# ------------------------------------------------------------------------------

PRICE_PATTERN = re.compile(r"(\d{3,7})\s*(k|nghin|nghìn|ngan|trieu|triệu|tr)?", re.IGNORECASE)


def extract_goal_constraints(goal_text: str) -> Dict[str, Any]:
    """Parse common constraints (price ceiling, official store, brand) from goal."""
    text = (goal_text or "").lower()
    clean = re.sub(r"[.,]", "", text)

    max_price = None
    m = PRICE_PATTERN.search(clean)
    if m:
        value = int(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in {"k", "nghin", "nghìn", "ngan"}:
            value *= 1000
        elif unit in {"tr", "trieu", "triệu"}:
            value *= 1_000_000
        max_price = value

    must_official = any(k in text for k in ["chính hãng", "mall", "official"])
    brand_match = re.search(
        r"(samsung|apple|xiaomi|oppo|asus|dell|hp|lenovo|vsmart|sony)",
        text,
    )

    return {
        "max_price": max_price,
        "must_official_store": must_official,
        "preferred_brand": brand_match.group(1) if brand_match else None,
    }


def infer_high_level_filters(dom_state: Dict[str, Any]) -> Tuple[bool, bool]:
    """Heuristics: detect presence of price / official-store filters."""
    filters = dom_state.get("filters") or []
    has_price = False
    has_official = False

    for f in filters:
        label = (f.get("label") or "").lower()
        fid = (f.get("id") or "").lower()
        text = f"{fid} {label}"
        if any(k in text for k in ["price", "giá", "khoảng giá", "max_price"]):
            has_price = True
        if any(k in text for k in ["mall", "chính hãng", "official"]):
            has_official = True

    return has_price, has_official


def build_history_summary(steps: List[Dict[str, Any]], upto_step_num: int, window: int = 3) -> List[str]:
    """Create short textual history summary for Planner."""
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
    if "listing" in page_type_l:
        return "PRODUCT_LISTING"
    if "product" in page_type_l:
        return "PRODUCT_SELECTED"
    if "cart" in page_type_l:
        return "IN_CART"
    if "checkout" in page_type_l or "payment" in page_type_l:
        return "CHECKOUT"
    return "UNKNOWN"


def build_available_actions_flat_from_filters(dom_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a flat list of candidate DSL actions from dom_state.filters[*].actions.
    """
    filters = dom_state.get("filters") or []
    actions: Dict[str, Dict[str, Any]] = {}

    for f in filters:
        label = (f.get("label") or "").strip()
        for aid in f.get("actions") or []:
            if not isinstance(aid, str):
                continue
            if aid in actions:
                continue
            aid_lower = aid.lower()
            if "input" in aid_lower or "text" in aid_lower or "price" in aid_lower:
                atype = "FILL"
            elif any(k in aid_lower for k in ["btn", "button", "checkbox", "check", "click", "apply"]):
                atype = "CLICK"
            else:
                atype = "CLICK"

            actions[aid] = {
                "id": aid,
                "type": atype,
                "description": label or f"Tác vụ {aid}",
                "dom_selector": None,
                "vision_ref": None,
            }

    return list(actions.values())


def derive_actions_from_page_state(page_state: Dict[str, Any], goal_text: str) -> List[Dict[str, Any]]:
    """Create lightweight fallback actions when filters don't provide any."""
    page_type = (page_state.get("page_type") or "").lower()
    dom_state = page_state.get("dom_state") or {}
    has_search = bool(dom_state.get("has_search"))

    actions: List[Dict[str, Any]] = []

    if has_search or "search" in page_type:
        actions.append(
            {
                "id": "SEARCH_BOX",
                "type": "FILL",
                "description": "Nhập truy vấn tìm kiếm",
                "text": goal_text[:80],
            }
        )
        actions.append(
            {
                "id": "SUBMIT_SEARCH",
                "type": "PRESS",
                "description": "Gửi truy vấn (Enter/nút search)",
            }
        )

    if "listing" in page_type or "search" in page_type:
        actions.append(
            {
                "id": "SELECT_TOP_RESULT",
                "type": "CLICK",
                "description": "Mở sản phẩm đầu tiên phù hợp",
            }
        )

    if "product" in page_type:
        actions.append(
            {
                "id": "ADD_TO_CART",
                "type": "CLICK",
                "description": "Thêm sản phẩm vào giỏ",
            }
        )

    if "cart" in page_type:
        actions.append(
            {
                "id": "GO_TO_CHECKOUT_BUTTON",
                "type": "CLICK",
                "description": "Chuyển sang trang thanh toán",
            }
        )

    if "checkout" in page_type or "payment" in page_type:
        actions.append(
            {
                "id": "REVIEW_ORDER_SECTION",
                "type": "WAIT_FOR",
                "description": "Kiểm tra lại địa chỉ/tổng tiền",
            }
        )

    return actions


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

    Expected structure:
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


def infer_next_plan_step(
    goal_text: str,
    page_state: Dict[str, Any],
    has_price_filter: bool,
    has_official_filter: bool,
) -> Dict[str, Any]:
    """Heuristic next_plan_step when we do not have teacher labels."""
    constraints = extract_goal_constraints(goal_text)
    page_type = (page_state.get("page_type") or "unknown").lower()
    dom_state = page_state.get("dom_state") or {}

    if "checkout" in page_type or "payment" in page_type:
        step_type = "REVIEW_ORDER"
        description = "Kiểm tra tổng tiền/địa chỉ và chờ xác nhận trước khi thanh toán."
    elif "cart" in page_type:
        step_type = "GO_TO_CHECKOUT"
        description = "Đi tới bước thanh toán để xem phí và địa chỉ giao hàng."
    elif "product" in page_type:
        step_type = "GO_TO_CART"
        description = "Thêm sản phẩm phù hợp vào giỏ để chuẩn bị thanh toán."
    elif has_price_filter or constraints.get("max_price"):
        step_type = "APPLY_FILTER"
        description = "Đặt bộ lọc giá/Mall để thu hẹp kết quả theo yêu cầu."
    elif "listing" in page_type:
        step_type = "SELECT_PRODUCT"
        description = "Chọn sản phẩm nổi bật đáp ứng mục tiêu."
    else:
        step_type = "SEARCH_PRODUCT"
        description = "Nhập từ khóa và tìm sản phẩm phù hợp với mục tiêu."

    # Enrich constraints
    enriched_constraints = {
        "max_price": constraints.get("max_price"),
        "must_official_store": constraints.get("must_official_store"),
        "preferred_brand": constraints.get("preferred_brand"),
        "page_type_hint": heuristic_current_step(page_state.get("page_type", "")),
        "has_price_filter": has_price_filter,
        "has_official_filter": has_official_filter,
        "product_count": len(dom_state.get("products") or []),
        "cart_items": len(dom_state.get("cart_items") or []),
    }

    return {
        "plan_version": 1,
        "overall_strategy": describe_overall_strategy(goal_text, constraints, has_price_filter, has_official_filter),
        "next_plan_step": {
            "step_id": step_type,
            "type": step_type,
            "description": description,
            "constraints": {k: v for k, v in enriched_constraints.items() if v},
        },
    }


def describe_overall_strategy(
    goal_text: str,
    constraints: Dict[str, Any],
    has_price_filter: bool,
    has_official_filter: bool,
) -> str:
    """Generate a short overall_strategy string without placeholders."""
    parts: List[str] = []
    if constraints.get("max_price"):
        parts.append(f"Giữ giá <= {constraints['max_price']:,}")
    if constraints.get("must_official_store") or has_official_filter:
        parts.append("Ưu tiên gian hàng chính hãng/Mall")
    if constraints.get("preferred_brand"):
        parts.append(f"Tập trung thương hiệu {constraints['preferred_brand']}")
    if not parts:
        parts.append("Tìm kiếm rồi lọc kết quả phù hợp mục tiêu")
    return " -> ".join(parts)


def choose_controller_action(
    goal_summary: str,
    page_state: Dict[str, Any],
    actions_flat: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Pick one action and a justification when teacher labels are absent."""
    page_type = (page_state.get("page_type") or "").lower()
    dom_state = page_state.get("dom_state") or {}

    def find_action(predicate):
        for act in actions_flat:
            try:
                if predicate(act):
                    return act
            except Exception:
                continue
        return None

    max_price = constraints.get("max_price")
    if max_price:
        price_action = find_action(
            lambda a: "price" in (a.get("id", "") + " " + a.get("description", "")).lower()
            and a.get("type", "").upper() in {"FILL", "CLICK"}
        )
        if price_action:
            chosen = dict(price_action)
            if chosen.get("type", "").upper() == "FILL":
                chosen["text"] = str(max_price)
            reason = f"Thiết lập trần giá {max_price:,} theo yêu cầu."
            return chosen, reason

    if constraints.get("must_official_store"):
        official_action = find_action(
            lambda a: any(k in (a.get("description", "") + a.get("id", "")).lower() for k in ["mall", "official", "chính hãng"])
        )
        if official_action:
            chosen = dict(official_action)
            reason = "Ưu tiên gian hàng chính hãng theo mục tiêu."
            return chosen, reason

    if "search" in page_type:
        search_fill = find_action(lambda a: a.get("type", "").upper() == "FILL")
        if search_fill:
            chosen = dict(search_fill)
            chosen["text"] = goal_summary
            reason = "Nhập từ khóa tìm kiếm để lấy kết quả phù hợp."
            return chosen, reason

    if "listing" in page_type or "search" in page_type:
        select_click = find_action(lambda a: a.get("type", "").upper() == "CLICK")
        if select_click:
            chosen = dict(select_click)
            reason = "Chọn sản phẩm nổi bật từ danh sách để xem chi tiết."
            return chosen, reason

    if "cart" in page_type:
        checkout = find_action(
            lambda a: any(k in (a.get("description", "") + a.get("id", "")).lower() for k in ["checkout", "thanh toán", "pay"])
        )
        if checkout:
            chosen = dict(checkout)
            reason = "Đi tới bước thanh toán theo kế hoạch."
            return chosen, reason

    if "checkout" in page_type or "payment" in page_type:
        review = find_action(lambda a: a.get("type", "").upper() in {"WAIT_FOR", "CLICK"})
        if review:
            chosen = dict(review)
            reason = "Kiểm tra thông tin đơn hàng trước khi xác nhận."
            return chosen, reason

    # Fallback: pick the first available action.
    if actions_flat:
        fallback = dict(actions_flat[0])
        reason = "Không có gợi ý cụ thể; chọn action khả dụng đầu tiên để tiến thêm bước."
        return fallback, reason

    return {"action_id": "NO_ACTION", "type": "WAIT", "description": "Không có action khả dụng"}, "Không tìm thấy action khả dụng."


# ------------------------------------------------------------------------------
# Sample builders
# ------------------------------------------------------------------------------


def build_planner_sample(ep: Dict[str, Any], step: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create one PlannerInput + PlannerOutput sample."""
    goal_text = ep.get("goal", "")
    page_state = step.get("page_state") or {}
    dom_state = page_state.get("dom_state") or {}
    page_type = page_state.get("page_type", "unknown")
    has_price, has_official = infer_high_level_filters(dom_state)
    constraints = extract_goal_constraints(goal_text)

    step_num = int(step.get("step") or 0)

    planner_input = {
        "goal": {
            "raw_user_goal": goal_text,
            "constraints": constraints,
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

    planner_labels, _ = extract_teacher_labels(step)
    teacher_next = planner_labels.get("next_plan_step") if isinstance(planner_labels, dict) else None

    if isinstance(teacher_next, dict):
        planner_output = {
            "plan_version": 1,
            "overall_strategy": planner_labels.get("overall_strategy", ""),
            "next_plan_step": teacher_next,
        }
    else:
        planner_output = infer_next_plan_step(goal_text, page_state, has_price, has_official)

    planner_output_placeholder = planner_output  # Kept for backward compatibility field

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
    constraints = extract_goal_constraints(goal_text)

    if has_price or has_official or constraints.get("max_price"):
        step_type = "APPLY_FILTER"
        desc = "Lọc theo giá/nguồn hàng theo yêu cầu."
    elif "checkout" in (page_type or "").lower():
        step_type = "REVIEW_ORDER"
        desc = "Kiểm tra thông tin thanh toán trước khi xác nhận."
    else:
        step_type = "SEARCH_PRODUCT"
        desc = "Tìm và mở sản phẩm phù hợp."

    step_num = int(step.get("step") or 0)

    actions_from_filters = build_available_actions_flat_from_filters(dom_state)
    fallback_actions = derive_actions_from_page_state(page_state_full, goal_text)
    all_actions = actions_from_filters + [a for a in fallback_actions if a not in actions_from_filters]

    controller_input = {
        "goal": {
            "summary": goal_summary,
        },
        "current_plan_step": {
            "type": step_type,
            "description": desc,
            "constraints": constraints,
        },
        "page_state": {
            "page_type": page_type,
            "dom_state": dom_state,
            "vision_state": vision_state,
        },
        "available_actions_flat": all_actions,
        "last_action_result": build_last_action_result(step),
        "short_history": build_short_history_for_controller(steps, step_num),
    }

    _, controller_labels = extract_teacher_labels(step)
    chosen = controller_labels.get("chosen_action") if isinstance(controller_labels, dict) else None

    if isinstance(chosen, dict):
        controller_output = {
            "chosen_action": chosen,
            "reason": controller_labels.get("reason", chosen.get("reason", "")),
        }
    else:
        chosen_action, reason = choose_controller_action(goal_summary, page_state_full, all_actions, constraints)
        controller_output = {
            "chosen_action": chosen_action,
            "reason": reason,
        }

    controller_output_placeholder = controller_output  # kept for consistency

    return {
        "episode_id": ep.get("episode_id"),
        "step": step.get("step"),
        "controller_input": controller_input,
        "controller_output": controller_output,
        "controller_output_placeholder": controller_output_placeholder,
    }


# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_test(rows: List[Dict[str, Any]], test_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into train/test with a fixed seed for reproducibility."""
    if not rows:
        return [], []
    rng = random.Random(42)
    rows_shuffled = rows[:]
    rng.shuffle(rows_shuffled)
    test_size = int(len(rows_shuffled) * test_ratio)
    test_rows = rows_shuffled[:test_size] if test_size > 0 else []
    train_rows = rows_shuffled[test_size:] if rows_shuffled[test_size:] else rows_shuffled
    return train_rows, test_rows


def write_dataset(rows: List[Dict[str, Any]], root_dir: Path, name: str, test_ratio: float) -> None:
    """Write train/test splits for a dataset name."""
    train_rows, test_rows = split_train_test(rows, test_ratio)

    train_path = root_dir / name / "train.jsonl"
    write_jsonl(train_path, train_rows)
    print(f"[ok] Wrote {len(train_rows)} {name} samples to {train_path}")

    test_path = root_dir / name / "test.jsonl"
    write_jsonl(test_path, test_rows)
    print(f"[ok] Wrote {len(test_rows)} {name} samples to {test_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main(default_dataset: str = "both") -> None:
    ap = argparse.ArgumentParser(
        description="Generate Planner/Controller schema samples from collected episodes."
    )
    ap.add_argument(
        "--episodes_dir",
        default="data/raw/lazada/episodes",
        help="Folder chứa ep_*.json (vd: data/raw/shopee/episodes).",
    )
    ap.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Root folder cho processed splits (planner/..., controller/...).",
    )
    ap.add_argument(
        "--dataset",
        choices=["planner", "controller", "both"],
        default=default_dataset,
        help="Chọn dataset cần build.",
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
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Tỷ lệ test split (0-0.5).",
    )
    args = ap.parse_args()

    episodes_dir = Path(args.episodes_dir)
    processed_dir = Path(args.processed_dir)

    max_eps = args.max_episodes if args.max_episodes > 0 else None
    episodes = load_episodes(episodes_dir, max_eps)
    if not episodes:
        synthetic_eps = load_synthetic_episodes()
        if synthetic_eps:
            print(f"[info] No episodes found in {episodes_dir}, fallback to synthetic samples ({DEFAULT_SYNTHETIC_DIR})")
            episodes = synthetic_eps
        else:
            print(f"[error] No episode JSON found in {episodes_dir} and no synthetic data available.")
            return

    planner_rows: List[Dict[str, Any]] = []
    controller_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        steps = ep.get("steps") or []
        steps_subset = steps if args.max_steps_per_episode <= 0 else steps[: args.max_steps_per_episode]

        for s in steps_subset:
            if not s.get("page_state"):
                continue
            planner_rows.append(build_planner_sample(ep, s, steps))
            controller_rows.append(build_controller_sample(ep, s, steps))

    if args.dataset in ("planner", "both"):
        if not planner_rows:
            print("[warn] No Planner samples generated (no steps with page_state).")
        write_dataset(planner_rows, processed_dir, "planner", args.test_ratio)

    if args.dataset in ("controller", "both"):
        if not controller_rows:
            print("[warn] No Controller samples generated (no steps with page_state).")
        write_dataset(controller_rows, processed_dir, "controller", args.test_ratio)


if __name__ == "__main__":
    main()
