# -*- coding: utf-8 -*-
"""
Convert planner/controller schema JSONL -> text pairs:
  - data/processed/planner/train_text.jsonl
  - data/processed/planner/test_text.jsonl
  - data/processed/controller/train_text.jsonl
  - data/processed/controller/test_text.jsonl

Mỗi dòng:
  {
    "episode_id": "...",
    "step": 3,
    "input": "<prompt string>",
    "output": "<json label string>"
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------
# IO helpers
# ---------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[warn] JSONL not found: {path}")
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"[warn] Cannot parse line in {path}: {exc}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------
# Pair builders
# ---------------------------------------------------------


def build_planner_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    """Tạo (input_text, output_text) cho Planner từ 1 dòng JSONL."""
    pi = row.get("planner_input") or {}
    po = row.get("planner_output") or {}

    goal = ((pi.get("goal") or {}).get("raw_user_goal") or "").strip()
    constraints = (pi.get("goal") or {}).get("constraints") or {}
    hl_state = pi.get("high_level_state") or {}
    history = pi.get("history_summary") or []
    avail_actions = pi.get("available_high_level_actions") or []

    max_hist = 4
    history = history[:max_hist]

    cons_parts = []
    if constraints.get("max_price"):
        cons_parts.append(f"max_price={constraints['max_price']:,}")
    if constraints.get("must_official_store"):
        cons_parts.append("must_official_store=True")
    if constraints.get("preferred_brand"):
        cons_parts.append(f"brand={constraints['preferred_brand']}")
    cons_str = ", ".join(cons_parts) if cons_parts else "Không rõ"

    hist_str = "\n".join(f"- {h}" for h in history) if history else "- (trống)"
    actions_str = "\n".join(f"- {a}" for a in avail_actions) if avail_actions else "- (trống)"

    page_type = hl_state.get("page_type", "unknown")
    current_step = hl_state.get("current_step", "UNKNOWN")
    has_price = hl_state.get("has_price_filter", False)
    has_official = hl_state.get("has_official_store_filter", False)

    input_text = f"""[PLANNER]
Goal: {goal}
Constraints: {cons_str}

Page type: {page_type}
Current progress step: {current_step}
Has price filter: {has_price}
Has official-store filter: {has_official}

Recent history:
{hist_str}

Available high-level actions:
{actions_str}

Hãy quyết định bước kế tiếp (next_plan_step) ở cấp độ high-level.
Chỉ trả về JSON với cấu trúc:
{{
  "step_id": "...",
  "type": "...",
  "description": "...",
  "constraints": {{ ... }}
}}
"""

    # Label: chỉ lấy next_plan_step cho gọn
    next_plan = (po.get("next_plan_step") or {})
    output_text = json.dumps(next_plan, ensure_ascii=False)

    return input_text.strip(), output_text


def build_controller_pair(
    row: Dict[str, Any],
    max_dom_chars: int = 1500,
    max_actions: int = 12,
    max_history: int = 4,
) -> Tuple[str, str]:
    """Tạo (input_text, output_text) cho Controller từ 1 dòng JSONL."""
    ci = row.get("controller_input") or {}
    co = row.get("controller_output") or {}

    goal = ((ci.get("goal") or {}).get("summary") or "").strip()
    cps = ci.get("current_plan_step") or {}
    page_state = ci.get("page_state") or {}
    dom_state = page_state.get("dom_state") or {}
    vision_state = page_state.get("vision_state") or {}

    page_type = page_state.get("page_type", "unknown")

    dom_text = (dom_state.get("text_excerpt") or "")[:max_dom_chars]
    short_history = (ci.get("short_history") or [])[:max_history]

    actions = (ci.get("available_actions_flat") or [])[:max_actions]

    history_str = "\n".join(f"- {h}" for h in short_history) if short_history else "- (trống)"
    actions_lines = []
    for idx, a in enumerate(actions, start=1):
        aid = a.get("id") or ""
        atype = a.get("type") or ""
        desc = a.get("description") or ""
        actions_lines.append(f"{idx}. id={aid} | type={atype} | desc={desc}")
    actions_str = "\n".join(actions_lines) if actions_lines else "(không có action candidate)"

    plan_type = cps.get("type", "")
    plan_desc = cps.get("description", "")
    plan_cons = cps.get("constraints") or {}
    plan_cons_str = json.dumps(plan_cons, ensure_ascii=False)

    last_res = ci.get("last_action_result") or {}
    last_res_str = json.dumps(last_res, ensure_ascii=False)

    input_text = f"""[CONTROLLER]
Goal: {goal}

Current plan step:
- type: {plan_type}
- description: {plan_desc}
- constraints: {plan_cons_str}

Page:
- type: {page_type}

DOM summary (truncated):
{dom_text}

Recent controller history:
{history_str}

Last action result (if any):
{last_res_str}

Candidate actions (chỉ chọn trong danh sách dưới):
{actions_str}

Hãy chọn 1 action tiếp theo từ danh sách trên, trả về JSON cho chosen_action
với các trường: id, type, description và các field cần thiết khác (nếu có).
"""

    chosen_action = (co.get("chosen_action") or {})
    output_text = json.dumps(chosen_action, ensure_ascii=False)

    return input_text.strip(), output_text


# ---------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------


def convert_split(
    in_path: Path,
    out_path: Path,
    builder,
    max_samples: int | None = None,
) -> None:
    rows = read_jsonl(in_path)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        try:
            inp, out = builder(r)
            out_rows.append(
                {
                    "episode_id": r.get("episode_id"),
                    "step": r.get("step"),
                    "input": inp,
                    "output": out,
                }
            )
        except Exception as exc:
            print(f"[warn] Failed to build pair for row (episode={r.get('episode_id')} step={r.get('step')}): {exc}")

    write_jsonl(out_path, out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert planner/controller schema JSONL -> text pairs for LLM training."
    )
    ap.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Root chứa planner/train.jsonl, controller/train.jsonl, ...",
    )
    ap.add_argument(
        "--dataset",
        choices=["planner", "controller", "both"],
        default="both",
        help="Chọn loại dataset cần convert.",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Giới hạn số sample mỗi split (0 = không giới hạn).",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    max_samples = args.max_samples if args.max_samples > 0 else None

    if args.dataset in ("planner", "both"):
        planner_train = processed_dir / "planner" / "train.jsonl"
        planner_test = processed_dir / "planner" / "test.jsonl"
        planner_train_out = processed_dir / "planner" / "train_text.jsonl"
        planner_test_out = processed_dir / "planner" / "test_text.jsonl"

        convert_split(planner_train, planner_train_out, build_planner_pair, max_samples)
        convert_split(planner_test, planner_test_out, build_planner_pair, max_samples)

    if args.dataset in ("controller", "both"):
        controller_train = processed_dir / "controller" / "train.jsonl"
        controller_test = processed_dir / "controller" / "test.jsonl"
        controller_train_out = processed_dir / "controller" / "train_text.jsonl"
        controller_test_out = processed_dir / "controller" / "test_text.jsonl"

        convert_split(controller_train, controller_train_out, build_controller_pair, max_samples)
        convert_split(controller_test, controller_test_out, build_controller_pair, max_samples)


if __name__ == "__main__":
    main()
