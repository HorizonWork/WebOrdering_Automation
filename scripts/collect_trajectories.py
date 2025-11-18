# -*- coding: utf-8 -*-
"""
Collect web interaction trajectories for training / analysis.

Mục tiêu:
    - Chạy agent (ReAct hiện tại) trên Shopee/Lazada hoặc site bất kỳ.
    - Ghi lại toàn bộ episode (steps, thought, action, observation, result).
    - Chuẩn hoá thành JSON để sau này:
        * Tái tạo episode.
        * Tách sample cho Planner / Controller / SFT.
    - (Tuỳ chọn) gọi teacher LLM (vd: Gemini Flash) để gán nhãn thêm cho mỗi step.

Ví dụ (thu thập 1 episode):

    python scripts/collect_trajectories.py ^
        --query "Tìm iPhone 15 trên Shopee" ^
        --url "https://shopee.vn" ^
        --out_dir data/trajectories ^
        --max_steps 10 ^
        --headless

Thu thập nhiều episode từ file tasks.jsonl:
    - Mỗi dòng: {"query": "...", "url": "..."}

    python scripts/collect_trajectories.py ^
        --tasks data/tasks_shopping.jsonl ^
        --out_dir data/trajectories/shopping ^
        --episodes 20

Gắn teacher Gemini (tùy chọn, cần: pip install google-generativeai, GEMINI_API_KEY):

    python scripts/collect_trajectories.py ^
        --query "Tìm iPhone 15 trên Shopee" ^
        --url "https://shopee.vn" ^
        --with_teacher ^
        --teacher_backend gemini ^
        --teacher_model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve project root (walk up until pyproject.toml)
ROOT_DIR = Path(__file__).resolve().parent
for parent in ROOT_DIR.parents:
    if (parent / "pyproject.toml").exists():
        ROOT_DIR = parent
        break
else:
    ROOT_DIR = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.agent_orchestrator import AgentOrchestrator, ExecutionResult  # noqa: E402
from src.planning.react_engine import ReActEngine  # noqa: E402
from src.perception.ui_detector import UIDetector  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    timestamp: str
    thought: str
    action: Dict[str, Any]
    result: Dict[str, Any]
    page_state: Dict[str, Any]
    teacher_labels: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeRecord:
    episode_id: str
    goal: str
    start_url: str
    created_at: str
    steps: List[StepRecord]
    final_status: str
    final_url: Optional[str]
    success: bool
    error: Optional[str]
    summary: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Teacher LLM (optional, e.g. Gemini)
# ---------------------------------------------------------------------------


class TeacherBase:
    """Base class for teacher LLMs (optional)."""

    def annotate_step(
        self,
        goal: str,
        page_state: Dict[str, Any],
        short_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Return optional labels for this step, e.g.:

            {
              "planner": {"next_plan_step": {...}},
              "controller": {"chosen_action": {...}},
              "raw_response": "<original LLM text>"
            }
        """
        raise NotImplementedError


class GeminiTeacher(TeacherBase):
    """
    Simple wrapper around Gemini API for bootstrapping labels.

    Yêu cầu:
        pip install google-generativeai
        export GEMINI_API_KEY=...
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-generativeai chưa được cài. "
                "Cài bằng: pip install google-generativeai"
            ) from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu GEMINI_API_KEY trong environment.")

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def annotate_step(
        self,
        goal: str,
        page_state: Dict[str, Any],
        short_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Build a compact JSON-only prompt and let Gemini propose:
          - planner.next_plan_step
          - controller.chosen_action
        """
        prompt_text = (
            "Bạn là teacher cho web-shopping agent.\n"
            "Dựa trên goal và page_state hiện tại, hãy đề xuất:\n"
            "  (1) Bước kế hoạch cấp cao tiếp theo (Planner) với schema:\n"
            '      {"next_plan_step": {"step_id": <one_of>, "type": "...", "description": "..."} }\n'
            "      Các step_id hợp lệ: SEARCH_PRODUCT, APPLY_FILTER, SELECT_PRODUCT,\n"
            "      GO_TO_CART, GO_TO_CHECKOUT, FILL_CHECKOUT_INFO, REVIEW_ORDER, TERMINATE.\n"
            "  (2) Một action cụ thể gần nhất để tiến tới bước đó, với schema:\n"
            '      {"chosen_action": {"action_id": "...", "type": "...", "text": "...?"}}\n'
            "      Ở đây action_id có thể dựa trên element description trong page_state.\n"
            "Chỉ trả JSON:\n"
            '{\"planner\": {...}, \"controller\": {...}}\n\n'
            f"Goal: {goal}\n"
            f"Page state (rút gọn): {json.dumps(page_state, ensure_ascii=False)[:2000]}\n"
            f"Short history (thought+action): {json.dumps(short_history, ensure_ascii=False)[:1500]}\n"
        )

        prompt = {
            "role": "user",
            "parts": [prompt_text],
        }

        try:  # pragma: no cover - network call
            resp = self._model.generate_content([prompt])
            raw = resp.text or ""
        except Exception as exc:
            logger.warning(f"GeminiTeacher lỗi: {exc}")
            return None

        raw = raw.strip()

        # Try parse whole response as JSON
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data["raw_response"] = raw
                return data
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract JSON object substring
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = raw[start : end + 1]
                data = json.loads(snippet)
                if isinstance(data, dict):
                    data["raw_response"] = raw
                    return data
        except Exception:
            pass

        logger.debug("GeminiTeacher: không parse được JSON, bỏ qua label step này.")
        return None


def build_teacher(backend: str, model_name: str) -> Optional[TeacherBase]:
    backend = backend.lower()
    if backend == "none":
        return None
    if backend == "gemini":
        return GeminiTeacher(model_name=model_name)
    raise ValueError(f"Teacher backend không hỗ trợ: {backend}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tasks(tasks_path: Optional[str]) -> List[Tuple[str, str]]:
    """
    Load danh sách (query, url).
    Hỗ trợ JSON, JSONL (NDJSON).
    """
    if not tasks_path:
        return []

    p = Path(tasks_path)
    if not p.exists():
        raise FileNotFoundError(f"Tasks file không tồn tại: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    rows: List[Dict[str, Any]] = []
    # Try JSON (array or single object)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            rows = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            rows = [data]
    except json.JSONDecodeError:
        # Fallback: NDJSON
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Lỗi JSON trong {p}: {exc}") from exc

    pairs: List[Tuple[str, str]] = []
    for r in rows:
        q = r.get("query") or r.get("goal") or r.get("task")
        u = r.get("url") or r.get("start_url")
        if isinstance(q, str) and isinstance(u, str):
            pairs.append((q, u))
    return pairs


def build_page_state(observation: Dict[str, Any], ui_detector: UIDetector) -> Dict[str, Any]:
    """
    Chuẩn hoá observation từ AgentOrchestrator thành page_state đơn giản.
    """
    url = observation.get("url", "")
    dom_html = observation.get("dom", "") or ""
    elements = observation.get("elements") or observation.get("interactive_elements") or []

    try:
        ui_info = ui_detector.detect_all(dom_html) if dom_html else {
            "search_box": {"found": False},
            "login_form": {"found": False},
            "product_listing": {"found": False},
            "navigation": {"found": False},
            "page_type": "unknown",
        }
    except Exception as exc:
        logger.warning(f"UIDetector lỗi: {exc}")
        ui_info = {
            "search_box": {"found": False},
            "login_form": {"found": False},
            "product_listing": {"found": False},
            "navigation": {"found": False},
            "page_type": "unknown",
        }

    page_state: Dict[str, Any] = {
        "url": url,
        "page_type": ui_info.get("page_type", "unknown"),
        "dom_state": {
            "distilled_length": len(dom_html),
            "has_search_box": bool(ui_info.get("search_box", {}).get("found")),
            "has_login_form": bool(ui_info.get("login_form", {}).get("found")),
            "has_product_listing": bool(ui_info.get("product_listing", {}).get("found")),
        },
        "elements": elements,
        "ui_detection": ui_info,
        # vision_state có thể bổ sung sau nếu cần (từ VisionEnhancer)
    }

    return page_state


def history_to_step_records(
    history: List[Dict[str, Any]],
    goal: str,
    ui_detector: UIDetector,
    teacher: Optional[TeacherBase] = None,
) -> List[StepRecord]:
    """
    Convert history từ ReActEngine (list[dict]) thành list StepRecord có page_state & (optional) teacher labels.
    """
    steps: List[StepRecord] = []

    for idx, s in enumerate(history, 1):
        observation = s.get("observation", {}) or {}
        page_state = build_page_state(observation, ui_detector)

        teacher_labels: Optional[Dict[str, Any]] = None
        if teacher is not None:
            short_history: List[Dict[str, Any]] = []
            for h in history[max(0, idx - 3) : idx]:
                short_history.append(
                    {
                        "step": h.get("step"),
                        "thought": h.get("thought"),
                        "action": h.get("action"),
                        "result": h.get("result"),
                    }
                )
            teacher_labels = teacher.annotate_step(goal=goal, page_state=page_state, short_history=short_history)

        steps.append(
            StepRecord(
                step=s.get("step", idx),
                timestamp=s.get("timestamp", datetime.now().isoformat()),
                thought=s.get("thought", ""),
                action=s.get("action", {}),
                result=s.get("result", {}),
                page_state=page_state,
                teacher_labels=teacher_labels,
            )
        )

    return steps


def execution_result_to_status(result: ExecutionResult, react_engine: ReActEngine) -> str:
    """Map ExecutionResult + history sang final_status string."""
    if not result.success and result.error:
        return "FAIL"

    progress = react_engine.analyze_progress()
    if progress.get("completion_status") == "complete":
        return "SUCCESS"

    # Nếu max_steps đạt mà chưa complete
    if len(react_engine.history) >= react_engine.max_steps:
        return "TIMEOUT"

    # Mặc định: chưa rõ, tạm coi là FAIL nếu success False
    return "FAIL" if not result.success else "SUCCESS"


def episode_to_json_dict(ep: EpisodeRecord) -> Dict[str, Any]:
    """Convert EpisodeRecord (dataclasses) thành dict JSON-friendly."""
    data = asdict(ep)
    return data


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------


async def collect_one_episode(
    query: str,
    start_url: str,
    max_steps: int,
    headless: bool,
    out_dir: Path,
    teacher: Optional[TeacherBase] = None,
) -> Path:
    """
    Chạy 1 episode với AgentOrchestrator hiện tại và lưu JSON vào out_dir/episodes.
    """
    logger.info("=" * 70)
    logger.info("Collecting one episode")
    logger.info(f"Query: {query}")
    logger.info(f"URL  : {start_url}")
    logger.info("=" * 70)

    orchestrator = AgentOrchestrator(
        max_steps=max_steps,
        headless=headless,
        enable_learning=False,  # tránh side-effect trong collection
        enable_guardrails=True,
    )

    try:
        result: ExecutionResult = await orchestrator.execute_task(
            query=query,
            start_url=start_url,
        )

        history = orchestrator.react_engine.get_history()
        ui_detector = UIDetector()
        step_records = history_to_step_records(
            history=history,
            goal=query,
            ui_detector=ui_detector,
            teacher=teacher,
        )

        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        final_status = execution_result_to_status(result, orchestrator.react_engine)

        ep = EpisodeRecord(
            episode_id=episode_id,
            goal=query,
            start_url=start_url,
            created_at=datetime.now().isoformat(),
            steps=step_records,
            final_status=final_status,
            final_url=result.final_url,
            success=result.success,
            error=result.error,
            summary=result.summary,
            metadata=result.metadata or {},
        )

        episodes_dir = ensure_dir(out_dir / "episodes")
        out_path = episodes_dir / f"{episode_id}.json"
        out_path.write_text(
            json.dumps(episode_to_json_dict(ep), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Episode saved: {out_path}")
        return out_path

    finally:
        await orchestrator.close()


async def async_main(args: argparse.Namespace) -> None:
    setup_logging(level=args.log_level, log_file=None)

    out_dir = ensure_dir(Path(args.out_dir))

    teacher: Optional[TeacherBase] = None
    if args.with_teacher:
        teacher = build_teacher(args.teacher_backend, args.teacher_model)
        logger.info(f"Teacher backend: {args.teacher_backend}, model={args.teacher_model}")

    tasks: List[Tuple[str, str]] = []
    if args.tasks:
        tasks = load_tasks(args.tasks)

    if not tasks:
        # Single query/url từ CLI
        if not args.query or not args.url:
            raise SystemExit("Cần --query và --url (hoặc cung cấp --tasks).")
        tasks = [(args.query, args.url)]

    # Giới hạn số episode
    if args.episodes is not None and args.episodes > 0:
        tasks = tasks[: args.episodes]

    logger.info(f"Collecting {len(tasks)} episode(s)")

    for i, (q, u) in enumerate(tasks, 1):
        logger.info(f"\n--- Episode {i}/{len(tasks)} ---")
        try:
            await collect_one_episode(
                query=q,
                start_url=u,
                max_steps=args.max_steps,
                headless=args.headless,
                out_dir=out_dir,
                teacher=teacher,
            )
        except Exception as exc:
            logger.error(f"Episode {i} failed: {exc}", exc_info=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Collect web interaction trajectories (episodes) for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single-task mode
    ap.add_argument("--query", help="Goal / natural language query (Vietnamese).")
    ap.add_argument("--url", help="Starting URL.")

    # Multi-task mode from file
    ap.add_argument(
        "--tasks",
        help="Path to JSON/JSONL file, mỗi dòng/entry có {\"query\": ..., \"url\": ...}.",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Số episode tối đa cần thu thập (cắt bớt từ tasks).",
    )

    ap.add_argument(
        "--out_dir",
        default=str(ROOT_DIR / "data" / "trajectories"),
        help="Thư mục lưu output episodes/*.json.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Giới hạn số bước agent.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Chạy browser headless (không hiện UI).",
    )
    ap.add_argument(
        "--log_level",
        default="INFO",
        help="Mức log (DEBUG/INFO/WARNING/ERROR).",
    )

    # Teacher options
    ap.add_argument(
        "--with_teacher",
        action="store_true",
        help="Bật teacher LLM để gán nhãn step (vd Gemini).",
    )
    ap.add_argument(
        "--teacher_backend",
        default="gemini",
        choices=["gemini", "none"],
        help="Backend teacher.",
    )
    ap.add_argument(
        "--teacher_model",
        default="gemini-2.0-flash",
        help="Tên model teacher (vd: gemini-2.0-flash).",
    )

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

