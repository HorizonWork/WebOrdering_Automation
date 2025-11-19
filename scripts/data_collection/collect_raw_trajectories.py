# -*- coding: utf-8 -*-
"""
Collect web interaction trajectories for training / analysis.

- Gọi AgentOrchestrator để chạy 1 task (query + start_url)
- Lấy history từ ReactEngine
- Chuyển history thành EpisodeRecord (list StepRecord)
- Lưu ra JSON trong data/raw/<platform>/{episodes|screenshots|dom_snapshots}/
- Mirror episode JSON theo nhãn success/fail sang data/trajectories/{successful,failed}/

Lưu ý:
    - Teacher (Gemini) là OPTIONAL, KHÔNG nên bật khi bạn không muốn dùng API.
    - UIDetector được dùng để suy ra page_type, search_box,... từ DOM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import uuid
import traceback
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

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


def episode_to_json_dict(ep: EpisodeRecord) -> Dict[str, Any]:
    """Convert EpisodeRecord (dataclass) thành dict để dump JSON."""
    return asdict(ep)


# ---------------------------------------------------------------------------
# Teacher LLM (OPTIONAL – bạn có thể bỏ qua hoàn toàn phần này)
# ---------------------------------------------------------------------------


class TeacherBase:
    def annotate_step(
        self,
        goal: str,
        page_state: Dict[str, Any],
        short_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class GeminiTeacher(TeacherBase):
    """
    Wrapper cho Gemini API với JSON mode.

    KHÔNG dùng nếu bạn không muốn gọi API:
    - Đừng truyền --with_teacher
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("Thiếu thư viện: pip install google-generativeai") from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ Thiếu GEMINI_API_KEY, Teacher sẽ không hoạt động.")
            self._model = None
            return

        genai.configure(api_key=api_key)

        # Cấu hình model trả về JSON
        self._model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"},
        )

    def annotate_step(
        self,
        goal: str,
        page_state: Dict[str, Any],
        short_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._model:
            return None

        prompt_text = (
            "You are an expert Web Automation Teacher.\n"
            "Analyze the current state and goal to provide the Optimal Next Action.\n\n"
            f"User Goal: {goal}\n"
            f"Current URL: {page_state.get('url', 'unknown')}\n"
            f"Page Elements Summary: {json.dumps(page_state.get('dom_state', {}), ensure_ascii=False)}\n"
            f"Recent History: {json.dumps(short_history, ensure_ascii=False)}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            "  \"planner\": {\n"
            "    \"next_plan_step\": {\"step_id\": \"...\", \"type\": \"...\", \"description\": \"...\"}\n"
            "  },\n"
            "  \"controller\": {\n"
            "    \"chosen_action\": {\"action_id\": \"...\", \"type\": \"...\", \"parameters\": {...}, \"reason\": \"...\"}\n"
            "  }\n"
            "}"
        )

        try:
            # Rate limiting để tránh 429 (nếu có dùng)
            time.sleep(4)
            resp = self._model.generate_content(prompt_text)
            raw = resp.text or ""
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"GeminiTeacher Error: {exc}")
            return None


def build_teacher(backend: str, model_name: str) -> Optional[TeacherBase]:
    backend = backend.lower()
    if backend == "gemini":
        return GeminiTeacher(model_name=model_name)
    logger.warning(f"Unknown teacher backend: {backend}")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def determine_platform(start_url: Optional[str], platform_hint: str) -> str:
    """
    Normalize platform name cho bucket data/raw.
    """
    hint = (platform_hint or "auto").lower()
    if hint != "auto":
        return hint

    url = (start_url or "").lower()
    if "shopee" in url:
        return "shopee"
    if "lazada" in url:
        return "lazada"
    return "misc"


def prepare_raw_platform_dirs(raw_root: Path, platform: str) -> Dict[str, Path]:
    """T?o th? m?c episodes/screenshots/dom_snapshots cho platform c? th?."""
    base = ensure_dir(raw_root / platform)
    episodes = ensure_dir(base / "episodes")
    screenshots = ensure_dir(base / "screenshots")
    dom_snaps = ensure_dir(base / "dom_snapshots")
    return {"base": base, "episodes": episodes, "screenshots": screenshots, "dom": dom_snaps}


def mirror_episode_to_outcome(episode_path: Path, success: bool, trajectory_root: Path) -> None:
    """Copy episode JSON sang data/trajectories/{successful,failed}."""
    bucket = "successful" if success else "failed"
    dest_dir = ensure_dir(trajectory_root / bucket)
    dest_path = dest_dir / episode_path.name
    try:
        shutil.copy2(episode_path, dest_path)
    except Exception as exc:
        logger.warning(f"Khong mirror duoc episode sang {dest_path}: {exc}")


def load_tasks(tasks_path: Optional[str]) -> List[Tuple[str, str]]:
    """
    Đọc file tasks.jsonl (mỗi dòng: {"query": "...", "url": "..."})
    Trả về list (query, url).
    """
    if not tasks_path:
        return []
    p = Path(tasks_path)
    if not p.exists():
        raise FileNotFoundError(f"File tasks không tồn tại: {p}")

    pairs: List[Tuple[str, str]] = []
    text = p.read_text(encoding="utf-8").strip()

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "query" in row and "url" in row:
                pairs.append((row["query"], row["url"]))
        except Exception:
            logger.warning(f"Không parse được dòng task: {line!r}")
    return pairs


def build_page_state(observation: Dict[str, Any], ui_detector: UIDetector) -> Dict[str, Any]:
    """
    Tạo snapshot trạng thái trang gọn nhẹ từ observation của orchestrator.

    observation dự kiến chứa:
        - url: str
        - dom: str (HTML)
        - interactive_elements: list[...] (từ DOMDistiller)
        - vision_state: dict (từ OmniParser / VisionEnhancer – nếu có)
    """
    url = observation.get("url", "")
    dom_html = observation.get("dom", "") or ""
    elements = observation.get("interactive_elements") or []
    vision_state = observation.get("vision_state") or {}

    # Detect UI (page_type, search_box, product_listing, ...)
    ui_info: Dict[str, Any] = {"page_type": "generic"}
    if dom_html:
        try:
            # UIDetector.detect_all chỉ nhận 1 tham số: html
            ui_info = ui_detector.detect_all(dom_html) or {"page_type": "generic"}
        except Exception as e:
            logger.warning(f"UIDetector failed: {e}")
            ui_info = {"page_type": "generic"}

    return {
        "url": url,
        "page_type": ui_info.get("page_type", "generic"),
        "dom_state": {
            "element_count": len(elements),
            "has_search": bool(ui_info.get("search_box", {}).get("found")),
        },
        "elements": elements,          # bạn có thể cắt bớt nếu quá dài
        "ui_detection": ui_info,
        "vision_state": vision_state,
    }


def history_to_step_records(
    history: List[Dict[str, Any]],
    goal: str,
    ui_detector: UIDetector,
    teacher: Optional[TeacherBase],
) -> List[StepRecord]:
    """
    Convert history (ReactEngine) -> list StepRecord.

    history[i] dự kiến có:
      - timestamp
      - thought
      - action
      - result
      - observation
    """
    records: List[StepRecord] = []
    total_steps = len(history)

    logger.info(f"📝 Converting {total_steps} steps to StepRecord...")

    for idx, h in enumerate(history):
        obs = h.get("observation", {}) or {}
        page_state = build_page_state(obs, ui_detector)

        labels = None
        if teacher:
            # Teacher là optional – chỉ dùng nếu bạn bật --with_teacher
            if (idx + 1) % 2 == 0:
                logger.info(f"   Teacher labeling step {idx+1}/{total_steps}...")

            short_hist = [
                {"action": s.get("action"), "result": s.get("result")}
                for s in history[max(0, idx - 3) : idx]
            ]
            labels = teacher.annotate_step(goal, page_state, short_hist)

        records.append(
            StepRecord(
                step=idx + 1,
                timestamp=h.get("timestamp", ""),
                thought=h.get("thought", ""),
                action=h.get("action", {}),
                result=h.get("result", {}),
                page_state=page_state,
                teacher_labels=labels,
            )
        )
    return records


def execution_result_to_status(result: ExecutionResult, history_len: int, max_steps: int) -> str:
    if not result.success and result.error:
        return "ERROR"
    if result.success:
        return "SUCCESS"
    if history_len >= max_steps:
        return "TIMEOUT"
    return "FAIL"


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------


async def collect_one_episode(
    query: str,
    start_url: str,
    max_steps: int,
    headless: bool,
    user_data_dir: Optional[str],
    raw_root: Path,
    trajectory_root: Path,
    platform_hint: str,
    teacher: Optional[TeacherBase] = None,
    policy: str = "react",
) -> Optional[Path]:
    logger.info(f"\n🔹 START EPISODE: {query}")
    logger.info(f"   URL: {start_url}")
    platform_name = determine_platform(start_url, platform_hint)
    logger.info(f"   Platform bucket: {platform_name}")

    orchestrator = AgentOrchestrator(
        max_steps=max_steps,
        headless=headless,
        enable_learning=False,
        enable_guardrails=False,
        user_data_dir=user_data_dir,
    )

    episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    result_data: Optional[ExecutionResult] = None
    error_msg: Optional[str] = None

    try:
        result_data = await orchestrator.execute_task(
            query=query,
            start_url=start_url,
            policy=policy,
        )
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during execution: {e}")
        error_msg = str(e)
    finally:
        logger.info("💾 Saving trajectory data...")

        try:
            history = orchestrator.react_engine.get_history()
        except Exception:
            history = []

        if not history and not error_msg:
            logger.warning("⚠️ Empty history and no error. Nothing to save.")
            await orchestrator.close()
            return None

        try:
            ui_detector = UIDetector()
            steps = history_to_step_records(history, query, ui_detector, teacher)

            final_status = "CRASHED"
            if result_data:
                final_status = execution_result_to_status(result_data, len(history), max_steps)
            elif error_msg:
                final_status = "ERROR"

            ep = EpisodeRecord(
                episode_id=episode_id,
                goal=query,
                start_url=start_url,
                created_at=datetime.now().isoformat(),
                steps=steps,
                final_status=final_status,
                final_url=getattr(result_data, "final_url", None) or "unknown",
                success=getattr(result_data, "success", False),
                error=getattr(result_data, "error", None) or error_msg,
                summary=getattr(result_data, "summary", None) or "Interrupted",
                metadata=getattr(result_data, "metadata", None) or {},
            )

            raw_paths = prepare_raw_platform_dirs(raw_root, platform_name)
            ensure_dir(raw_paths["screenshots"] / episode_id)
            ensure_dir(raw_paths["dom"] / episode_id)
            save_path = raw_paths["episodes"] / f"{episode_id}.json"

            json_content = json.dumps(episode_to_json_dict(ep), ensure_ascii=False, indent=2)
            save_path.write_text(json_content, encoding="utf-8")

            logger.info(f"✅ Episode saved: {save_path} (Status: {final_status}, Platform: {platform_name})")
            mirror_episode_to_outcome(save_path, ep.success, trajectory_root)

            await orchestrator.close()
            return save_path

        except Exception as save_err:
            logger.error(f"🔥 Failed to save episode file: {save_err}")
            traceback.print_exc()
            await orchestrator.close()
            return None


async def async_main(args: argparse.Namespace):
    setup_logging(level=args.log_level, log_file=None)

    raw_root_path = Path(args.raw_root)
    if args.out_dir:
        logger.warning("--out_dir is deprecated. Please switch to --raw_root.")
        raw_root_path = Path(args.out_dir)
    raw_root = ensure_dir(raw_root_path)
    trajectory_root = ensure_dir(Path(args.trajectory_root))

    # Setup Teacher (optional)
    teacher: Optional[TeacherBase] = None
    if args.with_teacher:
        logger.info(f"🎓 Initializing Teacher ({args.teacher_backend})...")
        teacher = build_teacher(args.teacher_backend, args.teacher_model)

    tasks = load_tasks(args.tasks)
    if not tasks and args.query and args.url:
        tasks = [(args.query, args.url)]

    if args.episodes:
        tasks = tasks[: args.episodes]

    logger.info(f"🚀 Starting collection for {len(tasks)} tasks.")

    for i, (q, u) in enumerate(tasks, 1):
        logger.info(f"--- Task {i}/{len(tasks)} ---")
        await collect_one_episode(
            query=q,
            start_url=u,
            max_steps=args.max_steps,
            headless=args.headless,
            user_data_dir=args.user_data_dir,
            raw_root=raw_root,
            trajectory_root=trajectory_root,
            platform_hint=args.platform,
            teacher=teacher,
            policy=args.policy,
        )


def main():
    parser = argparse.ArgumentParser(description="WOA Data Collector")

    parser.add_argument("--query", help="Task query")
    parser.add_argument("--url", help="Start URL")
    parser.add_argument("--tasks", help="Path to tasks.jsonl")
    parser.add_argument("--episodes", type=int, help="Max episodes to run")
    parser.add_argument("--user_data_dir", help="Path to Chrome User Data (Profile)")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--raw_root", default="data/raw", help="Root folder for raw Shopee/Lazada data.")
    parser.add_argument(
        "--trajectory_root",
        default="data/trajectories",
        help="Folder chứa phân loại trajectories successful/failed.",
    )
    parser.add_argument(
        "--platform",
        default="auto",
        choices=["auto", "shopee", "lazada", "misc"],
        help="Bucket platform cho data/raw. auto = đoán từ start_url.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="[Deprecated] Alias cho --raw_root (sẽ bỏ trong tương lai).",
    )
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--log_level", default="INFO")

    # Teacher options (optional, có thể bỏ qua nếu không muốn dùng API)
    parser.add_argument("--with_teacher", action="store_true", help="Enable teacher model for auto labels")
    parser.add_argument("--teacher_backend", default="gemini")
    parser.add_argument("--teacher_model", default="gemini-2.0-flash")

    # Execution policy: ReAct LLM vs rule-based vs human teleop
    parser.add_argument(
        "--policy",
        default="react",
        choices=["react", "rules_shopee", "rules_lazada", "human_teleop"],
        help="Execution policy for collecting trajectories",
    )

    args = parser.parse_args()

    if not (args.tasks or (args.query and args.url)):
        print("Error: Provide --tasks OR --query and --url")
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

