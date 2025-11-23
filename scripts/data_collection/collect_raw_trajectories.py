# -*- coding: utf-8 -*-
"""
Collect web interaction trajectories for training / analysis.
UPDATED VERSION with bytes serialization fix.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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

import yaml

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

from src.orchestrator.agent_orchestrator import AgentOrchestrator, ExecutionResult
from src.perception.ui_detector import UIDetector
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

UI_DETECTOR = UIDetector()
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
    assets: Dict[str, str]  # Paths to saved screenshot/dom files
    observation: Dict[str, Any]
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
# Teacher LLM
# ---------------------------------------------------------------------------


class TeacherBase:
    async def annotate_step(
        self,
        goal: str,
        page_state: Dict[str, Any],
        short_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class GeminiTeacher(TeacherBase):
    """
    Wrapper cho Gemini API với JSON mode.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("Thiếu thư viện: pip install google-generativeai") from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Silently skip if no API key
            self._model = None
            return

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"},
        )

    async def annotate_step(
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
            '  "planner": {\n'
            '    "next_plan_step": {"step_id": "...", "type": "...", "description": "..."}\n'
            "  },\n"
            '  "controller": {\n'
            '    "chosen_action": {"action_id": "...", "type": "...", "parameters": {...}, "reason": "..."}\n'
            "  }\n"
            "}"
        )

        try:
            loop = asyncio.get_running_loop()
            
            def _call_gemini():
                time.sleep(2)
                return self._model.generate_content(prompt_text)

            resp = await loop.run_in_executor(None, _call_gemini)
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
    base = ensure_dir(raw_root / platform)
    return {
        "base": base,
        "episodes": ensure_dir(base / "episodes"),
        "screenshots": ensure_dir(base / "screenshots"),
        "dom": ensure_dir(base / "dom_snapshots"),
    }


def mirror_episode_to_outcome(episode_path: Path, success: bool, trajectory_root: Path) -> None:
    bucket = "successful" if success else "failed"
    dest_dir = ensure_dir(trajectory_root / bucket)
    dest_path = dest_dir / episode_path.name
    try:
        shutil.copy2(episode_path, dest_path)
    except Exception as exc:
        logger.warning(f"Khong mirror duoc episode sang {dest_path}: {exc}")


def load_tasks(tasks_path: Optional[str]) -> List[Tuple[str, str]]:
    if not tasks_path:
        return []
    p = Path(tasks_path)
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")

    pairs: List[Tuple[str, str]] = []
    try:
        text_block = p.read_text(encoding="utf-8").strip()
        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(text_block)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        pairs.append((str(item.get("query", "")), str(item.get("url", ""))))
            return pairs
        
        # JSON Check
        if text_block.startswith("[") and text_block.endswith("]"):
            try:
                data = json.loads(text_block)
                for item in data:
                    pairs.append((item.get("query"), item.get("url")))
                return pairs
            except: pass
            
        # JSONL
        for line in text_block.splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            pairs.append((row.get("query"), row.get("url")))
            
    except Exception as e:
        logger.error(f"Error loading tasks: {e}")
        
    return pairs


def execution_result_to_status(result: Optional[ExecutionResult], steps_count: int, max_steps: int) -> str:
    """Helper để xác định trạng thái cuối cùng."""
    if not result:
        return "CRASHED"
    if getattr(result, "success", False):
        return "SUCCESS"
    if getattr(result, "error", None):
        return "ERROR"
    if steps_count >= max_steps:
        return "MAX_STEPS_REACHED"
    return "FAILED"


def clean_observation_for_json(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove bytes objects from observation dict to make it JSON-serializable.
    """
    clean_obs = {}
    for key, val in obs.items():
        # Skip bytes objects (like screenshot)
        if isinstance(val, bytes):
            continue
        # Recursively clean nested dicts
        elif isinstance(val, dict):
            clean_obs[key] = clean_observation_for_json(val)
        else:
            clean_obs[key] = val
    return clean_obs
def infer_page_type_from_url_and_ui(url: str, ui_info: Optional[Dict[str, Any]]) -> str:
    """
    Heuristic suy ra page_type từ URL + kết quả UIDetector.

    Kết quả trả về align với ví dụ trong observation_state.schema.json:
      - "search_results"
      - "product_detail"
      - "cart"
      - "checkout"
      - "login"
      - "home"
      - "unknown"
    """
    url_l = (url or "").lower()
    ui_info = ui_info or {}
    ui_page_type = (ui_info.get("page_type") or "").lower()

    # URL hard patterns ưu tiên trước
    if "cart" in url_l:
        return "cart"
    if "checkout" in url_l or "payment" in url_l:
        return "checkout"
    if "login" in url_l:
        return "login"

    # Search / listing
    if any(k in url_l for k in ["search", "catalog", "q=", "keyword="]):
        return "search_results"

    # Product detail
    if any(k in url_l for k in ["/product", "/products/", "/pdp", "-i."]) or url_l.endswith(".html"):
        return "product_detail"

    # Fallback theo UIDetector
    if "login" in ui_page_type:
        return "login"
    if "product_listing" in ui_page_type:
        return "search_results"
    if "search" in ui_page_type:
        return "search_results"

    if not ui_page_type:
        return "unknown"
    return ui_page_type


def build_page_state_from_observation(
    observation: Dict[str, Any],
    assets_meta: Dict[str, str],
    ui_detector: Optional[UIDetector] = None,
) -> Dict[str, Any]:
    """
    Build 1 snapshot page_state gần với ObservationState:
    {
      "url": ...,
      "page_type": ...,
      "dom_state": {...},
      "vision_state": {...},
      "actions": [...]
    }
    """
    url = observation.get("url", "")
    # HTML hoặc text để feed vào UIDetector
    html_or_text = (
        observation.get("dom_html")
        or observation.get("raw_html")
        or observation.get("dom")
        or ""
    )

    ui_info: Dict[str, Any] = {}
    detector = ui_detector or UI_DETECTOR
    if detector and html_or_text:
        try:
            ui_info = detector.detect_all(html_or_text) or {}
        except Exception as exc:
            logger.debug(f"UIDetector.detect_all failed: {exc}")
            ui_info = {}

    page_type = infer_page_type_from_url_and_ui(url, ui_info)

    elements = (
        observation.get("interactive_elements")
        or observation.get("elements")
        or []
    )

    # DOM-state: để simple trước, có thể enrich sau
    dom_state: Dict[str, Any] = {
        "title": observation.get("title", ""),
        "products": [],  # TODO: parse từ DOM nếu muốn
        "filters": [],   # TODO: parse từ DOM nếu muốn
    }
    # Cho teacher / debug, thêm text_excerpt (không có trong schema nhưng không sao)
    dom_text = observation.get("dom") or ""
    if dom_text:
        dom_state["text_excerpt"] = dom_text[:20000]

    vision_state: Dict[str, Any] = {
        "screenshot_id": assets_meta.get("screenshot"),
        "elements": elements,
    }

    page_state: Dict[str, Any] = {
        "url": url,
        "page_type": page_type,
        "dom_state": dom_state,
        "vision_state": vision_state,
        "actions": [],  # hiện tại chưa build DSL action list, sẽ làm sau
    }

    # Nếu muốn xem UI detection trong debug / offline thì có thể giữ thêm:
    # page_state["ui_detection"] = ui_info

    return page_state

async def process_history_and_save_assets(
    episode_id: str,
    history: List[Dict[str, Any]],
    query: str,
    raw_paths: Dict[str, Path],
    teacher: Optional[TeacherBase],
) -> List[StepRecord]:
    """
    Duyệt qua history thô:
    1. Lưu ảnh/DOM ra file.
    2. Build page_state align với ObservationState (dùng UIDetector).
    3. (Optional) Gọi Teacher để lấy teacher_labels.
    4. Tạo StepRecord sạch (không có bytes).
    """
    records: List[StepRecord] = []

    screenshot_dir = ensure_dir(raw_paths["screenshots"] / episode_id)
    dom_dir = ensure_dir(raw_paths["dom"] / episode_id)

    for i, item in enumerate(history):
        step_num = i + 1

        # Chuẩn hóa observation
        observation = item.get("observation") or {}
        if not observation:
            result_candidate = item.get("result", {})
            if isinstance(result_candidate, dict) and any(
                key in result_candidate for key in ("url", "dom", "dom_html", "screenshot")
            ):
                observation = result_candidate

        result_payload = item.get("result", {}) or {}

        # 1. Xử lý Assets
        assets_meta: Dict[str, str] = {}

        # -- Screenshot --
        screenshot = (
            item.get("screenshot")
            or observation.get("screenshot")
            or (result_payload if isinstance(result_payload, dict) else {}).get("screenshot")
        )
        if isinstance(screenshot, bytes):
            try:
                file_name = f"step_{step_num}.png"
                file_path = screenshot_dir / file_name
                file_path.write_bytes(screenshot)
                assets_meta["screenshot"] = f"{episode_id}/{file_name}"
            except Exception as e:
                logger.warning(f"Failed to save screenshot step {step_num}: {e}")

        # -- DOM Snapshot --
        dom_content = (
            observation.get("dom_html")
            or observation.get("dom")
        )
        if isinstance(dom_content, str) and dom_content:
            try:
                file_name = f"step_{step_num}.html"
                file_path = dom_dir / file_name
                file_path.write_text(dom_content, encoding="utf-8")
                assets_meta["dom"] = f"{episode_id}/{file_name}"
            except Exception as e:
                logger.warning(f"Failed to save DOM step {step_num}: {e}")

        # 2. Build page_state (align ObservationState) dùng UIDetector
        page_state = build_page_state_from_observation(
            observation=observation,
            assets_meta=assets_meta,
            ui_detector=UI_DETECTOR,
        )

        # 3. Gọi Teacher (nếu có) để sinh planner/controller labels
        teacher_label = None
        if teacher:
            short_hist = history[max(0, i - 2) : i]
            try:
                teacher_label = await teacher.annotate_step(query, page_state, short_hist)
            except Exception as e:
                logger.warning(f"Teacher labeling failed at step {step_num}: {e}")

        # 4. Tạo Record - Clean observation & result (remove bytes)
        clean_obs = clean_observation_for_json(observation)
        clean_result = clean_observation_for_json(result_payload) if isinstance(result_payload, dict) else {}

        records.append(
            StepRecord(
                step=step_num,
                timestamp=item.get("timestamp", datetime.now().isoformat()),
                thought=item.get("thought", ""),
                action=item.get("action", {}),
                result=clean_result,
                page_state=page_state,
                assets=assets_meta,
                observation=clean_obs,
                teacher_labels=teacher_label,
            )
        )

    return records

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
    logger.info(f"\n🚀 START EPISODE: {query}")
    logger.info(f"   URL: {start_url}")
    platform_name = determine_platform(start_url, platform_hint)
    
    raw_paths = prepare_raw_platform_dirs(raw_root, platform_name)
    
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
        logger.error(f"💥 CRITICAL ERROR during execution: {e}")
        error_msg = str(e)
        traceback.print_exc()
    finally:
        logger.info("💾 Processing and Saving trajectory data...")

        try:
            history = orchestrator.react_engine.get_history()
        except Exception:
            history = []

        if not history and not error_msg:
            logger.warning("⚠️ Empty history and no error. Nothing to save.")
            await orchestrator.close()
            return None

        try:
            # Xử lý history: Lưu ảnh, lưu DOM, gọi Teacher
            steps = await process_history_and_save_assets(
                episode_id=episode_id,
                history=history,
                query=query,
                raw_paths=raw_paths,
                teacher=teacher
            )

            final_status = execution_result_to_status(result_data, len(history), max_steps)
            if error_msg: 
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

            save_path = raw_paths["episodes"] / f"{episode_id}.json"
            json_content = json.dumps(episode_to_json_dict(ep), ensure_ascii=False, indent=2)
            save_path.write_text(json_content, encoding="utf-8")

            logger.info(f"yes Episode saved: {save_path} (Status: {final_status})")
            
            mirror_episode_to_outcome(save_path, ep.success, trajectory_root)

            await orchestrator.close()
            return save_path

        except Exception as save_err:
            logger.error(f"no Failed to save episode file: {save_err}")
            traceback.print_exc()
            await orchestrator.close()
            return None


async def async_main(args: argparse.Namespace) -> None:
    setup_logging(level=args.log_level, log_file=None)

    tasks = load_tasks(args.tasks)
    if not tasks and args.query and args.url:
        tasks = [(args.query, args.url)]

    if args.episodes:
        tasks = tasks[: args.episodes]

    if not tasks:
        logger.error("Khong tim thay task nao. Bo sung --tasks (jsonl/yaml) hoac --query va --url.")
        return

    raw_root_path = Path(args.raw_root)
    trajectory_root = ensure_dir(Path(args.trajectory_root))

    teacher: Optional[TeacherBase] = None
    if args.with_teacher:
        logger.info(f"🧠 Initializing Teacher ({args.teacher_backend})...")
        try:
            teacher = build_teacher(args.teacher_backend, args.teacher_model)
        except Exception as exc:
            logger.error(f"Teacher init failed, continue without teacher: {exc}")
            teacher = None

    logger.info(f"🏁 Starting collection for {len(tasks)} tasks.")

    for i, (q, u) in enumerate(tasks, 1):
        logger.info(f"--- Task {i}/{len(tasks)} ---")
        
        # Nếu dùng human_teleop, buộc phải hiện trình duyệt
        is_headless = args.headless
        if args.policy == "human_teleop":
            logger.info("Human Teleop: Forcing headless=False")
            is_headless = False

        # Check for policy-platform mismatch and warn user
        policy_platform = ""
        url_platform = determine_platform(u, "auto")
        if "shopee" in args.policy.lower():
            policy_platform = "shopee"
        elif "lazada" in args.policy.lower():
            policy_platform = "lazada"
        elif "react" in args.policy.lower():
            policy_platform = "auto"  # React policy adapts automatically
        else:
            policy_platform = "auto"  # Other policies are general purpose

        if policy_platform != "auto" and policy_platform != url_platform:
            logger.warning(f"⚠️  POLICY-PLATFORM MISMATCH: Policy '{args.policy}' for '{policy_platform}' but URL is for '{url_platform}'. This may cause failures.")
        
        await collect_one_episode(
            query=q,
            start_url=u,
            max_steps=args.max_steps,
            headless=is_headless,
            user_data_dir=args.user_data_dir,
            raw_root=raw_root_path,
            trajectory_root=trajectory_root,
            platform_hint=args.platform,
            teacher=teacher,
            policy=args.policy,
        )
        
        await asyncio.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="WOA Data Collector")

    parser.add_argument("--query", help="Task query")
    parser.add_argument("--url", help="Start URL")
    parser.add_argument("--tasks", help="Path to tasks.jsonl or .yaml")
    parser.add_argument("--episodes", type=int, help="Limit number of tasks to run")
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
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--log_level", default="INFO")

    # Teacher options
    parser.add_argument("--with_teacher", action="store_true", help="Enable teacher model for auto labels")
    parser.add_argument("--teacher_backend", default="gemini")
    parser.add_argument("--teacher_model", default="gemini-2.0-flash")

    # Execution policy
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
