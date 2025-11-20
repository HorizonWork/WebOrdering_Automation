# -*- coding: utf-8 -*-
"""
Collect web interaction trajectories for training / analysis.

- G?i AgentOrchestrator ?? ch?y 1 task (query + start_url)
- L?y history t? ReactEngine
- Chuy?n history th?nh EpisodeRecord (list StepRecord)
- L?u ra JSON trong data/raw/<platform>/{episodes|screenshots|dom_snapshots}/
- Mirror episode JSON theo nh?n success/fail sang data/trajectories/{successful,failed}/

L?u ?:
    - Teacher (Gemini) l? OPTIONAL, KH?NG n?n b?t khi b?n kh?ng mu?n d?ng API.
    - UIDetector ???c d?ng ?? suy ra page_type, search_box,... t? DOM.
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
    """Convert EpisodeRecord (dataclass) th?nh dict ?? dump JSON."""
    return asdict(ep)


# ---------------------------------------------------------------------------
# Teacher LLM (OPTIONAL ? b?n c? th? b? qua ho?n to?n ph?n n?y)
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
    Wrapper cho Gemini API v?i JSON mode.

    KH?NG d?ng n?u b?n kh?ng mu?n g?i API:
    - ??ng truy?n --with_teacher
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("Thi?u th? vi?n: pip install google-generativeai") from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("?? Thi?u GEMINI_API_KEY, Teacher s? kh?ng ho?t ??ng.")
            self._model = None
            return

        genai.configure(api_key=api_key)

        # C?u h?nh model tr? v? JSON
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
            # Rate limiting ?? tr?nh 429 (n?u c? d?ng)
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
    ??c tasks t? JSONL ({"query": "...", "url": "..."}), JSON list, ho?c YAML list.
    Tr? v? list (query, url).
    """
    if not tasks_path:
        return []
    p = Path(tasks_path)
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")

    pairs: List[Tuple[str, str]] = []

    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not parse YAML tasks: {exc}")
            return pairs

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "query" in item and "url" in item:
                    pairs.append((str(item["query"]), str(item["url"])))
                else:
                    logger.warning(f"Skip invalid YAML task: {item!r}")
        else:
            logger.warning("YAML tasks file must be a list of {query,url}.")
        return pairs

    text_block = p.read_text(encoding="utf-8").strip()

    if p.suffix.lower() == ".json" and text_block:
        try:
            data = json.loads(text_block)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "query" in item and "url" in item:
                        pairs.append((item["query"], item["url"]))
                return pairs
        except Exception as exc:
            logger.warning(f"Could not parse JSON tasks: {exc}")

    for line in text_block.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "query" in row and "url" in row:
                pairs.append((row["query"], row["url"]))
        except Exception:
            logger.warning(f"Could not parse task line: {line!r}")
    return pairs


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
    logger.info(f"\n?? START EPISODE: {query}")
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
        logger.error(f"? CRITICAL ERROR during execution: {e}")
        error_msg = str(e)
    finally:
        logger.info("?? Saving trajectory data...")

        try:
            history = orchestrator.react_engine.get_history()
        except Exception:
            history = []

        if not history and not error_msg:
            logger.warning("?? Empty history and no error. Nothing to save.")
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

            logger.info(f"? Episode saved: {save_path} (Status: {final_status}, Platform: {platform_name})")
            mirror_episode_to_outcome(save_path, ep.success, trajectory_root)

            await orchestrator.close()
            return save_path

        except Exception as save_err:
            logger.error(f"?? Failed to save episode file: {save_err}")
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
    if args.out_dir:
        logger.warning("--out_dir is deprecated. Please switch to --raw_root.")
        raw_root_path = Path(args.out_dir)
    raw_root = ensure_dir(raw_root_path)
    trajectory_root = ensure_dir(Path(args.trajectory_root))

    teacher: Optional[TeacherBase] = None
    if args.with_teacher:
        logger.info(f"?? Initializing Teacher ({args.teacher_backend})...")
        try:
            teacher = build_teacher(args.teacher_backend, args.teacher_model)
        except Exception as exc:
            logger.error(f"Teacher init failed, continue without teacher: {exc}")
            teacher = None

    logger.info(f"?? Starting collection for {len(tasks)} tasks.")

    for i, (q, u) in enumerate(tasks, 1):
        logger.info(f"--- Task {i}/{len(tasks)} ---")
        if args.policy == "human_teleop":
            logger.info("Human Teleop: Forcing headless=False so you can see the browser.")
            args.headless = False

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


def main() -> None:
    parser = argparse.ArgumentParser(description="WOA Data Collector")

    parser.add_argument("--query", help="Task query")
    parser.add_argument("--url", help="Start URL")
    parser.add_argument("--tasks", help="Path to tasks.jsonl or .yaml")
    parser.add_argument("--episodes", type=int, help="Max episodes to run")
    parser.add_argument("--user_data_dir", help="Path to Chrome User Data (Profile)")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--raw_root", default="data/raw", help="Root folder for raw Shopee/Lazada data.")
    parser.add_argument(
        "--trajectory_root",
        default="data/trajectories",
        help="Folder ch?a ph?n lo?i trajectories successful/failed.",
    )
    parser.add_argument(
        "--platform",
        default="auto",
        choices=["auto", "shopee", "lazada", "misc"],
        help="Bucket platform cho data/raw. auto = ?o?n t? start_url.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="[Deprecated] Alias cho --raw_root (s? b? trong t??ng lai).",
    )
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--log_level", default="INFO")

    # Teacher options (optional, c? th? b? qua n?u kh?ng mu?n d?ng API)
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
