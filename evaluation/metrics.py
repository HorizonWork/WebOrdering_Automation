"""Evaluation metrics helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class EpisodeMetric:
    episode_id: str
    success: bool
    steps: int
    reward: float = 0.0


def success_rate(episodes: Iterable[EpisodeMetric]) -> float:
    episodes = list(episodes)
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.success) / len(episodes)


def average_steps(episodes: Iterable[EpisodeMetric]) -> float:
    episodes = list(episodes)
    if not episodes:
        return 0.0
    return sum(ep.steps for ep in episodes) / len(episodes)


def to_dict(episodes: Iterable[EpisodeMetric]) -> List[Dict[str, object]]:
    return [ep.__dict__ for ep in episodes]


def save_metrics(path: Path, summary: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
