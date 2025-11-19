"""GPT-4 baseline stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GPT4Agent:
    model_name: str = "gpt-4o"

    def run_task(self, task: Dict[str, str]) -> Dict[str, str]:
        return {
            "task_id": task["task_id"],
            "status": "not_implemented",
            "summary": "GPT-4 baseline not wired yet",
        }
