"""Gemini baseline stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GeminiAgent:
    model_name: str = "gemini-1.5-flash"

    def run_task(self, task: Dict[str, str]) -> Dict[str, str]:
        # Placeholder for real Gemini integration
        return {
            "task_id": task["task_id"],
            "status": "not_implemented",
            "summary": "Gemini baseline not wired yet",
        }
