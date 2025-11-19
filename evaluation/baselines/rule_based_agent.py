"""Rule-based baseline stub."""

from __future__ import annotations

from typing import Dict


class RuleBasedAgent:
    def run_task(self, task: Dict[str, str]) -> Dict[str, str]:
        return {
            "task_id": task["task_id"],
            "status": "not_implemented",
            "summary": "Rule-based heuristic TBD",
        }
