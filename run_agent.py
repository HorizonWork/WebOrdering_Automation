"""Simple CLI entrypoint to run the WOA agent.

This script is a thin wrapper around ``AgentOrchestrator`` and is intended
for quick manual experiments (not as the main test harness).
"""

import asyncio

from src.orchestrator.agent_orchestrator import AgentOrchestrator


async def main() -> None:
    """Run a demo task against a default URL."""
    # Use default settings for most options; optionally point to local checkpoints.
    orchestrator = AgentOrchestrator(
        max_steps=None,  # fall back to config.settings.max_steps
        headless=None,  # fall back to config.settings.headless
        phobert_checkpoint="checkpoints/phobert",
        vit5_checkpoint="checkpoints/vit5",
    )

    result = await orchestrator.execute_task(
        query="Mua ao khoac nam duoi 500k",
        start_url="https://example.com",
    )

    print("Success:", result.success)
    print("Steps:", result.steps)
    print("Execution time (s):", result.execution_time)
    if result.error:
        print("Error:", result.error)


if __name__ == "__main__":
    asyncio.run(main())

