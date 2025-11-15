import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.agent_orchestrator import AgentOrchestrator


async def main() -> None:
    """Run a single agent task for quick manual smoke-testing."""
    agent = AgentOrchestrator(
        max_steps=15,
        headless=False,  # Hi?n th? browser d? debug
    )

    result = await agent.execute_task(
        query="Tìm điện thoại iPhone trên Lazada", start_url="https://www.lazada.vn",
    )

    print(f"? Success: {result.success}")
    print(f"?? Steps: {result.steps}")
    print(f"?? History: {len(result.history)} actions")
    if result.error:
        print(f"?? Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
