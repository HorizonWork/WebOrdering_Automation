import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def main():
    # Khởi tạo agent
    agent = AgentOrchestrator(
        max_steps=30,
        headless=False  # Hiển thị browser để debug
    )
    
    # Thực thi task
    result = await agent.execute_task(
        query="Tìm áo khoác nam giá dưới 500k trên Shopee",
        start_url="https://shopee.vn"
    )
    
    # In kết quả
    print(f"✅ Success: {result['success']}")
    print(f"📊 Steps: {result['steps']}")
    print(f"📝 History: {len(result['history'])} actions")

if __name__ == "__main__":
    asyncio.run(main())