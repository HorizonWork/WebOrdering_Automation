# run_agent.py
import asyncio
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.models.phobert_encoder import PhoBERTEncoder
from src.models.vit5_planner import ViT5Planner

async def main():
    # Initialize components với checkpoints đã train
    encoder = PhoBERTEncoder()
    planner = ViT5Planner(checkpoint_path="checkpoints/vit5")  # Use fine-tuned model
    
    agent = AgentOrchestrator(
        phobert_encoder=encoder,
        vit5_planner=planner
    )
    
    # Execute task
    result = await agent.execute_task(
        query="Tìm điện thoại iPhone trên Lazada",
        headless=False
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
