import asyncio
import json
from pathlib import Path
from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def test_history():
    """Test that history has correct format"""
    
    orchestrator = AgentOrchestrator(
        max_steps=2,
        headless=False,
    )
    
    try:
        await orchestrator.execute_task(
            query="Test",
            start_url="https://google.com",
            policy="react"
        )
        
        # Get history
        history = orchestrator.react_engine.get_history()
        
        print(f"\n📊 History Steps: {len(history)}")
        
        if history:
            step1 = history[0]
            print(f"\nyes Step 1 Keys: {list(step1.keys())}")
            print(f"   - thought: {bool(step1.get('thought'))}")
            print(f"   - action: {bool(step1.get('action'))}")
            print(f"   - observation: {bool(step1.get('observation'))}")
            print(f"   - result: {bool(step1.get('result'))}")
            
            obs = step1.get('observation', {})
            print(f"\nyes Observation Keys: {list(obs.keys())}")
            print(f"   - url: {obs.get('url', 'MISSING')}")
            print(f"   - dom: {len(obs.get('dom', ''))} chars")
            
            # Save to file
            with open('test_history.json', 'w', encoding='utf-8') as f:
                # Remove screenshot bytes before saving
                clean_history = []
                for h in history:
                    clean_h = {k: v for k, v in h.items() if k != 'screenshot'}
                    clean_history.append(clean_h)
                json.dump(clean_history, f, ensure_ascii=False, indent=2)
            
            print(f"\nyes Saved to test_history.json")
        
    finally:
        await orchestrator.close()

asyncio.run(test_history())
