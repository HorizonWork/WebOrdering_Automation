"""
Full Pipeline Test - End-to-end WOA Agent testing
Tests the complete flow:
1. Initialize agent
2. Load trained models (PhoBERT + ViT5)
3. Execute a task on a website
4. Check results

Usage:
    python tests/full_pipeline_test.py --task "Tìm áo khoác" --url "https://shopee.vn"
"""

import sys
from pathlib import Path
import asyncio
import argparse

# Add project root (walk up until pyproject exists)
ROOT_DIR = Path(__file__).resolve().parent
for parent in ROOT_DIR.parents:
    if (parent / "pyproject.toml").exists():
        ROOT_DIR = parent
        break
else:
    ROOT_DIR = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


async def test_full_pipeline():
    """Test full pipeline"""
    
    parser = argparse.ArgumentParser(description="Test full WOA pipeline")
    parser.add_argument("--task", default="Tìm áo khoác", help="Task to execute")
    parser.add_argument("--url", default="https://ladaza.vn", help="Starting URL")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps")
    parser.add_argument("--headless", action="store_true", help="Headless browser")
    parser.add_argument("--phobert_model", default="checkpoints/phobert", help="PhoBERT checkpoint")
    parser.add_argument("--vit5_model", default="checkpoints/vit5", help="ViT5 checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO", log_file="logs/pipeline_test.log")
    
    print("=" * 70)
    print("🚀 Full WOA Agent Pipeline Test")
    print("=" * 70 + "\n")
    
    print(f"Task: {args.task}")
    print(f"URL: {args.url}")
    print(f"Max steps: {args.max_steps}")
    print(f"PhoBERT: {args.phobert_model}")
    print(f"ViT5: {args.vit5_model}\n")
    
    # Initialize agent
    print("🔧 Initializing agent...")
    try:
        orchestrator = AgentOrchestrator(
            max_steps=args.max_steps,
            headless=args.headless,
            phobert_checkpoint=args.phobert_model,
            vit5_checkpoint=args.vit5_model,
        )
        print("✅ Agent initialized\n")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return
    
    # Execute task
    print("🎯 Executing task...")
    print("-" * 70)
    
    try:
        result = await orchestrator.execute_task(
            query=args.task,
            start_url=args.url,
        )
        
        print("-" * 70 + "\n")
        
        # Print results
        print("📊 Results:")
        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps}")
        print(f"  Time: {result.execution_time:.2f}s")
        print(f"  Final URL: {result.final_url}")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        print(f"\n📝 Summary:")
        print(f"  {result.summary}")
        
        # Print step details
        if result.history:
            print(f"\n📋 Step Details:")
            for i, step in enumerate(result.history[:5], 1):  # Show first 5
                action = step.get('action', {})
                observation = step.get('observation', {})
                
                print(f"\n  Step {i}:")
                if isinstance(action, dict):
                    print(f"    Action: {action.get('skill', 'unknown')}({action.get('params', {})})")
                else:
                    print(f"    Action: {action}")
                print(f"    Status: {step.get('result', {}).get('status', 'unknown')}")
                print(f"    URL: {observation.get('url', 'N/A')[:50]}")
            
            if len(result.history) > 5:
                print(f"\n  ... and {len(result.history) - 5} more steps")
        
        # Return success
        print("\n" + "=" * 70)
        if result.success:
            print("✅ Pipeline test PASSED!")
        else:
            print("❌ Pipeline test FAILED (but no crash)")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        print("\n🔒 Cleaning up...")
        await orchestrator.close()
        print("✅ Done")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
