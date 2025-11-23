
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Mock playwright before importing src
sys.modules["playwright"] = MagicMock()
sys.modules["playwright.async_api"] = MagicMock()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def test_enhanced_teleop_logic():
    # Mock dependencies
    with patch("src.orchestrator.agent_orchestrator.AgentOrchestrator._perceive") as mock_perceive, \
         patch("asyncio.get_event_loop") as mock_get_loop:
        
        # Setup Orchestrator with mocks
        orchestrator = AgentOrchestrator(headless=True, enable_guardrails=False)
        orchestrator.browser_manager = AsyncMock()
        orchestrator.react_engine = MagicMock()
        orchestrator.state_manager = MagicMock()
        
        # Mock Page
        mock_page = AsyncMock()
        
        # Mock _perceive return
        mock_perceive.return_value = {
            "url": "http://test.com",
            "dom": "<html></html>",
            "screenshot": b"",
            "elements": []
        }
        
        # Mock asyncio loop and run_in_executor
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        # Scenario:
        # 1. Scroll event (JS)
        # 2. Manual 'wait 5' command
        # 3. Manual 'captcha' command
        # 4. Quit
        
        f1 = asyncio.Future(); f1.set_result("")        # Step 1: Enter (record scroll)
        f2 = asyncio.Future(); f2.set_result("wait 5")  # Step 2: Wait command
        f3 = asyncio.Future(); f3.set_result("captcha") # Step 3: Captcha command
        f5 = asyncio.Future(); f5.set_result("")        # Step 5: Enter (click + scroll noise)
        f6 = asyncio.Future(); f6.set_result("q")       # Step 6: Quit
        f7 = asyncio.Future(); f7.set_result("y")       # Confirm success
        
        mock_loop.run_in_executor.side_effect = [f1, f2, f3, f5, f6, f7]
            
        # Mock JS events
        mock_page.evaluate.side_effect = [
            None, # inject JS
            None, # re-inject JS (step 1)
            [     # getRecordedEvents (step 1)
                {"type": "scroll", "scrollX": 0, "scrollY": 500, "timestamp": 123456}
            ],
            None, # re-inject JS (step 2)
            [],   # getRecordedEvents (step 2 - empty, user typed wait)
            None, # re-inject JS (step 3)
            [],   # getRecordedEvents (step 3 - empty, user typed captcha)
            None, # re-inject JS (step 5)
            [     # getRecordedEvents (step 5 - click + scroll noise)
                {"type": "click", "selector": "#btn", "timestamp": 123457},
                {"type": "scroll", "scrollX": 0, "scrollY": 0, "timestamp": 123458}
            ],
            None  # re-inject JS (step 6)
        ]
        
        await orchestrator._run_teleop_loop(mock_page, "test query", "http://test.com")
        
        # Verify add_step calls
        calls = orchestrator.react_engine.add_step.call_args_list
        assert len(calls) >= 4
        
        # Check Step 1: Scroll
        action1 = calls[0][1]['action']
        assert action1['skill'] == 'scroll'
        
        # Check Step 4: Click (should prioritize click over scroll noise)
        action4 = calls[3][1]['action']
        print(f"Step 4 Action: {action4}")
        assert action4['skill'] == 'click'
        assert action4['params']['selector'] == '#btn'

if __name__ == "__main__":
    asyncio.run(test_enhanced_teleop_logic())
