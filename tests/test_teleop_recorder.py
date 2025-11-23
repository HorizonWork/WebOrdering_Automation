
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Mock playwright before importing src
sys.modules["playwright"] = MagicMock()
sys.modules["playwright.async_api"] = MagicMock()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def test_teleop_recorder_logic():
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
        
        # Create Futures for inputs
        f1 = asyncio.Future()
        f1.set_result("") # First input: Enter (record)
        
        f2 = asyncio.Future()
        f2.set_result("q") # Second input: Quit
        
        mock_loop.run_in_executor.side_effect = [f1, f2]
            
        # Mock JS events for the first step
        # window.getRecordedEvents() -> returns a list of events
        mock_page.evaluate.side_effect = [
            None, # inject JS
            None, # re-inject JS (loop start)
            [     # getRecordedEvents
                {"type": "click", "selector": "button#submit", "timestamp": 123456},
                {"type": "change", "selector": "input#search", "value": "hello", "tagName": "input"}
            ],
            None # re-inject JS (step 2)
        ]
        
        await orchestrator._run_teleop_loop(mock_page, "test query", "http://test.com")
        
        # Verify add_step was called with correct action
        calls = orchestrator.react_engine.add_step.call_args_list
        assert len(calls) >= 1
        
        args, kwargs = calls[0]
        action = kwargs['action']
        
        print(f"Captured Action: {action}")
        
        assert action['skill'] == 'fill'
        assert action['params']['selector'] == 'input#search'
        assert action['params']['text'] == 'hello'

if __name__ == "__main__":
    asyncio.run(test_teleop_recorder_logic())
