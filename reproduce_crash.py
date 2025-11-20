
import asyncio
from src.execution.skills.interaction import InteractionSkills

class MockPage:
    class MockKeyboard:
        async def press(self, key):
            print(f"MockPage: Pressed {key}")
    
    def __init__(self):
        self.keyboard = self.MockKeyboard()

async def test_press_crash():
    skills = InteractionSkills()
    page = MockPage()
    
    print("Testing press with key='Enter'...")
    result = await skills.press(page, key="Enter")
    print(f"Result: {result}")
    
    print("\nTesting press with missing key (should default to Enter)...")
    # This would previously crash
    try:
        result = await skills.press(page, key=None)
        print(f"Result: {result}")
        if result['status'] == 'success':
            print("SUCCESS: Crash prevented!")
        else:
            print("FAILED: Operation failed but didn't crash.")
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    asyncio.run(test_press_crash())
