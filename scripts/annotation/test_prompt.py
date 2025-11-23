
import asyncio
import os
import sys
from pathlib import Path
import google.generativeai as genai

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

async def test_prompt():
    api_key = Path("my_keys.txt").read_text().splitlines()[0].strip()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})

    # Mock data for one step
    goal = "Mua sạc dự phòng Anker"
    original_action = "click on button.search-btn"
    dom_text = "[mmid-1] <input> (placeholder='Search in Shopee')\n[mmid-2] <button> Search"
    
    prompt = f"""
    Goal: {goal}
    Original Action: {original_action}
    DOM Context:
    {dom_text}
    
    Explain WHY the user performed this action.
    Return JSON:
    {{
        "thought": "reasoning here",
        "description": "short summary"
    }}
    """
    
    print("Sending prompt...")
    resp = await model.generate_content_async(prompt)
    print("Response:")
    print(resp.text)

if __name__ == "__main__":
    asyncio.run(test_prompt())
