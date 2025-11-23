import asyncio
import json
import argparse
from playwright.async_api import async_playwright

async def replay_episode(episode_path):
    with open(episode_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"▶️ Replaying Episode: {data['episode_id']}")
    print(f"🎯 Goal: {data['goal']}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=1000) # Slow motion để dễ nhìn
        page = await browser.new_page()
        
        # 1. Start URL
        await page.goto(data['start_url'])
        print(f"✅ Opened: {data['start_url']}")
        
        # 2. Execute Steps
        for step in data['steps']:
            action = step.get('action', {})
            act_type = action.get('type', '').upper()
            params = action.get('params', {})
            
            print(f"\n🔹 Step {step['step']}: {act_type}")
            
            try:
                if "CLICK" in act_type:
                    # Tìm element dựa trên description hoặc text (VÌ DATA THỦ CÔNG CHƯA CÓ SELECTOR CHUẨN)
                    # Đây là lúc check xem data thủ công có "đủ" để tìm lại element không
                    desc = action.get('description', '')
                    print(f"   Attempting to click based on: '{desc}'")
                    
                    # Heuristic đơn giản để tìm element từ description (Demo)
                    if "Search in Lazada" in desc:
                        await page.click("input[type='search']") 
                    elif "kính lúp" in desc:
                        await page.click(".search-box__button--1oH7")
                    elif "mũ bảo hiểm" in desc and "439k" in desc:
                         # Thử tìm text
                        await page.click("text=439.000") 
                    else:
                        print("   ⚠️ Warning: Cannot auto-replay this step without exact selector. Skipped.")

                elif "FILL" in act_type:
                    text = params.get('text', '')
                    print(f"   Filling text: '{text}'")
                    await page.fill("input[type='search']", text) # Giả sử input search
                
                elif "SCROLL" in act_type:
                    amount = params.get('amount', 500)
                    print(f"   Scrolling: {amount}px")
                    await page.mouse.wheel(0, amount)
                    
                elif "WAIT" in act_type:
                    duration = params.get('duration', 2)
                    print(f"   Waiting: {duration}s")
                    await page.wait_for_timeout(duration * 1000)
                    
            except Exception as e:
                print(f"   ❌ Action Failed: {e}")
        
        print("\n✅ Replay Finished!")
        await asyncio.sleep(5)
        await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to manual episode json")
    args = parser.parse_args()
    asyncio.run(replay_episode(args.file))
