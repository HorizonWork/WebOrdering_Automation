
import json
import os
from pathlib import Path

def create_dummy_episode():
    data = {
      "episode_id": "test_001",
      "goal": "mua iphone 15 pro max giá dưới 30 triệu",
      "steps": [
        {
          "step": 1,
          "timestamp": "2023-10-27T10:00:00",
          "thought": "(human teleop)",
          "action": {
            "skill": "fill",
            "params": {
              "selector": "input[name='q']",
              "text": "iphone 15 pro max"
            }
          },
          "result": {
            "status": "success",
            "message": "Recorded human action"
          },
          "page_state": {
            "url": "https://shopee.vn/",
            "page_type": "search",
            "dom_state": {
              "element_count": 10,
              "has_search": True,
              "filters": []
            },
            "elements": [],
            "vision_state": {}
          },
          "observation": {
            "url": "https://shopee.vn/",
            "dom": "<html>...</html>",
            "title": "Shopee"
          }
        },
        {
          "step": 2,
          "timestamp": "2023-10-27T10:00:05",
          "thought": "(human teleop)",
          "action": {
            "skill": "click",
            "params": {
              "selector": "button.search-btn"
            }
          },
          "result": {
            "status": "success",
            "message": "Recorded human action"
          },
          "page_state": {
            "url": "https://shopee.vn/search?keyword=iphone...",
            "page_type": "listing",
            "dom_state": {
              "element_count": 50,
              "has_search": True,
              "filters": [
                {
                  "id": "price_filter",
                  "label": "Khoảng Giá",
                  "actions": ["fill_min_price", "fill_max_price"]
                }
              ]
            },
            "elements": [],
            "vision_state": {}
          },
          "observation": {
            "url": "https://shopee.vn/search...",
            "dom": "<html>...</html>",
            "title": "Search Results"
          }
        }
      ]
    }
    
    output_dir = Path("data/raw/test_verification/episodes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ep_test_001.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Created dummy episode at {output_file}")

if __name__ == "__main__":
    create_dummy_episode()
