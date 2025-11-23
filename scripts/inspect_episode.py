
import json

file_path = r"D:\FA25\DSP\WebOrdering_Automation\data\raw\shopee\episodes\ep_20251121_165457_24e950c3.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Episode ID: {data.get('episode_id')}")
    print(f"Final Status: {data.get('final_status')}")
    
    steps = data.get('steps', [])
    print(f"Total Steps: {len(steps)}")
    
    if steps:
        first_step = steps[0]
        print("\n--- Step 1 Action ---")
        print(json.dumps(first_step.get('action'), indent=2, ensure_ascii=False))
        
        print("\n--- Step 1 Result ---")
        print(json.dumps(first_step.get('result'), indent=2, ensure_ascii=False))
        
except Exception as e:
    print(f"Error reading file: {e}")
