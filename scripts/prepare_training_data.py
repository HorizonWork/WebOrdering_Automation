# scripts/prepare_training_data.py
import json
from pathlib import Path

def convert_samples_to_training_format(raw_samples_path: str, output_path: str):
    """
    Convert collected samples to training format.
    
    Expected input format (from your collection):
    {
        "trajectory": [
            {"url": "...", "action": "...", "result": "..."},
            ...
        ]
    }
    
    Output format:
    [
        {
            "query": "Tìm điện thoại iPhone",
            "state": "Trang Lazada, có search box",
            "thought": "Cần click vào ô tìm kiếm",
            "action": {"skill": "click", "params": {"selector": "#q"}}
        },
        ...
    ]
    """
    # Implement conversion logic based on your data structure
    pass
