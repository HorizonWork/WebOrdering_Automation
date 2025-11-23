
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.definitions import ACTION_TYPES

def verify_planner_sample(sample: Dict[str, Any], line_num: int) -> List[str]:
    errors = []
    inp = sample.get("planner_input", {})
    out = sample.get("planner_output", {})
    
    # Check High Level State
    hl_state = inp.get("high_level_state", {})
    if hl_state.get("page_type") == "unknown":
        errors.append(f"Line {line_num}: page_type is 'unknown'")
    if hl_state.get("current_step") == "UNKNOWN":
        errors.append(f"Line {line_num}: current_step is 'UNKNOWN' (page_type: {hl_state.get('page_type')})")
        
    # Check Output
    next_step = out.get("next_plan_step", {})
    if not next_step.get("reasoning"):
        errors.append(f"Line {line_num}: Missing reasoning")
    if not next_step.get("description"):
        errors.append(f"Line {line_num}: Missing description")
        
    return errors

def verify_controller_sample(sample: Dict[str, Any], line_num: int) -> List[str]:
    errors = []
    inp = sample.get("controller_input", {})
    out = sample.get("controller_output", {})
    
    # Check Page State
    page_state = inp.get("page_state", {})
    if page_state.get("page_type") == "unknown":
        errors.append(f"Line {line_num}: page_type is 'unknown'")
        
    # Check Available Actions
    actions = inp.get("available_actions_flat", [])
    if not actions:
        # It's possible to have no actions, but suspicious
        pass 
        
    for i, act in enumerate(actions):
        if not act.get("type") in ACTION_TYPES:
            errors.append(f"Line {line_num}: Action {i} has invalid type '{act.get('type')}'")
        if "params" not in act:
            errors.append(f"Line {line_num}: Action {i} missing 'params'")
            
    # Check Chosen Action
    chosen = out.get("chosen_action", {})
    if not chosen:
        errors.append(f"Line {line_num}: Missing chosen_action")
        return errors
        
    if chosen.get("id") == "unknown_id":
        errors.append(f"Line {line_num}: chosen_action.id is 'unknown_id'")
        
    if "params" not in chosen:
        errors.append(f"Line {line_num}: chosen_action missing 'params'. Content: {chosen}")
        
    if not out.get("reason"):
        errors.append(f"Line {line_num}: Missing reason")

    return errors

def verify_file(filepath: str, type: str, report_file):
    report_file.write(f"\nVerifying {filepath}...\n")
    if not Path(filepath).exists():
        report_file.write(f"File not found: {filepath}\n")
        return

    errors = []
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            count += 1
            try:
                sample = json.loads(line)
                if type == "planner":
                    errs = verify_planner_sample(sample, i + 1)
                else:
                    errs = verify_controller_sample(sample, i + 1)
                errors.extend(errs)
            except json.JSONDecodeError:
                errors.append(f"Line {i + 1}: Invalid JSON")

    report_file.write(f"Checked {count} samples.\n")
    if errors:
        report_file.write(f"Found {len(errors)} errors:\n")
        for e in errors[:20]: # Show first 20
            report_file.write(f"  - {e}\n")
        if len(errors) > 20:
            report_file.write(f"  ... and {len(errors) - 20} more.\n")
    else:
        report_file.write("✅ No errors found!\n")

if __name__ == "__main__":
    with open("verification_report.txt", "w", encoding="utf-8") as f:
        verify_file("data/processed/shopee_manual/planner/train.jsonl", "planner", f)
        verify_file("data/processed/shopee_manual/controller/train.jsonl", "controller", f)
    print("Verification complete. Check verification_report.txt")
