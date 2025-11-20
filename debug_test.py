import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.unit.test_selectors import test_load_selectors_config, test_detect_shopee_elements, test_detect_lazada_elements, test_detect_all_integration

try:
    print("Running test_load_selectors_config...")
    test_load_selectors_config()
    print("PASS")

    print("Running test_detect_shopee_elements...")
    test_detect_shopee_elements()
    print("PASS")

    print("Running test_detect_lazada_elements...")
    test_detect_lazada_elements()
    print("PASS")

    print("Running test_detect_all_integration...")
    test_detect_all_integration()
    print("PASS")
    
    print("\nALL TESTS PASSED MANUALLY")
except Exception as e:
    print(f"\nTEST FAILED: {e}")
    import traceback
    traceback.print_exc()
