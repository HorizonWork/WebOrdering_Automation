"""Quick test for demo.py with natural language input"""

# Simulate user input
import sys
from io import StringIO

# Mock stdin
test_input = "Tôi muốn mua iPhone 15 được đánh giá trên 3 sao\ny\n"
sys.stdin = StringIO(test_input)

# Run demo import test
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("="*70)
print("DEMO.PY - NATURAL LANGUAGE TEST")
print("="*70 + "\n")

try:
    from demo import prompt_user_request
    
    print("Testing natural language input...")
    print(f"Input: {test_input.split(chr(10))[0]}\n")
    
    result = prompt_user_request()
    
    print("\n" + "="*70)
    print("✅ RESULT")
    print("="*70)
    print(f"Query: {result.query}")
    print(f"Min Rating: {result.min_rating}")
    print(f"Buy Now: {result.buy_now}")
    print(f"Quantity: {result.quantity}")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
