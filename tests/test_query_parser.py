"""Quick test for QueryParser"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.query_parser import QueryParser

print("="*70)
print("QUERY PARSER TEST")
print("="*70 + "\n")

parser = QueryParser(model="llama3.2:1b")

# Single test
query = "Tôi muốn mua iPhone 15 được đánh giá trên 3 sao"
print(f"Input: {query}\n")

result = parser.parse(query)

print("\nResult:")
print(f"  Query: {result.query}")
print(f"  Min Rating: {result.min_rating}")
print(f"  Min Price: {result.min_price}")
print(f"  Max Price: {result.max_price}")
print(f"  Buy Now: {result.buy_now}")
print(f"  Quantity: {result.quantity}")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
