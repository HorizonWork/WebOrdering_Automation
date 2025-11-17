"""
Test various Vietnamese natural language queries with QueryParser.
"""
from src.utils.logger import setup_logging, get_logger
from src.utils.query_parser import QueryParser

logger = get_logger(__name__)

def test_queries():
    """Test various natural language query patterns."""
    setup_logging()
    
    # Initialize QueryParser
    parser = QueryParser(model="llama3.2:1b")
    
    # Test queries
    test_cases = [
        # Simple search
        "tai nghe bluetooth",
        
        # With price filter
        "Tìm tai nghe bluetooth giá dưới 500 nghìn",
        "Laptop từ 20 đến 30 triệu",
        
        # With rating filter  
        "Tôi muốn mua iPhone 15 được đánh giá trên 3 sao",
        "Điện thoại Samsung rating trên 4 sao",
        
        # Complex query
        "Mua ngay 2 cái laptop gaming từ 20 đến 30 triệu rating trên 4 sao",
        
        # Buy now vs add to cart
        "Thêm vào giỏ hàng tai nghe AirPods",
        "Mua ngay iPhone 15 Pro Max",
        
        # English
        "Find bluetooth headphones under 1 million VND",
        
        # URL
        "https://www.lazada.vn/products/iphone-15.html",
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING NATURAL LANGUAGE QUERY PARSER")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"📝 Test {i}/{len(test_cases)}")
        print(f"{'─'*80}")
        print(f"Input: {query}")
        print()
        
        # Parse query
        result = parser.parse(query)
        
        # Display results
        print("✅ Extracted Parameters:")
        print(f"   🔍 Search Query: {result.query or '(none)'}")
        if result.product_url:
            print(f"   🔗 Product URL: {result.product_url}")
        if result.min_price:
            print(f"   💰 Min Price: {result.min_price:,.0f} VND")
        if result.max_price:
            print(f"   💰 Max Price: {result.max_price:,.0f} VND")
        if result.min_rating:
            print(f"   ⭐ Min Rating: {result.min_rating} stars")
        print(f"   🛒 Action: {'BUY NOW' if result.buy_now else 'ADD TO CART'}")
        print(f"   📦 Quantity: {result.quantity}")
        print()
    
    print("="*80)
    print(f"✅ ALL {len(test_cases)} TESTS COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_queries()
