"""
Query Parser - Natural Language Understanding with Ollama
Extract structured parameters from user's natural language query

Uses local Ollama models to parse queries like:
"Tôi muốn mua iPhone 15 được đánh giá trên 3 sao"
→ {query: "iPhone 15", min_rating: 3.0, ...}
"""

import json
import httpx
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedQuery:
    """Parsed query parameters"""
    query: Optional[str] = None  # Search keyword
    product_url: Optional[str] = None  # Direct product URL
    max_products: int = 10  # Max products to fetch
    min_price: Optional[float] = None  # Minimum price (VND)
    max_price: Optional[float] = None  # Maximum price (VND)
    min_rating: Optional[float] = None  # Minimum rating (1-5)
    buy_now: bool = False  # True = Buy Now, False = Add to Cart
    quantity: int = 1  # Quantity to order
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class QueryParser:
    """
    Parse natural language queries using Ollama
    
    Example:
        parser = QueryParser()
        result = parser.parse("Tôi muốn mua iPhone 15 giá dưới 20 triệu")
        # → ParsedQuery(query="iPhone 15", max_price=20000000, ...)
    """
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 30
    ):
        """
        Initialize query parser
        
        Args:
            model: Ollama model name
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        
        logger.info(f"🧠 QueryParser initialized")
        logger.info(f"   Model: {model}")
        logger.info(f"   Host: {host}")
        
    def parse(self, user_input: str) -> ParsedQuery:
        """
        Parse natural language query
        
        Args:
            user_input: User's natural language query
            
        Returns:
            ParsedQuery with extracted parameters
        """
        logger.info(f"📝 Parsing query: {user_input}")
        
        # Build prompt for LLM
        prompt = self._build_prompt(user_input)
        
        try:
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Parse JSON response
            parsed = self._parse_response(response)
            
            logger.info("✅ Query parsed successfully")
            logger.info(f"   → Search: {parsed.query}")
            logger.info(f"   → Price: [{parsed.min_price} - {parsed.max_price}]")
            logger.info(f"   → Rating: ≥{parsed.min_rating}")
            logger.info(f"   → Action: {'BUY NOW' if parsed.buy_now else 'ADD TO CART'}")
            logger.info(f"   → Quantity: {parsed.quantity}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Failed to parse query: {e}")
            logger.warning("⚠️  Using fallback: treating entire input as search query")
            
            # Fallback: treat entire input as search query
            return ParsedQuery(query=user_input)
    
    def _build_prompt(self, user_input: str) -> str:
        """Build prompt for LLM"""
        
        prompt = f"""You are a Vietnamese e-commerce query parser. Extract structured information from the user's query.

User Query: "{user_input}"

Extract the following information and return ONLY a valid JSON object (no markdown, no explanation):

{{
  "query": "search keywords (product name, brand, etc.)",
  "product_url": "direct product URL if provided, otherwise null",
  "max_products": 10,
  "min_price": minimum price in VND (number or null),
  "max_price": maximum price in VND (number or null),
  "min_rating": minimum star rating 1-5 (number or null),
  "buy_now": true if user wants to buy immediately, false for add to cart,
  "quantity": number of items (default 1)
}}

Vietnamese price keywords:
- "dưới X triệu" → max_price: X * 1000000
- "trên X triệu" → min_price: X * 1000000
- "từ X đến Y triệu" → min_price: X * 1000000, max_price: Y * 1000000
- "khoảng X triệu" → min_price: (X-1) * 1000000, max_price: (X+1) * 1000000

Vietnamese rating keywords:
- "đánh giá trên X sao" → min_rating: X
- "rating > X" → min_rating: X
- "từ X sao" → min_rating: X

Action keywords:
- "mua ngay", "buy now" → buy_now: true
- "thêm vào giỏ", "add to cart" → buy_now: false

Quantity keywords:
- "X cái", "X chiếc", "X sản phẩm" → quantity: X

Examples:
Input: "Tôi muốn mua iPhone 15 được đánh giá trên 3 sao"
Output: {{"query": "iPhone 15", "product_url": null, "max_products": 10, "min_price": null, "max_price": null, "min_rating": 3.0, "buy_now": false, "quantity": 1}}

Input: "Tìm tai nghe bluetooth giá dưới 500 nghìn"
Output: {{"query": "tai nghe bluetooth", "product_url": null, "max_products": 10, "min_price": null, "max_price": 500000, "min_rating": null, "buy_now": false, "quantity": 1}}

Input: "Mua ngay laptop gaming từ 20 đến 30 triệu rating trên 4 sao"
Output: {{"query": "laptop gaming", "product_url": null, "max_products": 10, "min_price": 20000000, "max_price": 30000000, "min_rating": 4.0, "buy_now": true, "quantity": 1}}

Now extract from the user query above. Return ONLY the JSON object, nothing else:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        logger.debug(f"🦙 Calling Ollama ({self.model})...")
        
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for structured output
                "top_p": 0.9,
                "num_predict": 256
            }
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            output = result.get("response", "")
            
            logger.debug(f"   Raw output: {output[:200]}...")
            
            return output
    
    def _parse_response(self, response: str) -> ParsedQuery:
        """Parse LLM response to ParsedQuery"""
        
        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            lines = cleaned.split('\n')
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()
        
        # Extract JSON from response
        try:
            # Try to find JSON object in response
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON object found")
            
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {cleaned}")
            raise
        
        # Create ParsedQuery from data
        parsed = ParsedQuery(
            query=data.get("query"),
            product_url=data.get("product_url"),
            max_products=data.get("max_products", 10),
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            min_rating=data.get("min_rating"),
            buy_now=data.get("buy_now", False),
            quantity=data.get("quantity", 1)
        )
        
        return parsed


# Test
if __name__ == "__main__":
    import asyncio
    
    print("="*70)
    print("🧪 QUERY PARSER TEST")
    print("="*70 + "\n")
    
    parser = QueryParser(model="llama3.2:1b")
    
    # Test cases
    test_queries = [
        "Tôi muốn mua iPhone 15 được đánh giá trên 3 sao",
        "Tìm tai nghe bluetooth giá dưới 500 nghìn",
        "Mua ngay laptop gaming từ 20 đến 30 triệu rating trên 4 sao",
        "Tìm điện thoại Samsung giá khoảng 10 triệu",
        "Áo khoác nam",
        "https://www.lazada.vn/products/iphone-15-pro-max.html",
    ]
    
    for query in test_queries:
        print("\n" + "-"*70)
        print(f"📝 Input: {query}")
        print("-"*70)
        
        try:
            result = parser.parse(query)
            print(f"\n✅ Parsed:")
            print(f"   Query: {result.query}")
            print(f"   URL: {result.product_url}")
            print(f"   Price: [{result.min_price} - {result.max_price}]")
            print(f"   Rating: ≥{result.min_rating}")
            print(f"   Action: {'BUY NOW' if result.buy_now else 'ADD TO CART'}")
            print(f"   Quantity: {result.quantity}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED")
    print("="*70)
