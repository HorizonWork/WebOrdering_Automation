"""
Vietnamese Processor - Vietnamese text processing utilities
"""

import sys
from pathlib import Path
import re
from typing import List

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VietnameseProcessor:
    """
    Vietnamese text processing utilities.
    
    **Features**:
        - Remove diacritics
        - Normalize text
        - Extract keywords
        - Clean text
    """
    
    def __init__(self):
        """Initialize Vietnamese processor"""
        # Vietnamese character mapping (with diacritics → without)
        self.vietnamese_map = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        # Common stopwords (Vietnamese)
        self.stopwords = {
            'và', 'là', 'của', 'có', 'được', 'với', 'cho', 'từ',
            'này', 'đó', 'các', 'những', 'một', 'về', 'trong', 'để'
        }
        
        logger.info("VietnameseProcessor initialized")
    
    def remove_diacritics(self, text: str) -> str:
        """
        Remove Vietnamese diacritics.
        
        Args:
            text: Vietnamese text
            
        Returns:
            Text without diacritics
        """
        result = []
        for char in text.lower():
            result.append(self.vietnamese_map.get(char, char))
        
        return ''.join(result)
    
    def normalize(self, text: str) -> str:
        """
        Normalize Vietnamese text.
        
        - Lowercase
        - Remove extra whitespace
        - Remove special characters
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Vietnamese text
            remove_stopwords: Remove stopwords
            
        Returns:
            List of keywords
        """
        # Normalize
        text = self.normalize(text)
        
        # Split into words
        words = text.split()
        
        # Remove stopwords
        if remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        
        # Remove short words
        words = [w for w in words if len(w) > 2]
        
        return words
    
    def clean_html_text(self, text: str) -> str:
        """Clean text extracted from HTML"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', text)
        
        return text.strip()
    
    def contains_vietnamese(self, text: str) -> bool:
        """Check if text contains Vietnamese characters"""
        vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        
        return any(char in vietnamese_chars for char in text.lower())


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("VietnameseProcessor - Test")
    print("=" * 70 + "\n")
    
    processor = VietnameseProcessor()
    
    # Test 1: Remove diacritics
    print("Test 1: Remove Diacritics")
    print("-" * 40)
    
    test_texts = [
        "Tìm kiếm áo khoác",
        "Đăng nhập vào hệ thống",
        "Mua sắm trực tuyến"
    ]
    
    for text in test_texts:
        result = processor.remove_diacritics(text)
        print(f"Original: {text}")
        print(f"Result:   {result}\n")
    
    # Test 2: Extract keywords
    print("\nTest 2: Extract Keywords")
    print("-" * 40)
    
    text = "Tìm kiếm các sản phẩm áo khoác giá rẻ và chất lượng tốt"
    keywords = processor.extract_keywords(text)
    print(f"Text: {text}")
    print(f"Keywords: {keywords}\n")
    
    # Test 3: Check Vietnamese
    print("\nTest 3: Check Vietnamese")
    print("-" * 40)
    
    test_texts = [
        "Hello world",
        "Xin chào",
        "123456"
    ]
    
    for text in test_texts:
        has_vn = processor.contains_vietnamese(text)
        print(f"{text}: {'yes Vietnamese' if has_vn else 'no Not Vietnamese'}")
    
    print("\n" + "=" * 70)
    print("yes VietnameseProcessor test completed!")
    print("=" * 70)
