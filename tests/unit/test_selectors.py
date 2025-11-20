import pytest
from src.perception.ui_detector import UIDetector
import yaml
from pathlib import Path

# Mock HTML for testing
SHOPEE_HTML = """
<html>
    <body>
        <h1>Welcome to Shopee</h1>
        <div class="shopee-searchbar">
            <input class="shopee-searchbar-input__input" placeholder="Search">
            <button class="btn-solid-primary">Search</button>
        </div>
        <div class="product-list">
            <div data-sqe="item">
                <a href="/product/1">Product 1</a>
                <div data-sqe="name">Product Name</div>
                <div>$10</div>
            </div>
        </div>
    </body>
</html>
"""

LAZADA_HTML = """
<html>
    <body>
        <h1>Welcome to Lazada</h1>
        <div class="lzd-search">
            <input id="q" placeholder="Search">
            <button class="search-box__button--1oH7">Search</button>
        </div>
        <div class="product-list">
            <div data-qa-locator="product-item">
                <a href="/product/1">Product 1</a>
                <span class="ooOxS">$10</span>
            </div>
        </div>
    </body>
</html>
"""

def test_load_selectors_config():
    """Test that selectors.yaml is loaded correctly"""
    detector = UIDetector()
    assert detector.selectors_config is not None
    assert "shopee" in detector.selectors_config
    assert "lazada" in detector.selectors_config
    assert "search_input" in detector.selectors_config["shopee"]

def test_detect_shopee_elements():
    """Test detection of Shopee elements using config"""
    detector = UIDetector()
    results = detector.detect_platform_specific_elements(SHOPEE_HTML, "shopee")
    
    assert results["search_input"]["found"] is True
    assert results["search_button"]["found"] is True
    assert results["product_item"]["found"] is True
    assert results["product_price"]["found"] is True

def test_detect_lazada_elements():
    """Test detection of Lazada elements using config"""
    detector = UIDetector()
    results = detector.detect_platform_specific_elements(LAZADA_HTML, "lazada")
    
    assert results["search_input"]["found"] is True
    assert results["search_button"]["found"] is True
    assert results["product_item"]["found"] is True
    assert results["product_price"]["found"] is True

def test_detect_all_integration():
    """Test integration with detect_all"""
    detector = UIDetector()
    
    # Test Shopee detection
    shopee_results = detector.detect_all(SHOPEE_HTML)
    assert "platform_specific" in shopee_results
    assert shopee_results["platform_specific"]["search_input"]["found"] is True
    
    # Test Lazada detection
    lazada_results = detector.detect_all(LAZADA_HTML)
    assert "platform_specific" in lazada_results
    assert lazada_results["platform_specific"]["search_input"]["found"] is True
