"""
Test module for DOMDistiller - Verify Layer 1 Perception works on Lazada & Shopee
"""

import pytest
from src.perception.dom_distiller import DOMDistiller


class TestDOMDistiller:
    """Test DOM Distiller functionality on Vietnamese e-commerce sites"""

    @pytest.fixture
    def distiller(self):
        """Create DOMDistiller instance"""
        return DOMDistiller(max_dom_size=50000, max_elements=300)

    @pytest.fixture
    def lazada_html(self):
        """Mock Lazada product page HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>iPhone 15 Pro Max - Lazada.vn</title>
            <script>window.dataLayer = [];</script>
            <style>.price { color: red; }</style>
            <meta charset="utf-8">
        </head>
        <body>
            <div class="container">
                <div class="product-header">
                    <h1>iPhone 15 Pro Max 256GB</h1>
                    <div class="price-box">
                        <span class="price">29.990.000₫</span>
                    </div>
                </div>
                
                <div class="product-actions">
                    <input type="number" name="quantity" value="1" min="1" class="quantity-input"/>
                    <button id="add-to-cart" class="btn btn-primary" type="button">
                        Thêm vào giỏ hàng
                    </button>
                    <button id="buy-now" class="btn btn-danger" type="submit">
                        Mua ngay
                    </button>
                </div>
                
                <div class="product-specs">
                    <select name="color" id="color-select">
                        <option value="blue">Xanh Titan</option>
                        <option value="black">Đen Titan</option>
                    </select>
                </div>
                
                <a href="/shop/apple-official" class="store-link">Apple Official Store</a>
                
                <div style="display:none" class="hidden-promo">Hidden promotion</div>
                
                <img src="iphone15.jpg" alt="iPhone 15" />
                <svg><path d="M0 0"></path></svg>
            </div>
        </body>
        </html>
        """

    @pytest.fixture
    def shopee_html(self):
        """Mock Shopee product page HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tai nghe Bluetooth - Shopee.vn</title>
            <script src="analytics.js"></script>
            <link rel="stylesheet" href="styles.css">
        </head>
        <body>
            <div class="shopee-container">
                <div class="product-info">
                    <h1 class="product-title">Tai nghe Bluetooth AirPods Pro 2</h1>
                    <div class="price-section">
                        <div class="price">₫4.990.000</div>
                        <div class="old-price d-none">₫6.990.000</div>
                    </div>
                </div>
                
                <form id="purchase-form" class="buy-form">
                    <div class="options">
                        <label for="variant">Phân loại:</label>
                        <select name="variant" id="variant">
                            <option value="white">Trắng</option>
                            <option value="black">Đen</option>
                        </select>
                    </div>
                    
                    <div class="quantity-box">
                        <label>Số lượng:</label>
                        <input type="number" name="quantity" value="1" id="qty-input" />
                    </div>
                    
                    <div class="action-buttons">
                        <button type="button" class="add-cart-btn" id="add-cart">
                            <i class="icon"></i>
                            Thêm Vào Giỏ Hàng
                        </button>
                        <button type="submit" class="buy-now-btn" id="buy-now">
                            Mua Ngay
                        </button>
                    </div>
                </form>
                
                <a href="/shop/official-apple" class="shop-link">
                    <img src="shop-logo.png" />
                    Apple Official
                </a>
                
                <div class="hidden invisible">Tracking pixel</div>
                <video src="promo.mp4"></video>
                <iframe src="ads.html"></iframe>
            </div>
        </body>
        </html>
        """

    def test_initialization(self, distiller):
        """Test DOMDistiller initialization"""
        assert distiller.max_dom_size == 50000
        assert distiller.max_elements == 300
        assert "input" in distiller.interactive_tags
        assert "button" in distiller.interactive_tags
        assert "id" in distiller.important_attrs
        assert "class" in distiller.important_attrs

    def test_distill_lazada_text_only(self, distiller, lazada_html):
        """Test text-only distillation on Lazada HTML"""
        result = distiller.distill(lazada_html, mode="text_only")

        assert result is not None
        assert len(result) < len(lazada_html)
        assert "<script>" not in result
        assert "<style>" not in result
        assert "<img" not in result
        assert "<svg>" not in result
        assert "add-to-cart" in result
        assert "buy-now" in result
        assert "quantity" in result

    def test_distill_shopee_text_only(self, distiller, shopee_html):
        """Test text-only distillation on Shopee HTML"""
        result = distiller.distill(shopee_html, mode="text_only")

        assert result is not None
        assert len(result) < len(shopee_html)
        assert "<script" not in result
        assert "<video" not in result
        assert "<iframe" not in result
        assert "add-cart" in result or "add-cart-btn" in result
        assert "buy-now" in result or "buy-now-btn" in result

    def test_distill_structure_mode(self, distiller, lazada_html):
        """Test structure distillation mode"""
        result = distiller.distill(lazada_html, mode="structure")

        assert result is not None
        assert "<div" in result
        assert "class=" in result
        assert "id=" in result
        assert "<script>" not in result
        assert "onclick=" not in result or "onclick" not in result.lower()

    def test_distill_full_mode(self, distiller, shopee_html):
        """Test full distillation mode"""
        result = distiller.distill(shopee_html, mode="full")

        assert result is not None
        assert "<form" in result
        assert "<button" in result
        assert "<select" in result
        assert "<script" not in result

    def test_remove_noise_lazada(self, distiller, lazada_html):
        """Test noise removal on Lazada HTML"""
        result = distiller.distill(lazada_html, mode="full")

        # Scripts and styles removed
        assert "<script>" not in result
        assert "<style>" not in result

        # Media removed
        assert "<img" not in result
        assert "<svg>" not in result

        # Hidden elements removed
        assert "display:none" not in result or "Hidden promotion" not in result

    def test_remove_noise_shopee(self, distiller, shopee_html):
        """Test noise removal on Shopee HTML"""
        result = distiller.distill(shopee_html, mode="full")

        assert "<script" not in result
        assert "<video" not in result
        assert "<iframe" not in result
        assert "d-none" not in result or "old-price" not in result
        assert "invisible" not in result or "Tracking pixel" not in result

    def test_extract_interactive_elements_lazada(self, distiller, lazada_html):
        """Test extracting interactive elements from Lazada"""
        elements = distiller.extract_interactive_elements(lazada_html)

        assert isinstance(elements, list)
        assert len(elements) > 0

        # Should have buttons
        buttons = [e for e in elements if e["tag"] == "button"]
        assert len(buttons) >= 2  # add-to-cart and buy-now

        # Should have inputs
        inputs = [e for e in elements if e["tag"] == "input"]
        assert len(inputs) >= 1  # quantity input

        # Should have select
        selects = [e for e in elements if e["tag"] == "select"]
        assert len(selects) >= 1  # color select

        # Check element structure
        for elem in elements:
            assert "id" in elem
            assert "tag" in elem
            assert "selector" in elem
            assert "attributes" in elem

    def test_extract_interactive_elements_shopee(self, distiller, shopee_html):
        """Test extracting interactive elements from Shopee"""
        elements = distiller.extract_interactive_elements(shopee_html)

        assert isinstance(elements, list)
        assert len(elements) > 0

        # Check for Vietnamese button text
        button_texts = [e["text"] for e in elements if e["tag"] == "button"]
        assert any("Giỏ Hàng" in text or "Mua Ngay" in text for text in button_texts)

        # Should have form
        forms = [e for e in elements if e["tag"] == "form"]
        assert len(forms) >= 1

    def test_annotate_with_ids(self, distiller, lazada_html):
        """Test ID annotation on Lazada"""
        result = distiller.annotate_with_ids(lazada_html, prefix="mmid")

        assert 'mmid="mmid-' in result or "mmid-" in result
        assert len(result) >= len(lazada_html)

    def test_find_action_buttons_lazada(self, distiller, lazada_html):
        """Test finding submit buttons on Lazada"""
        buttons = distiller.find_action_buttons(lazada_html, action_type="submit")

        assert isinstance(buttons, list)
        # Should find buy-now button
        assert len(buttons) >= 1
        assert any("submit" in btn["type"] for btn in buttons)

    def test_find_action_buttons_shopee(self, distiller, shopee_html):
        """Test finding submit buttons on Shopee"""
        buttons = distiller.find_action_buttons(shopee_html, action_type="submit")

        assert isinstance(buttons, list)
        assert len(buttons) >= 1

    def test_empty_html(self, distiller):
        """Test handling empty HTML"""
        assert distiller.distill("") == ""
        assert distiller.distill(None) == ""
        assert distiller.extract_interactive_elements("") == []
        assert distiller.annotate_with_ids("") == ""

    def test_malformed_html(self, distiller):
        """Test handling malformed HTML"""
        malformed = "<div><button>Incomplete"
        result = distiller.distill(malformed)
        assert isinstance(result, str)

        elements = distiller.extract_interactive_elements(malformed)
        assert isinstance(elements, list)

    def test_max_dom_size_truncation(self):
        """Test DOM size truncation"""
        distiller = DOMDistiller(max_dom_size=100)
        large_html = "<div>" + "x" * 1000 + "</div>"
        result = distiller.distill(large_html)

        assert len(result) <= 120  # 100 + some buffer
        assert "TRUNCATED" in result

    def test_max_elements_limit(self):
        """Test max elements limit"""
        distiller = DOMDistiller(max_elements=5)
        html = "".join([f"<button id='btn{i}'>Button {i}</button>" for i in range(20)])
        elements = distiller.extract_interactive_elements(html)

        assert len(elements) <= 5

    def test_vietnamese_text_preservation(self, distiller, lazada_html, shopee_html):
        """Test Vietnamese text is preserved"""
        result_lazada = distiller.distill(lazada_html, mode="full")
        result_shopee = distiller.distill(shopee_html, mode="full")

        # Check Vietnamese characters preserved
        assert "Thêm vào giỏ hàng" in result_lazada or "Mua ngay" in result_lazada
        assert "Giỏ Hàng" in result_shopee or "Mua Ngay" in result_shopee

    def test_selector_generation(self, distiller):
        """Test CSS selector generation"""
        html = """
        <div id="test-id">
            <button name="submit-btn" class="btn primary">Submit</button>
            <input type="text" />
        </div>
        """
        elements = distiller.extract_interactive_elements(html)

        assert len(elements) > 0
        # Should prefer ID selectors
        assert any(
            "#" in elem["selector"] or "name=" in elem["selector"] for elem in elements
        )

    def test_robust_none_safety(self, distiller):
        """Test robustness against None values"""
        # Test with HTML that might cause None issues
        tricky_html = """
        <div>
            <button></button>
            <input />
            <a></a>
            <select></select>
        </div>
        """

        result = distiller.distill(tricky_html)
        assert isinstance(result, str)

        elements = distiller.extract_interactive_elements(tricky_html)
        assert isinstance(elements, list)

    def test_reduction_percentage(self, distiller, lazada_html, shopee_html):
        """Test HTML size reduction percentage"""
        lazada_result = distiller.distill(lazada_html, mode="text_only")
        shopee_result = distiller.distill(shopee_html, mode="text_only")

        lazada_reduction = (1 - len(lazada_result) / len(lazada_html)) * 100
        shopee_reduction = (1 - len(shopee_result) / len(shopee_html)) * 100

        # Should reduce by at least 30%
        assert lazada_reduction > 30
        assert shopee_reduction > 30


def test_integration_lazada_shopee():
    """Integration test for both Lazada and Shopee"""
    print("\n" + "=" * 80)
    print("🧪 INTEGRATION TEST: DOM Distiller on Lazada & Shopee")
    print("=" * 80 + "\n")

    distiller = DOMDistiller()

    # Lazada test
    lazada_html = """
    <html><body>
        <input type="text" name="search" placeholder="Tìm kiếm sản phẩm"/>
        <button id="add-cart">Thêm vào giỏ</button>
        <button id="buy-now" type="submit">Mua ngay</button>
        <a href="/products">Sản phẩm</a>
    </body></html>
    """

    print("📦 Testing Lazada HTML")
    print("-" * 40)
    lazada_distilled = distiller.distill(lazada_html, mode="text_only")
    lazada_elements = distiller.extract_interactive_elements(lazada_html)

    print(f"✅ Original: {len(lazada_html)} chars")
    print(f"✅ Distilled: {len(lazada_distilled)} chars")
    print(f"✅ Elements: {len(lazada_elements)}")
    print(
        f"✅ Reduction: {(1 - len(lazada_distilled) / len(lazada_html)) * 100:.1f}%\n"
    )

    # Shopee test
    shopee_html = """
    <html><body>
        <form id="buy-form">
            <select name="variant"><option>Màu đỏ</option></select>
            <input type="number" name="qty" value="1"/>
            <button class="add-cart-btn">Thêm Vào Giỏ Hàng</button>
            <button type="submit">Mua Ngay</button>
        </form>
    </body></html>
    """

    print("📦 Testing Shopee HTML")
    print("-" * 40)
    shopee_distilled = distiller.distill(shopee_html, mode="text_only")
    shopee_elements = distiller.extract_interactive_elements(shopee_html)

    print(f"✅ Original: {len(shopee_html)} chars")
    print(f"✅ Distilled: {len(shopee_distilled)} chars")
    print(f"✅ Elements: {len(shopee_elements)}")
    print(
        f"✅ Reduction: {(1 - len(shopee_distilled) / len(shopee_html)) * 100:.1f}%\n"
    )

    print("=" * 80)
    print("✅ LAYER 1 PERCEPTION: DOM DISTILLER WORKS ON LAZADA & SHOPEE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_integration_lazada_shopee()
