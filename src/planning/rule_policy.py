from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RulePolicy:
    """
    Advanced Heuristic Policy cho Shopee & Lazada.
    
    Logic luồng đi:
    1. Check Anti-bot/Captcha -> Wait/Error.
    2. Check Popup -> Close/Esc.
    3. Home Page -> Fill Search -> Press Enter.
    4. List Page -> Scroll (load item) -> Click Random Product.
    5. Product Page -> Select Variant (nếu có) -> Click Buy/Add Cart.
    """

    def __init__(self):
        # Bộ nhớ ngắn hạn
        self.last_action_type: Optional[str] = None
        self.last_action_params: Optional[Dict] = None
        self.consecutive_failures = 0
        
        # Initialize UIDetector to access config
        from src.perception.ui_detector import UIDetector
        self.ui_detector = UIDetector()

    def select_action(
        self,
        goal: str,
        page_state: Dict[str, Any],
        policy_name: str,
    ) -> Optional[Dict[str, Any]]:
        
        url = page_state.get("url", "")
        page_type = page_state.get("page_type", "unknown")
        
        logger.info(f"[RulePolicy] Policy: {policy_name} | URL: {url} | Last Action: {self.last_action_type}")

        # 1. Global: Phát hiện trang bị chặn/Captcha
        if "verify/traffic" in url or "punish" in url or "403" in str(page_state):
            logger.warning("⚠️ Detect Anti-bot page. Waiting...")
            return {"skill": "wait", "params": {"duration": 10}}

        # 2. Global: Xử lý Popup Shopee (Thử nhấn ESC để thoát popup quảng cáo)
        # Chỉ làm nếu bước trước không phải là ESC (tránh spam ESC)
        if "shopee" in url and self.last_action_type != "press":
            # Kiểm tra sơ bộ nếu DOM quá ngắn hoặc có dấu hiệu popup (cần logic phức tạp hơn, ở đây dùng heuristic)
            # Thỉnh thoảng nhấn ESC cho chắc
            if random.random() < 0.1: # 10% cơ hội nhấn ESC ngẫu nhiên
                 return self._record("press", {"key": "Escape"})

        # 3. Routing theo sàn
        action = None
        if "shopee" in policy_name or "shopee" in url:
            action = self._rules_shopee(goal, page_state)
        elif "lazada" in policy_name or "lazada" in url:
            action = self._rules_lazada(goal, page_state)
        else:
            logger.warning(f"Unknown policy/url. Defaulting to wait.")
            action = {"skill": "wait", "params": {"duration": 2}}

        return action

    def _record(self, skill: str, params: Dict) -> Dict:
        """Helper để lưu lại action vừa ra quyết định."""
        self.last_action_type = skill
        self.last_action_params = params
        return {"skill": skill, "params": params}

    # ------------------------------------------------------------------
    # SHOPEE INTELLIGENCE
    # ------------------------------------------------------------------
    def _rules_shopee(self, goal: str, page_state: Dict[str, Any]) -> Dict[str, Any]:
        url = page_state.get("url", "")
        
        # --- PHASE 1: SEARCH ---
        # Nếu URL chưa có 'search' và chưa có 'product' (tức là Home hoặc Category)
        if "search" not in url and "-i." not in url:
            # Logic: Nếu vừa FILL xong -> Bấm ENTER
            if self.last_action_type == "fill":
                return self._record("press", {"key": "Enter"})
            
            # Tìm ô search
            # Ưu tiên dùng selector từ config nếu có
            search_sel = "input.shopee-searchbar-input__input" # Default fallback
            if self.ui_detector.selectors_config and "shopee" in self.ui_detector.selectors_config:
                 search_sel = self.ui_detector.selectors_config["shopee"].get("search_input", search_sel)
            
            # Nếu tìm thấy hoặc chưa thử fill -> Fill
            return self._record("fill", {"selector": search_sel, "text": goal})

        # --- PHASE 2: PRODUCT LISTING ---
        if "search" in url:
            # Logic: Scroll xuống chút để load ảnh/item (Lazy load handling)
            if self.last_action_type != "scroll":
                return self._record("scroll", {"direction": "down"})

            # Tìm các item sản phẩm.
            # Class chuẩn Shopee 2024: .shopee-search-item-result__item a
            # Hoặc tìm thẻ a có data-sqe="link"
            
            # Ở đây ta trả về selector chung, Executor sẽ queryAll và chọn
            # Để thông minh, ta dùng :nth-match của Playwright hoặc chỉ định index ngẫu nhiên
            idx = random.randint(1, 4) # Chọn ngẫu nhiên từ 1 đến 4 để tránh Ads đầu tiên
            
            # Selector này trỏ vào thẻ <a> bao quanh sản phẩm
            target_sel = f"div[data-sqe='item'] >> nth={idx} >> a"
            if self.ui_detector.selectors_config and "shopee" in self.ui_detector.selectors_config:
                 base_item_sel = self.ui_detector.selectors_config["shopee"].get("product_item", "div[data-sqe='item'] a")
                 # Cần xử lý logic nth-match nếu selector base không hỗ trợ sẵn
                 # Ở đây giả định selector base trả về list items
                 target_sel = f"{base_item_sel} >> nth={idx}"
            
            return self._record("click", {"selector": target_sel})

        # --- PHASE 3: PRODUCT DETAIL (PDP) ---
        if "-i." in url: # Dấu hiệu URL sản phẩm Shopee
            # 1. Chọn phân loại (Variant) nếu chưa chọn
            # Logic này phức tạp, tạm thời ta bỏ qua hoặc chọn đại button đầu tiên
            # button.product-variation:not(.product-variation--selected)
            
            # 2. Tìm nút Mua Ngay / Thêm Giỏ
            # Class: .btn-solid-primary (Mua ngay/Thêm giỏ)
            return self._record("click", {"selector": "button.btn-solid-primary"})

        # Fallback
        return self._record("wait", {"duration": 2})

    # ------------------------------------------------------------------
    # LAZADA INTELLIGENCE
    # ------------------------------------------------------------------
    def _rules_lazada(self, goal: str, page_state: Dict[str, Any]) -> Dict[str, Any]:
        url = page_state.get("url", "")
        
        # --- PHASE 1: SEARCH ---
        if "catalog" not in url and "products" not in url:
            if self.last_action_type == "fill":
                 return self._record("press", {"key": "Enter"})
            
            # Lazada Search ID = q
            search_sel = "#q"
            if self.ui_detector.selectors_config and "lazada" in self.ui_detector.selectors_config:
                 search_sel = self.ui_detector.selectors_config["lazada"].get("search_input", search_sel)

            return self._record("fill", {"selector": search_sel, "text": goal})

        # --- PHASE 2: PRODUCT LISTING ---
        if "catalog" in url or "search" in url:
            if self.last_action_type != "scroll":
                return self._record("scroll", {"direction": "down"})
            
            # Lazada Product Item
            # Selector: div[data-qa-locator="product-item"] a
            idx = random.randint(0, 3)
            target_sel = f"div[data-qa-locator='product-item'] >> nth={idx} >> a"
            
            if self.ui_detector.selectors_config and "lazada" in self.ui_detector.selectors_config:
                 base_item_sel = self.ui_detector.selectors_config["lazada"].get("product_item", "div[data-qa-locator='product-item'] a")
                 target_sel = f"{base_item_sel} >> nth={idx}"
            
            return self._record("click", {"selector": target_sel})

        # --- PHASE 3: PRODUCT DETAIL ---
        # Lazada URL sản phẩm thường có .html ở cuối và không có catalog
        if ".html" in url:
            # Nút "Add to Cart" hoặc "Buy Now"
            # Thường là button chứa text "Buy Now" hoặc class pdp-button
            # Playwright selector text engine
            return self._record("click", {"selector": "button:has-text('Buy Now')"})

        return self._record("wait", {"duration": 2})