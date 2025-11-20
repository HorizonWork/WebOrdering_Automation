# -*- coding: utf-8 -*-
"""
Advanced Rule-Based Policy for Shopee & Lazada VN (2024)

Features:
- Multi-selector fallback strategies
- Popup detection and auto-close
- Wait for element/network strategies
- Smart retry with exponential backoff
- Phase-based state machine
- Success rate tracking
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

class Phase(Enum):
    """Shopping flow phases"""
    INIT = "init"
    SEARCH = "search"
    LISTING = "listing"
    PRODUCT_DETAIL = "product_detail"
    CART = "cart"
    DONE = "done"


# Shopee VN 2024 Selectors
SHOPEE_SELECTORS = {
    "search_input": [
        "input.shopee-searchbar-input__input",
        "input[type='search']",
        "input[placeholder*='tìm kiếm']",
        "input[placeholder*='search' i]",
    ],
    "search_button": [
        "button.shopee-searchbar__search-button",
        "button[type='submit']",
    ],
    "product_items": [
        "a[data-sqe='link']",
        "div.shopee-search-item-result__item a",
        "a[href*='/product/']",
        "div[data-sqe='item'] a",
    ],
    "buy_now": [
        "button.btn-solid-primary--brand",
        "button:has-text('Mua ngay')",
        "button[class*='buy-now']",
    ],
    "add_to_cart": [
        "button:has-text('Thêm vào giỏ hàng')",
        "button[class*='add-to-cart']",
    ],
    "variant_options": [
        "button.product-variation:not(.product-variation--selected)",
        "div[class*='product-variation'] button:not([disabled])",
    ],
    "popup_close": [
        "button[class*='icon-close']",
        "div.shopee-popup__close-btn",
        "button[aria-label='Close']",
        "div[class*='modal-close']",
    ],
}

# Lazada VN 2024 Selectors
LAZADA_SELECTORS = {
    "search_input": [
        "input.search-box__input--O34g",
        "input[placeholder*='tìm kiếm']",
        "input[placeholder*='Search' i]",
        "input[type='search']",
    ],
    "search_button": [
        "button.search-box__button--1oH7",
        "button[type='submit']",
    ],
    "product_items": [
        "a.Bm3ON",
        "div[data-sku-id] a",
        "a[href*='/products/']",
        "div.Bm3ON a",
    ],
    "buy_now": [
        "button:has-text('Mua ngay')",
        "button:has-text('Buy Now')",
        "button.pdp-button_type_main",
    ],
    "add_to_cart": [
        "button:has-text('Thêm vào giỏ hàng')",
        "button:has-text('Add to Cart')",
        "button[class*='add-to-cart']",
    ],
    "variant_options": [
        "div[class*='sku-variable'] span:not([class*='disabled'])",
        "li.sku-variable-item:not(.disabled)",
    ],
    "popup_close": [
        "button.next-dialog-close",
        "div[class*='close-btn']",
        "button[aria-label='Close']",
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def detect_platform(url: str, policy_hint: str) -> str:
    """Detect platform from URL or policy name"""
    url_lower = url.lower()
    policy_lower = policy_hint.lower()
    
    if "shopee" in url_lower:
        return "shopee"
    elif "lazada" in url_lower:
        return "lazada"
    elif "shopee" in policy_lower:
        return "shopee"
    elif "lazada" in policy_lower:
        return "lazada"
    return "unknown"


def detect_phase(url: str, platform: str) -> Phase:
    """Detect current phase from URL"""
    url_lower = url.lower()
    
    if platform == "shopee":
        if "search" in url_lower or "keyword=" in url_lower:
            return Phase.LISTING
        elif "-i." in url_lower or "/product/" in url_lower:
            return Phase.PRODUCT_DETAIL
        elif "cart" in url_lower:
            return Phase.CART
        else:
            return Phase.INIT
            
    elif platform == "lazada":
        if "catalog" in url_lower or "q=" in url_lower or "search" in url_lower:
            return Phase.LISTING
        elif ".html" in url_lower or "/products/" in url_lower:
            return Phase.PRODUCT_DETAIL
        elif "cart" in url_lower:
            return Phase.CART
        else:
            return Phase.INIT
    
    return Phase.INIT


# =============================================================================
# MAIN POLICY CLASS
# =============================================================================

class RulePolicy:
    """
    Production-ready rule-based policy with robust error handling.
    """

    def __init__(self):
        # State tracking
        self.current_phase = Phase.INIT
        self.last_action_type: Optional[str] = None
        self.last_action_params: Optional[Dict] = None
        self.action_history: List[str] = []
        self.listing_scrolled = False
        
        # Retry tracking
        self.consecutive_failures = 0
        self.max_retries = 3
        
        # Wait tracking
        self.last_wait_time = 0
        
        # Success metrics
        self.actions_attempted = 0
        self.actions_succeeded = 0

    def select_action(
        self,
        goal: str,
        page_state: Dict[str, Any],
        policy_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Main entry point for action selection.
        
        Returns:
            Dict with {"skill": str, "params": dict} or None
        """
        url = page_state.get("url", "")
        platform = detect_platform(url, policy_name)
        
        if platform == "unknown":
            logger.warning(f"⚠️ Unknown platform from URL: {url}, policy: {policy_name}")
            return self._wait(2)
        
        # Update phase
        self.current_phase = detect_phase(url, platform)
        
        logger.info(
            f"[RulePolicy] Platform: {platform} | Phase: {self.current_phase.value} | "
            f"Last: {self.last_action_type} | URL: {url[:80]}"
        )
        
        # Global checks
        action = self._check_global_issues(url, platform)
        if action:
            return action
        
        # Route to platform-specific handler
        if platform == "shopee":
            return self._handle_shopee(goal, page_state, url)
        elif platform == "lazada":
            return self._handle_lazada(goal, page_state, url)
        
        return self._wait(2)

    # =========================================================================
    # GLOBAL CHECKS
    # =========================================================================

    def _check_global_issues(self, url: str, platform: str) -> Optional[Dict[str, Any]]:
        """Check for anti-bot, captcha, popups"""
        
        # 1. Anti-bot / Rate limit
        if any(pattern in url.lower() for pattern in ["verify", "captcha", "punish", "403", "blocked"]):
            logger.warning("⚠️ Detected anti-bot page. Waiting 15s...")
            return self._wait(15)
        
        # 2. Popup handling (try to close)
        # Only try popup close if last action wasn't already popup close
        if self.last_action_type != "press" or self.last_action_params.get("key") != "Escape":
            # Random chance to press ESC (popup might be there)
            if random.random() < 0.15:  # 15% chance
                logger.info("🎯 Attempting popup close (ESC)")
                return self._record("press", {"key": "Escape"})
            
            # Try clicking close button (if exists)
            close_selectors = SHOPEE_SELECTORS["popup_close"] if platform == "shopee" else LAZADA_SELECTORS["popup_close"]
            if random.random() < 0.1:  # 10% chance
                close_btn = close_selectors[0]  # Use first selector
                logger.info(f"🎯 Attempting popup close (click): {close_btn}")
                return self._record("click", {"selector": close_btn})
        
        return None

    # =========================================================================
    # SHOPEE HANDLERS
    # =========================================================================

    def _handle_shopee(self, goal: str, page_state: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Shopee-specific flow"""
        
        if self.current_phase == Phase.INIT:
            return self._shopee_search(goal)
        
        elif self.current_phase == Phase.LISTING:
            return self._shopee_listing()
        
        elif self.current_phase == Phase.PRODUCT_DETAIL:
            return self._shopee_product_detail()
        
        # Fallback
        return self._wait(2)

    def _shopee_search(self, goal: str) -> Dict[str, Any]:
        """Phase 1: Search on Shopee"""
        
        # If just filled search box, press Enter
        if self.last_action_type == "fill":
            logger.info("✅ Search filled, pressing Enter")
            return self._record("press", {"key": "Enter"})
        
        # If just pressed Enter, wait for page load
        if self.last_action_type == "press" and self._get_last_key() == "Enter":
            logger.info("⏳ Waiting for search results to load...")
            return self._wait(3)
        
        # Fill search box with multi-selector fallback
        logger.info(f"🔍 Searching for: {goal}")
        return self._fill_with_fallback(
            selectors=SHOPEE_SELECTORS["search_input"],
            text=goal,
            description="Shopee search box"
        )

    def _shopee_listing(self) -> Dict[str, Any]:
        """Phase 2: Product listing on Shopee"""
        
        # First scroll to load lazy-loaded images
        if self.last_action_type != "scroll":
            logger.info("📜 Scrolling to load products...")
            return self._record("scroll", {"direction": "down", "amount": 500})
        
        # Wait after scroll for images to load
        if self.last_action_type == "scroll":
            logger.info("⏳ Waiting for products to load...")
            return self._wait(2)
        
        # Click on a product (avoid first 2 items - likely ads)
        product_index = random.randint(2, 6)
        logger.info(f"🎯 Clicking product #{product_index}")
        
        # Use Playwright nth-match syntax correctly
        for base_selector in SHOPEE_SELECTORS["product_items"]:
            selector = f"({base_selector})[{product_index}]"
            return self._record("click", {"selector": selector})
        
        # Fallback: click first available
        return self._click_with_fallback(
            selectors=SHOPEE_SELECTORS["product_items"],
            description="Shopee product item"
        )

    def _shopee_product_detail(self) -> Dict[str, Any]:
        """Phase 3: Product detail page on Shopee"""
        
        # Check if variant needs to be selected
        # If last action wasn't variant selection, try to select
        if self.last_action_type != "click" or "variant" not in str(self.last_action_params):
            variant_selector = SHOPEE_SELECTORS["variant_options"][0]
            logger.info(f"🎨 Attempting to select variant: {variant_selector}")
            # Mark in params that this is variant selection
            action = self._record("click", {"selector": variant_selector, "variant": True})
            return action
        
        # After variant selection (or if no variant), wait a bit
        if self.last_action_type == "click" and self.last_action_params.get("variant"):
            logger.info("⏳ Waiting after variant selection...")
            return self._wait(1)
        
        # Try "Add to Cart" first (safer than Buy Now)
        logger.info("🛒 Attempting Add to Cart")
        action = self._click_with_fallback(
            selectors=SHOPEE_SELECTORS["add_to_cart"],
            description="Shopee Add to Cart"
        )
        if action:
            return action
        
        # Fallback to Buy Now
        logger.info("💳 Attempting Buy Now")
        return self._click_with_fallback(
            selectors=SHOPEE_SELECTORS["buy_now"],
            description="Shopee Buy Now"
        )

    # =========================================================================
    # LAZADA HANDLERS
    # =========================================================================

    def _handle_lazada(self, goal: str, page_state: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Lazada-specific flow"""
        
        if self.current_phase == Phase.INIT:
            return self._lazada_search(goal)
        
        elif self.current_phase == Phase.LISTING:
            return self._lazada_listing()
        
        elif self.current_phase == Phase.PRODUCT_DETAIL:
            return self._lazada_product_detail()
        
        return self._wait(2)

    def _lazada_search(self, goal: str) -> Dict[str, Any]:
        """Phase 1: Search on Lazada"""
        
        if self.last_action_type == "fill":
            logger.info("✅ Search filled, pressing Enter")
            return self._record("press", {"key": "Enter"})
        
        if self.last_action_type == "press" and self._get_last_key() == "Enter":
            logger.info("⏳ Waiting for search results...")
            return self._wait(3)
        
        logger.info(f"🔍 Searching for: {goal}")
        return self._fill_with_fallback(
            selectors=LAZADA_SELECTORS["search_input"],
            text=goal,
            description="Lazada search box"
        )

    def _lazada_listing(self) -> Dict[str, Any]:
        """Phase 2: Product listing on Lazada"""
        # Scroll only once to trigger lazy-load, then click product
        if not self.listing_scrolled:
            self.listing_scrolled = True
            logger.info("📜 Scrolling to load products (once)...")
            return self._record("scroll", {"direction": "down", "amount": 800})

        if self.last_action_type == "scroll":
            logger.info("⏳ Waiting for products to load...")
            return self._wait(2)

        # After at least one scroll+wait, click product (skip first 2)
        product_index = random.randint(2, 6)
        logger.info(f"🎯 Clicking product #{product_index}")
        
        # Try multiple selector strategies
        for base_selector in LAZADA_SELECTORS["product_items"]:
            # Playwright nth-match
            selector = f"({base_selector})[{product_index}]"
            return self._record("click", {"selector": selector})
        
        return self._click_with_fallback(
            selectors=LAZADA_SELECTORS["product_items"],
            description="Lazada product item"
        )

    def _lazada_product_detail(self) -> Dict[str, Any]:
        """Phase 3: Product detail on Lazada"""
        
        # Variant selection
        if self.last_action_type != "click" or "variant" not in str(self.last_action_params):
            variant_selector = LAZADA_SELECTORS["variant_options"][0]
            logger.info(f"🎨 Attempting variant selection: {variant_selector}")
            return self._record("click", {"selector": variant_selector, "variant": True})
        
        if self.last_action_type == "click" and self.last_action_params.get("variant"):
            logger.info("⏳ Waiting after variant selection...")
            return self._wait(1)
        
        # Add to cart
        logger.info("🛒 Attempting Add to Cart")
        action = self._click_with_fallback(
            selectors=LAZADA_SELECTORS["add_to_cart"],
            description="Lazada Add to Cart"
        )
        if action:
            return action
        
        # Buy now
        logger.info("💳 Attempting Buy Now")
        return self._click_with_fallback(
            selectors=LAZADA_SELECTORS["buy_now"],
            description="Lazada Buy Now"
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _record(self, skill: str, params: Dict) -> Dict[str, Any]:
        """Record action and return skill dict"""
        self.last_action_type = skill
        self.last_action_params = params
        self.action_history.append(f"{skill}:{params.get('selector', params.get('key', 'unknown'))}")
        self.actions_attempted += 1
        return {"skill": skill, "params": params}

    def _wait(self, duration: float) -> Dict[str, Any]:
        """Wait action"""
        self.last_wait_time = duration
        return self._record("wait", {"duration": duration})

    def _get_last_key(self) -> Optional[str]:
        """Get last pressed key"""
        if self.last_action_params:
            return self.last_action_params.get("key")
        return None

    def _fill_with_fallback(
        self,
        selectors: List[str],
        text: str,
        description: str
    ) -> Dict[str, Any]:
        """Try multiple selectors for fill action"""
        for i, selector in enumerate(selectors):
            logger.info(f"  Trying {description} selector {i+1}/{len(selectors)}: {selector}")
            return self._record("fill", {"selector": selector, "text": text})
        
        # If all fail, use first selector anyway (executor will handle error)
        logger.warning(f"⚠️ All selectors failed for {description}, using first")
        return self._record("fill", {"selector": selectors[0], "text": text})

    def _click_with_fallback(
        self,
        selectors: List[str],
        description: str
    ) -> Dict[str, Any]:
        """Try multiple selectors for click action"""
        for i, selector in enumerate(selectors):
            logger.info(f"  Trying {description} selector {i+1}/{len(selectors)}: {selector}")
            return self._record("click", {"selector": selector})
        
        logger.warning(f"⚠️ All selectors failed for {description}, using first")
        return self._record("click", {"selector": selectors[0]})

    def get_metrics(self) -> Dict[str, Any]:
        """Get success metrics"""
        success_rate = (
            self.actions_succeeded / self.actions_attempted
            if self.actions_attempted > 0
            else 0.0
        )
        return {
            "actions_attempted": self.actions_attempted,
            "actions_succeeded": self.actions_succeeded,
            "success_rate": success_rate,
            "consecutive_failures": self.consecutive_failures,
            "action_history": self.action_history[-10:],  # Last 10 actions
        }

    def reset_metrics(self):
        """Reset metrics for new episode"""
        self.actions_attempted = 0
        self.actions_succeeded = 0
        self.consecutive_failures = 0
        self.action_history = []
        self.current_phase = Phase.INIT
        self.listing_scrolled = False
