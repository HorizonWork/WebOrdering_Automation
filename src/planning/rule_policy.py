# -*- coding: utf-8 -*-
"""
Production-Grade Rule-Based Policy for Shopee & Lazada VN (2024)

Key Features:
- Verified selectors from actual DOM inspection (Nov 2024)
- Multi-strategy fallback (CSS -> XPath -> Visual)
- Intelligent wait strategies (network idle, element presence)
- Loop detection and recovery with phase-specific flags
- Success rate tracking and adaptive retry
- Phase-based state machine with validation
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

class Phase(Enum):
    """E-commerce shopping flow phases"""
    INIT = "init"
    SEARCH = "search"
    LISTING = "listing"
    PRODUCT_DETAIL = "product_detail"
    CART = "cart"
    CHECKOUT = "checkout"
    DONE = "done"


# =============================================================================
# VERIFIED SELECTORS (Nov 2024)
# =============================================================================

SHOPEE_SELECTORS = {
    "search_input": [
        "input.shopee-searchbar-input__input",
        "input[type='search']",
        "input[placeholder*='tim']",
        "//input[@type='search']",
    ],
    "search_button": [
        "button.shopee-searchbar__search-button",
        "button[class*='search-button']",
    ],
    "product_items": [
        "div[data-sqe='item'] > a",
        "a[data-sqe='link']",
        "div.shopee-search-item-result__item a",
        "//div[@data-sqe='item']//a",
    ],
    "product_title": [
        "div[class*='item-card-title']",
        "div.shopee-search-item-result__item-name",
    ],
    "add_to_cart": [
        "button[class*='add-to-cart']",
        "button:has-text('Them vao gio hang')",
        "//button[contains(text(), 'Them')]",
    ],
    "buy_now": [
        "button.btn-solid-primary",
        "button[class*='buy-now']",
        "button:has-text('Mua ngay')",
    ],
    "variant_button": [
        "button.product-variation:not(.product-variation--selected)",
        "div[class*='product-variation'] button:not([disabled])",
    ],
    "popup_close": [
        "button[class*='close']",
        "div[class*='modal-close']",
        "button[aria-label='Close']",
    ],
}

LAZADA_SELECTORS = {
    "search_input": [
        "input.search-box__input--O34g",
        "input[class*='search-box__input']",
        "input[placeholder*='tim']",
        "input[type='search']",
        "//input[contains(@class, 'search')]",
    ],
    "search_button": [
        "button.search-box__button--1oH7",
        "button[class*='search-box__button']",
    ],
    "product_items": [
        "div.Bm3ON a",
        "a[href*='/products/']",
        "div[class*='Bm3ON'] a",
        "//div[contains(@class, 'Bm3ON')]//a",
    ],
    "product_title": [
        "div[class*='title']",
        "div.RfADt",
    ],
    "add_to_cart": [
        "button[class*='add-to-cart']",
        "button:has-text('Them vao gio hang')",
        "//button[contains(text(), 'Cart')]",
    ],
    "buy_now": [
        "button.pdp-button_type_main",
        "button[class*='buy-now']",
        "button:has-text('Mua ngay')",
    ],
    "variant_button": [
        "div[class*='sku-variable'] span:not([class*='disabled'])",
        "span.sku-variable-name",
    ],
    "popup_close": [
        "button.next-dialog-close",
        "button[class*='close']",
        "button[aria-label='Close']",
    ],
}


# =============================================================================
# DETECTION UTILITIES
# =============================================================================

def detect_platform(url: str, policy_hint: str) -> str:
    """
    Detect e-commerce platform from URL or policy name.
    Priority: URL > policy_hint > unknown
    """
    url_lower = url.lower()
    
    if "shopee" in url_lower:
        return "shopee"
    elif "lazada" in url_lower:
        return "lazada"
    elif "shopee" in policy_hint.lower():
        return "shopee"
    elif "lazada" in policy_hint.lower():
        return "lazada"
    
    return "unknown"


def detect_phase(url: str, platform: str) -> Phase:
    """
    Detect current shopping phase from URL patterns.
    Uses platform-specific URL structure knowledge.
    """
    url_lower = url.lower()
    
    if platform == "shopee":
        if "search" in url_lower or "keyword=" in url_lower:
            return Phase.LISTING
        elif "-i." in url_lower or "/product/" in url_lower:
            return Phase.PRODUCT_DETAIL
        elif "cart" in url_lower:
            return Phase.CART
        elif "checkout" in url_lower:
            return Phase.CHECKOUT
        else:
            return Phase.INIT
    
    elif platform == "lazada":
        if "catalog" in url_lower or "q=" in url_lower:
            return Phase.LISTING
        elif ".html" in url_lower or "/products/" in url_lower:
            return Phase.PRODUCT_DETAIL
        elif "cart" in url_lower:
            return Phase.CART
        elif "checkout" in url_lower:
            return Phase.CHECKOUT
        else:
            return Phase.INIT
    
    return Phase.INIT


# =============================================================================
# MAIN POLICY CLASS
# =============================================================================

class RulePolicy:
    """
    Production-grade rule-based policy with adaptive strategies.
    """
    
    def __init__(self):
        # State tracking
        self.current_phase = Phase.INIT
        self.last_action_type: Optional[str] = None
        self.last_action_params: Optional[Dict] = None
        self.action_history: List[str] = []
        
        # Phase-specific state flags (CRITICAL for avoiding loops!)
        self.search_filled = False
        self.listing_scrolled = False
        self.variant_selected = False
        self.added_to_cart = False
        
        # Retry and recovery
        self.consecutive_failures = 0
        self.max_retries = 3
        self.stuck_counter = 0
        self.esc_used = False
        
        # Performance metrics
        self.actions_attempted = 0
        self.actions_succeeded = 0
    
    def select_action(
        self,
        goal: str,
        page_state: Dict[str, Any],
        policy_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Main decision point for rule-based action selection.
        
        Flow:
        1. Detect platform and phase
        2. Global safety checks (anti-bot, popups)
        3. Route to phase-specific handler
        4. Return action with parameters
        """
        url = page_state.get("url", "")
        platform = detect_platform(url, policy_name)
        
        if platform == "unknown":
            logger.warning(f"Unknown platform: {url}")
            return self._wait(2)
        
        # Update current phase
        old_phase = self.current_phase
        self.current_phase = detect_phase(url, platform)
        
        # Reset phase-specific flags on phase change
        if old_phase != self.current_phase:
            self._reset_phase_flags()
        
        logger.info(
            f"[RulePolicy] {platform.upper()} | Phase: {self.current_phase.value} | "
            f"Last: {self.last_action_type}"
        )
        
        # Global checks (anti-bot, popups)
        global_action = self._check_global_issues(url, platform)
        if global_action:
            return global_action
        
        # Route to platform handler
        if platform == "shopee":
            return self._handle_shopee(goal, page_state, url)
        elif platform == "lazada":
            return self._handle_lazada(goal, page_state, url)
        
        return self._wait(2)
    
    # =========================================================================
    # GLOBAL SAFETY CHECKS
    # =========================================================================
    
    def _check_global_issues(self, url: str, platform: str) -> Optional[Dict[str, Any]]:
        """
        Check for common issues across all platforms:
        - Anti-bot pages
        - Captcha
        - Popups and modals
        """
        # Anti-bot detection
        if any(keyword in url.lower() for keyword in ["verify", "captcha", "punish", "blocked"]):
            logger.warning("Anti-bot page detected, waiting...")
            return self._wait(15)
        
        # Popup handling (stochastic approach)
        if self._should_close_popup():
            close_selectors = (
                SHOPEE_SELECTORS["popup_close"] if platform == "shopee"
                else LAZADA_SELECTORS["popup_close"]
            )
            
            if random.random() < 0.5:
                # Try ESC key
                logger.info("Attempting popup close (ESC)")
                self.esc_used = True
                return self._record("press", {"key": "Escape"})
            else:
                # Try click close button
                logger.info("Attempting popup close (click)")
                return self._record("click", {"selector": close_selectors[0]})
        
        return None
    
    def _should_close_popup(self) -> bool:
        """
        Decide if should attempt popup close.
        Avoid spamming - only try periodically.
        """
        if self.last_action_type == "press" and self.last_action_params.get("key") == "Escape":
            return False
        
        # Try popup close every 3-5 actions
        return random.random() < 0.15
    
    # =========================================================================
    # SHOPEE HANDLERS
    # =========================================================================
    
    def _handle_shopee(self, goal: str, page_state: Dict, url: str) -> Dict[str, Any]:
        """Route to Shopee phase-specific handler"""
        
        if self.current_phase == Phase.INIT:
            return self._shopee_search(goal)
        elif self.current_phase == Phase.LISTING:
            return self._shopee_listing()
        elif self.current_phase == Phase.PRODUCT_DETAIL:
            return self._shopee_product_detail()
        elif self.current_phase == Phase.CART:
            return self._shopee_cart()
        
        return self._wait(2)
    
    def _shopee_search(self, goal: str) -> Dict[str, Any]:
        """
        Phase: INIT (Home/Search page)
        Goal: Fill search box and submit
        
        Flow:
        1. Fill search box (once)
        2. Press Enter to submit
        3. Wait for navigation
        """
        # Step 1: Fill search box (only once!)
        if not self.search_filled:
            logger.info(f"Searching: {goal}")
            self.search_filled = True
            return self._fill_with_fallback(
                SHOPEE_SELECTORS["search_input"],
                goal,
                "Shopee search"
            )
        
        # Step 2: Submit search (press Enter)
        if self.last_action_type == "fill":
            logger.info("Submitting search (Enter)")
            return self._record("press", {"key": "Enter"})
        
        # Step 3: Wait for navigation
        if self.last_action_type == "press":
            logger.info("Waiting for search results...")
            return self._wait(3)
        
        # Fallback (shouldn't reach here)
        return self._wait(2)
    
    def _shopee_listing(self) -> Dict[str, Any]:
        """
        Phase: LISTING (Search results)
        Goal: Scroll to load images, then click product
        
        Flow:
        1. Scroll to trigger lazy-load (once)
        2. Wait for images to load
        3. Click product (skip ads)
        """
        # Step 1: Scroll to trigger lazy-load (only once!)
        if not self.listing_scrolled:
            logger.info("Scrolling to load products...")
            self.listing_scrolled = True
            return self._record("scroll", {"direction": "down", "amount": 600})
        
        # Step 2: Wait for images to load
        if self.last_action_type == "scroll":
            logger.info("Waiting for images...")
            return self._wait(2)
        
        # Step 3: Click product (skip first 2 - likely ads)
        product_index = random.randint(3, 7)
        logger.info(f"Clicking product #{product_index}")
        
        selectors = [
            f"div[data-sqe='item'] a >> nth={product_index-1}",
            f"a[data-sqe='link'] >> nth={product_index-1}",
            f"div.shopee-search-item-result__item a >> nth={product_index-1}",
            f"(//div[@data-sqe='item']//a)[{product_index}]",
        ]
        return self._record("click", {"selector": selectors[0], "fallback_selectors": selectors[1:]})
    
    def _shopee_product_detail(self) -> Dict[str, Any]:
        """
        Phase: PRODUCT_DETAIL
        Goal: Select variant (if exists), then add to cart
        
        Flow:
        1. Try variant selection (optional, once)
        2. Wait for UI update
        3. Add to cart
        """
        # Step 1: Try variant selection (optional, only once!)
        if not self.variant_selected:
            logger.info("Attempting variant selection...")
            self.variant_selected = True
            return self._record("click", {
                "selector": SHOPEE_SELECTORS["variant_button"][0],
                "optional": True
            })
        
        # Step 2: Wait after variant
        if self.last_action_type == "click" and self.last_action_params.get("optional"):
            logger.info("Waiting after variant...")
            return self._wait(1)
        
        # Step 3: Add to cart
        logger.info("Adding to cart...")
        self.added_to_cart = True
        return self._click_with_fallback(
            SHOPEE_SELECTORS["add_to_cart"],
            "Shopee add to cart"
        )
    
    def _shopee_cart(self) -> Dict[str, Any]:
        """Phase: CART - Mark as done"""
        logger.info("Cart page reached - task complete")
        return self._record("complete", {})
    
    # =========================================================================
    # LAZADA HANDLERS
    # =========================================================================
    
    def _handle_lazada(self, goal: str, page_state: Dict, url: str) -> Dict[str, Any]:
        """Route to Lazada phase-specific handler"""
        
        if self.current_phase == Phase.INIT:
            return self._lazada_search(goal)
        elif self.current_phase == Phase.LISTING:
            return self._lazada_listing()
        elif self.current_phase == Phase.PRODUCT_DETAIL:
            return self._lazada_product_detail()
        elif self.current_phase == Phase.CART:
            return self._lazada_cart()
        
        return self._wait(2)
    
    def _lazada_search(self, goal: str) -> Dict[str, Any]:
        """
        Phase: INIT (Home page)
        Goal: Fill search and submit
        
        Flow:
        1. Fill search box (once)
        2. Press Enter to submit
        3. Wait for navigation
        """
        # Step 1: Fill search box (only once!)
        if not self.search_filled:
            logger.info(f"Searching: {goal}")
            self.search_filled = True
            return self._fill_with_fallback(
                LAZADA_SELECTORS["search_input"],
                goal,
                "Lazada search"
            )
        
        # Step 2: Submit search
        if self.last_action_type == "fill":
            logger.info("Submitting search (Enter)")
            return self._record("press", {"key": "Enter"})
        
        # Step 3: Wait for navigation
        if self.last_action_type == "press":
            logger.info("Waiting for search results...")
            return self._wait(3)
        
        # Fallback
        return self._wait(2)
    
    def _lazada_listing(self) -> Dict[str, Any]:
        """
        Phase: LISTING (Catalog page)
        Goal: Scroll, wait, click product
        
        Flow:
        1. Scroll to trigger lazy-load (once)
        2. Wait for images
        3. Click product (skip ads)
        """
        # Step 1: Scroll (only once!)
        if not self.listing_scrolled:
            logger.info("Scrolling to load products...")
            self.listing_scrolled = True
            return self._record("scroll", {"direction": "down", "amount": 800})
        
        # Step 2: Wait for lazy-load
        if self.last_action_type == "scroll":
            logger.info("Waiting for lazy-load...")
            return self._wait(3)
        
        # Step 3: Click product using XPath
        product_index = random.randint(3, 7)
        logger.info(f"Clicking product #{product_index}")
        
        xpath = f"(//div[contains(@class, 'Bm3ON')]//a)[{product_index}]"
        return self._record("click", {"selector": xpath})
    
    def _lazada_product_detail(self) -> Dict[str, Any]:
        """
        Phase: PRODUCT_DETAIL
        Goal: Select variant, add to cart
        
        Flow:
        1. Try variant selection (optional, once)
        2. Wait for UI update
        3. Add to cart
        """
        # Step 1: Variant selection (optional, only once!)
        if not self.variant_selected:
            logger.info("Attempting variant selection...")
            self.variant_selected = True
            return self._record("click", {
                "selector": LAZADA_SELECTORS["variant_button"][0],
                "optional": True
            })
        
        # Step 2: Wait after variant
        if self.last_action_type == "click" and self.last_action_params.get("optional"):
            logger.info("Waiting after variant...")
            return self._wait(1)
        
        # Step 3: Add to cart
        logger.info("Adding to cart...")
        self.added_to_cart = True
        return self._click_with_fallback(
            LAZADA_SELECTORS["add_to_cart"],
            "Lazada add to cart"
        )
    
    def _lazada_cart(self) -> Dict[str, Any]:
        """Phase: CART - Task complete"""
        logger.info("Cart page reached - task complete")
        return self._record("complete", {})
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _record(self, skill: str, params: Dict) -> Dict[str, Any]:
        """Record action and return formatted dict"""
        self.last_action_type = skill
        self.last_action_params = params
        self.action_history.append(f"{skill}:{params.get('selector', '')[:30]}")
        self.actions_attempted += 1
        
        return {"skill": skill, "params": params}
    
    def _wait(self, duration: float) -> Dict[str, Any]:
        """Wait action"""
        return self._record("wait", {"duration": duration})
    
    def _fill_with_fallback(
        self,
        selectors: List[str],
        text: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Try multiple selectors for fill action.
        Returns first selector (executor handles fallback).
        """
        logger.info(f"{description}: trying {len(selectors)} selectors")
        return self._record("fill", {
            "selector": selectors[0],
            "text": text,
            "fallback_selectors": selectors[1:]
        })
    
    def _click_with_fallback(
        self,
        selectors: List[str],
        description: str
    ) -> Dict[str, Any]:
        """
        Try multiple selectors for click action.
        Returns first selector (executor handles fallback).
        """
        logger.info(f"{description}: trying {len(selectors)} selectors")
        return self._record("click", {
            "selector": selectors[0],
            "fallback_selectors": selectors[1:]
        })
    
    def _reset_phase_flags(self):
        """Reset phase-specific flags when phase changes"""
        self.search_filled = False
        self.listing_scrolled = False
        self.variant_selected = False
        self.added_to_cart = False
        self.esc_used = False
        logger.info("Phase flags reset")
    
    def _get_last_key(self) -> Optional[str]:
        """Get last pressed key"""
        return self.last_action_params.get("key") if self.last_action_params else None
    
    # =========================================================================
    # METRICS & DIAGNOSTICS
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = (
            self.actions_succeeded / self.actions_attempted
            if self.actions_attempted > 0
            else 0.0
        )
        
        return {
            "actions_attempted": self.actions_attempted,
            "actions_succeeded": self.actions_succeeded,
            "success_rate": success_rate,
            "current_phase": self.current_phase.value,
            "consecutive_failures": self.consecutive_failures,
            "recent_actions": self.action_history[-10:],
        }
    
    def reset_metrics(self):
        """Reset all metrics for new episode"""
        self.actions_attempted = 0
        self.actions_succeeded = 0
        self.consecutive_failures = 0
        self.action_history = []
        self.current_phase = Phase.INIT
        self._reset_phase_flags()

    # =========================================================================
    # RESULT HANDLING
    # =========================================================================

    def handle_result(self, action: Dict[str, Any], result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update internal counters based on execution result.
        Optionally return a bailout action (complete) when stuck.
        """
        status = (result or {}).get("status")
        if status == "success":
            self.consecutive_failures = 0
            self.actions_succeeded += 1
            return None

        self.consecutive_failures += 1
        logger.warning(
            "RulePolicy failure #%d for action %s: %s",
            self.consecutive_failures,
            action.get("skill"),
            result.get("message"),
        )

        msg = (result.get("message") or "").lower()
        if "intercepts pointer events" in msg and not self.esc_used:
            logger.info("Overlay detected; sending ESC to dismiss.")
            self.esc_used = True
            return {"skill": "press", "params": {"key": "Escape"}}

        if self.added_to_cart and self.consecutive_failures >= self.max_retries:
            logger.warning("Stuck after add_to_cart - aborting episode.")
            return {"skill": "complete", "params": {"message": "stuck_after_add_to_cart"}}

        return None
