"""
Safety Guardrails - Security & Safety Constraints
Prevents agent from accessing sensitive sites or performing risky actions
Inspired by OpenAI Operator's safety-first design
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class GuardrailViolation:
    """Violation of a guardrail rule"""
    rule: str
    severity: str  # 'high', 'medium', 'low'
    message: str
    blocked_value: str


class SafetyGuardrails:
    """
    Safety guardrails for web automation agent.
    
    **Safety Principles** (from OpenAI Operator):
        1. Human-in-the-loop for sensitive actions
        2. Restricted access to sensitive websites
        3. Action validation before execution
        4. Timeout and resource limits
        5. User confirmation for high-risk operations
    
    **Guardrails**:
        - ❌ Block banking websites
        - ❌ Block healthcare/medical sites
        - ❌ Block government portals (payment)
        - ❌ Block adult content
        - ⚠️  Confirm before payments > threshold
        - ⚠️  Confirm before account changes
        - ⚠️  Limit file uploads
        - ✅ Allow e-commerce (Shopee, Lazada, etc.)
    """
    
    def __init__(
        self,
        strict_mode: bool = True,
        payment_threshold: float = 1_000_000  # VND
    ):
        """
        Initialize safety guardrails.
        
        Args:
            strict_mode: Enable strict safety checks
            payment_threshold: Max payment without confirmation (VND)
        """
        self.strict_mode = strict_mode
        self.payment_threshold = payment_threshold
        
        # Define blocked domains
        self._init_blocked_domains()
        
        # Define sensitive actions
        self._init_sensitive_actions()
        
        # Define high-risk patterns
        self._init_risk_patterns()
        
        logger.info("🛡️  Safety Guardrails initialized")
        logger.info(f"   Strict mode: {self.strict_mode}")
        logger.info(f"   Payment threshold: {self.payment_threshold:,.0f} VND")
        logger.info(f"   Blocked domains: {len(self.blocked_domains)}")
    
    def _init_blocked_domains(self):
        """Initialize blocked domain patterns"""
        self.blocked_domains: Set[str] = {
            # Banking (Vietnam)
            'vietcombank.com.vn',
            'techcombank.com.vn',
            'mbbank.com.vn',
            'vpbank.com.vn',
            'acb.com.vn',
            'bidv.com.vn',
            'agribank.com.vn',
            'sacombank.com.vn',
            'tpbank.com.vn',
            'hdbank.com.vn',
            
            # International Banking
            'chase.com',
            'bankofamerica.com',
            'wellsfargo.com',
            'citibank.com',
            'hsbc.com',
            
            # Government
            'gdt.gov.vn',  # Tax
            'mof.gov.vn',  # Finance Ministry
            'mpi.gov.vn',  # Planning & Investment
            'dichvucong.gov.vn',  # Public services
            
            # Healthcare
            'vinmec.com',
            'benhvienbachmai.vn',
            'fv.com.vn',
            'columbia.com.vn',
            
            # Adult Content (basic patterns)
            'pornhub.com',
            'xvideos.com',
            'xnxx.com'
        }
        
        # Domain patterns (regex)
        self.blocked_patterns: List[str] = [
            r'.*bank.*\.vn$',  # Any Vietnamese bank
            r'.*\.gov\.vn$',  # Government sites
            r'.*hospital.*',  # Hospitals
            r'.*casino.*',  # Gambling
            r'.*betting.*'  # Betting sites
        ]
    
    def _init_sensitive_actions(self):
        """Initialize sensitive action patterns"""
        self.sensitive_actions = {
            'payment': ['pay', 'checkout', 'purchase', 'buy', 'thanh toán', 'mua'],
            'account': ['delete', 'close', 'deactivate', 'xóa tài khoản'],
            'transfer': ['transfer', 'send money', 'chuyển tiền'],
            'withdrawal': ['withdraw', 'rút tiền'],
            'personal_info': ['change password', 'update phone', 'đổi mật khẩu']
        }
    
    def _init_risk_patterns(self):
        """Initialize high-risk text patterns"""
        self.risk_patterns = {
            'password': re.compile(r'password|mật khẩu|pass', re.IGNORECASE),
            'credit_card': re.compile(r'credit card|thẻ tín dụng|card number', re.IGNORECASE),
            'social_security': re.compile(r'ssn|social security|cmnd|cccd', re.IGNORECASE),
            'bank_account': re.compile(r'bank account|số tài khoản', re.IGNORECASE)
        }
    
    def check_url_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed, False if blocked
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check exact match
        if domain in self.blocked_domains:
            logger.warning(f"🚫 URL blocked (exact match): {url}")
            return False
        
        # Check patterns
        for pattern in self.blocked_patterns:
            if re.match(pattern, domain):
                logger.warning(f"🚫 URL blocked (pattern match): {url}")
                return False
        
        # Allowed
        logger.debug(f"✅ URL allowed: {url}")
        return True
    
    def check_action_allowed(
        self,
        action: Dict,
        require_confirmation: bool = False
    ) -> bool:
        """
        Check if action is allowed.
        
        Args:
            action: Action dict with {skill, params}
            require_confirmation: Whether to require user confirmation
            
        Returns:
            True if allowed, False if blocked
        """
        skill = action.get('skill', '')
        params = action.get('params', {})
        
        # Always allow observation actions
        if skill in ['screenshot', 'get_dom', 'get_text', 'wait_for']:
            return True
        
        # Check for sensitive keywords in parameters
        param_str = str(params).lower()
        
        for category, keywords in self.sensitive_actions.items():
            for keyword in keywords:
                if keyword in param_str:
                    logger.warning(f"⚠️  Sensitive action detected: {category} - {keyword}")
                    
                    if self.strict_mode:
                        logger.warning("🚫 Action blocked in strict mode")
                        return False
                    
                    if require_confirmation:
                        logger.info("🔔 Action requires confirmation")
                        # In production, this would prompt user
                        # For now, block in strict mode
                        return not self.strict_mode
        
        # Check for high-risk patterns
        for risk_type, pattern in self.risk_patterns.items():
            if pattern.search(param_str):
                logger.warning(f"⚠️  High-risk pattern detected: {risk_type}")
                
                if self.strict_mode:
                    logger.warning("🚫 Action blocked due to risk pattern")
                    return False
        
        # Allowed
        return True
    
    def check_payment_amount(self, amount: float) -> bool:
        """
        Check if payment amount is within threshold.
        
        Args:
            amount: Payment amount (VND)
            
        Returns:
            True if within threshold, False if requires confirmation
        """
        if amount > self.payment_threshold:
            logger.warning(f"⚠️  Payment exceeds threshold: {amount:,.0f} VND > {self.payment_threshold:,.0f} VND")
            return False
        
        return True
    
    def validate_input_text(self, text: str) -> bool:
        """
        Validate input text for sensitive information.
        
        Args:
            text: Text to validate
            
        Returns:
            True if safe, False if contains sensitive info
        """
        for risk_type, pattern in self.risk_patterns.items():
            if pattern.search(text):
                logger.warning(f"⚠️  Sensitive information detected in input: {risk_type}")
                return False
        
        return True
    
    def get_allowed_domains(self) -> List[str]:
        """Get list of explicitly allowed domains (whitelist)"""
        allowed = [
            'shopee.vn',
            'lazada.vn',
            'tiki.vn',
            'sendo.vn',
            'thegioididong.com',
            'fpt.vn',
            'viettel.vn',
            'example.com',  # For testing
            'google.com',
            'facebook.com'
        ]
        return allowed
    
    def is_domain_whitelisted(self, url: str) -> bool:
        """Check if domain is in whitelist"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        for allowed in self.get_allowed_domains():
            if allowed in domain:
                return True
        
        return False
    
    def get_violation_report(self, violations: List[GuardrailViolation]) -> str:
        """Generate violation report"""
        if not violations:
            return "✅ No guardrail violations"
        
        report = "🚫 Guardrail Violations:\n"
        for i, v in enumerate(violations, 1):
            report += f"\n{i}. [{v.severity.upper()}] {v.rule}\n"
            report += f"   Message: {v.message}\n"
            report += f"   Blocked: {v.blocked_value}\n"
        
        return report
    
    def suggest_alternative(self, blocked_action: Dict) -> Optional[Dict]:
        """Suggest alternative action if one is blocked"""
        skill = blocked_action.get('skill')
        
        # Suggest screenshot instead of risky click
        if skill == 'click':
            return {
                'skill': 'screenshot',
                'params': {'full_page': True},
                'reason': 'Alternative: Take screenshot instead of risky click'
            }
        
        # Suggest observation instead of risky type
        if skill == 'type':
            return {
                'skill': 'get_dom',
                'params': {},
                'reason': 'Alternative: Observe page instead of risky input'
            }
        
        return None
    
    def get_safety_summary(self) -> Dict:
        """Get safety configuration summary"""
        return {
            'strict_mode': self.strict_mode,
            'payment_threshold': self.payment_threshold,
            'blocked_domains_count': len(self.blocked_domains),
            'blocked_patterns_count': len(self.blocked_patterns),
            'sensitive_actions_count': len(self.sensitive_actions),
            'risk_patterns_count': len(self.risk_patterns)
        }


# Test & Usage Examples
if __name__ == "__main__":
    print("=" * 70)
    print("Safety Guardrails - Test")
    print("=" * 70 + "\n")
    
    # Initialize
    guardrails = SafetyGuardrails(strict_mode=True)
    
    # Print summary
    print("Safety Configuration:")
    summary = guardrails.get_safety_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test 1: URL checking
    print("\n" + "=" * 70)
    print("Test 1: URL Checking")
    print("=" * 70)
    
    test_urls = [
        "https://shopee.vn",
        "https://vietcombank.com.vn",
        "https://example.com",
        "https://mbbank.com.vn",
        "https://tiki.vn"
    ]
    
    for url in test_urls:
        allowed = guardrails.check_url_allowed(url)
        status = "✅ Allowed" if allowed else "🚫 Blocked"
        print(f"{status}: {url}")
    
    # Test 2: Action checking
    print("\n" + "=" * 70)
    print("Test 2: Action Checking")
    print("=" * 70)
    
    test_actions = [
        {'skill': 'click', 'params': {'selector': '#search-btn'}},
        {'skill': 'type', 'params': {'selector': '#password', 'text': 'secret'}},
        {'skill': 'click', 'params': {'selector': '#pay-now'}},
        {'skill': 'screenshot', 'params': {}},
        {'skill': 'type', 'params': {'selector': '#search', 'text': 'áo khoác'}}
    ]
    
    for action in test_actions:
        allowed = guardrails.check_action_allowed(action)
        status = "✅ Allowed" if allowed else "🚫 Blocked"
        print(f"{status}: {action['skill']}({action['params']})")
    
    # Test 3: Payment amount
    print("\n" + "=" * 70)
    print("Test 3: Payment Amount Checking")
    print("=" * 70)
    
    test_amounts = [500_000, 1_000_000, 2_000_000, 5_000_000]
    
    for amount in test_amounts:
        allowed = guardrails.check_payment_amount(amount)
        status = "✅ OK" if allowed else "⚠️  Requires confirmation"
        print(f"{status}: {amount:,.0f} VND")
    
    # Test 4: Input validation
    print("\n" + "=" * 70)
    print("Test 4: Input Text Validation")
    print("=" * 70)
    
    test_inputs = [
        "Tìm áo khoác",
        "Enter password: 123456",
        "Credit card: 1234-5678-9012-3456",
        "CMND: 001234567890"
    ]
    
    for text in test_inputs:
        safe = guardrails.validate_input_text(text)
        status = "✅ Safe" if safe else "⚠️  Contains sensitive info"
        print(f"{status}: {text[:50]}...")
    
    # Test 5: Whitelist
    print("\n" + "=" * 70)
    print("Test 5: Whitelist Check")
    print("=" * 70)
    
    print(f"Allowed domains: {guardrails.get_allowed_domains()}")
    
    test_domains = [
        "https://shopee.vn/product",
        "https://unknown-site.com",
        "https://lazada.vn/cart"
    ]
    
    for url in test_domains:
        whitelisted = guardrails.is_domain_whitelisted(url)
        status = "✅ Whitelisted" if whitelisted else "⚠️  Not whitelisted"
        print(f"{status}: {url}")
    
    # Test 6: Alternative suggestions
    print("\n" + "=" * 70)
    print("Test 6: Alternative Suggestions")
    print("=" * 70)
    
    risky_action = {'skill': 'type', 'params': {'selector': '#password', 'text': 'secret'}}
    alternative = guardrails.suggest_alternative(risky_action)
    
    if alternative:
        print(f"Blocked: {risky_action}")
        print(f"Alternative: {alternative['skill']}({alternative['params']})")
        print(f"Reason: {alternative['reason']}")
    
    print("\n" + "=" * 70)
    print("✅ All Tests Completed!")
    print("=" * 70)
