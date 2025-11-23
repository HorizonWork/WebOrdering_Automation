"""
Error Analyzer - Analyze execution failures and suggest fixes
Identifies patterns in failed trajectories
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import re

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorAnalyzer:
    """
    Analyzes execution errors and suggests improvements.
    
    **Capabilities**:
        - Detect error patterns
        - Classify error types
        - Suggest fixes
        - Track error frequency
        - Generate error reports
    
    **Error Types**:
        - Selector not found (DOM issues)
        - Timeout (page loading)
        - Navigation failure (404, blocked)
        - Action failed (element not clickable)
        - Unexpected page state
    """
    
    def __init__(self):
        """Initialize error analyzer"""
        self.error_history: List[Dict] = []
        self.error_patterns = self._init_patterns()
        
        logger.info("ErrorAnalyzer initialized")
    
    def _init_patterns(self) -> Dict:
        """Initialize error detection patterns"""
        return {
            'selector_not_found': {
                'keywords': ['not found', 'không tìm thấy', 'selector', 'element'],
                'category': 'dom_issue',
                'severity': 'high',
                'fix_suggestions': [
                    'Kiểm tra lại selector CSS',
                    'Đợi element load (wait_for)',
                    'Thử selector thay thế',
                    'Kiểm tra dynamic content'
                ]
            },
            'timeout': {
                'keywords': ['timeout', 'hết thời gian', 'exceeded', 'timed out'],
                'category': 'timing_issue',
                'severity': 'medium',
                'fix_suggestions': [
                    'Tăng timeout duration',
                    'Kiểm tra kết nối mạng',
                    'Đợi networkidle',
                    'Reload trang nếu cần'
                ]
            },
            'navigation_failed': {
                'keywords': ['navigation', 'failed to navigate', 'net::ERR', '404', '500'],
                'category': 'network_issue',
                'severity': 'high',
                'fix_suggestions': [
                    'Kiểm tra URL chính xác',
                    'Kiểm tra kết nối internet',
                    'Thử lại sau vài giây',
                    'Kiểm tra website có hoạt động'
                ]
            },
            'element_not_clickable': {
                'keywords': ['not clickable', 'không thể click', 'obscured', 'covered'],
                'category': 'interaction_issue',
                'severity': 'medium',
                'fix_suggestions': [
                    'Scroll element vào view',
                    'Đợi element clickable',
                    'Đóng overlay/popup',
                    'Thử hover trước khi click'
                ]
            },
            'page_not_loaded': {
                'keywords': ['page not loaded', 'trang chưa tải', 'networkidle'],
                'category': 'timing_issue',
                'severity': 'medium',
                'fix_suggestions': [
                    'Tăng thời gian wait',
                    'Kiểm tra load event',
                    'Đợi specific element',
                    'Kiểm tra redirect'
                ]
            },
            'authentication_required': {
                'keywords': ['authentication', 'login required', 'unauthorized', 'cần đăng nhập'],
                'category': 'auth_issue',
                'severity': 'high',
                'fix_suggestions': [
                    'Đăng nhập trước khi thao tác',
                    'Kiểm tra session còn hạn',
                    'Kiểm tra cookies',
                    'Sử dụng LoginAgent'
                ]
            }
        }
    
    def analyze_error(
        self,
        error_message: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze single error.
        
        Args:
            error_message: Error message
            context: Additional context (action, observation)
            
        Returns:
            Analysis dict with category, severity, suggestions
        """
        error_lower = error_message.lower()
        
        # Detect error type
        detected_type = None
        for error_type, pattern in self.error_patterns.items():
            if any(kw in error_lower for kw in pattern['keywords']):
                detected_type = error_type
                break
        
        if not detected_type:
            detected_type = 'unknown'
            pattern = {
                'category': 'unknown',
                'severity': 'low',
                'fix_suggestions': ['Kiểm tra log chi tiết', 'Thử lại action']
            }
        else:
            pattern = self.error_patterns[detected_type]
        
        analysis = {
            'error_type': detected_type,
            'category': pattern['category'],
            'severity': pattern['severity'],
            'message': error_message,
            'suggestions': pattern['fix_suggestions'],
            'context': context or {}
        }
        
        # Add to history
        self.error_history.append(analysis)
        
        logger.info(f"🔍 Error analyzed: {detected_type} ({pattern['severity']} severity)")
        
        return analysis
    
    def analyze_trajectory(
        self,
        trajectory: Dict
    ) -> Dict:
        """
        Analyze failed trajectory.
        
        Args:
            trajectory: Failed trajectory with steps
            
        Returns:
            Analysis with error patterns and suggestions
        """
        errors = []
        
        for step in trajectory.get('steps', []):
            result = step.get('result', {})
            
            if result.get('status') == 'error':
                error_msg = result.get('message', '')
                
                analysis = self.analyze_error(
                    error_message=error_msg,
                    context={
                        'step_num': step.get('step_num'),
                        'action': step.get('action'),
                        'observation': step.get('observation', {}).get('url', 'N/A')
                    }
                )
                
                errors.append(analysis)
        
        # Find most common error category
        if errors:
            categories = [e['category'] for e in errors]
            most_common = Counter(categories).most_common(1)[0][0]
        else:
            most_common = 'none'
        
        return {
            'query': trajectory.get('query'),
            'total_errors': len(errors),
            'errors': errors,
            'most_common_category': most_common,
            'recovery_suggestions': self._generate_recovery_plan(errors)
        }
    
    def _generate_recovery_plan(self, errors: List[Dict]) -> List[str]:
        """Generate recovery plan from errors"""
        if not errors:
            return []
        
        # Get unique suggestions
        all_suggestions = []
        for error in errors:
            all_suggestions.extend(error['suggestions'])
        
        # Remove duplicates, keep order
        unique_suggestions = []
        seen = set()
        for s in all_suggestions:
            if s not in seen:
                unique_suggestions.append(s)
                seen.add(s)
        
        return unique_suggestions[:5]  # Top 5
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        if not self.error_history:
            return {
                'total_errors': 0,
                'by_category': {},
                'by_severity': {},
                'by_type': {}
            }
        
        categories = [e['category'] for e in self.error_history]
        severities = [e['severity'] for e in self.error_history]
        types = [e['error_type'] for e in self.error_history]
        
        return {
            'total_errors': len(self.error_history),
            'by_category': dict(Counter(categories)),
            'by_severity': dict(Counter(severities)),
            'by_type': dict(Counter(types))
        }
    
    def generate_report(self) -> str:
        """Generate error analysis report"""
        stats = self.get_error_statistics()
        
        report = f"""
{'='*70}
Error Analysis Report
{'='*70}

📊 Summary:
  Total errors: {stats['total_errors']}

📋 By Category:
"""
        for cat, count in stats['by_category'].items():
            report += f"  - {cat}: {count}\n"
        
        report += "\n⚠️  By Severity:\n"
        for sev, count in stats['by_severity'].items():
            report += f"  - {sev}: {count}\n"
        
        report += "\n🔍 By Type:\n"
        for typ, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"  - {typ}: {count}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def clear_history(self):
        """Clear error history"""
        self.error_history = []
        logger.info("Error history cleared")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("ErrorAnalyzer - Test")
    print("=" * 70 + "\n")
    
    analyzer = ErrorAnalyzer()
    
    # Test 1: Analyze single error
    print("Test 1: Analyze Single Error")
    print("-" * 40)
    
    error_msg = "Timeout: Element not found after 5000ms"
    analysis = analyzer.analyze_error(error_msg)
    
    print(f"Error type: {analysis['error_type']}")
    print(f"Category: {analysis['category']}")
    print(f"Severity: {analysis['severity']}")
    print(f"Suggestions: {len(analysis['suggestions'])}")
    
    # Test 2: Analyze trajectory
    print("\n\nTest 2: Analyze Failed Trajectory")
    print("-" * 40)
    
    trajectory = {
        'query': 'Tìm áo khoác',
        'steps': [
            {
                'step_num': 1,
                'action': {'skill': 'goto', 'params': {'url': 'shopee.vn'}},
                'result': {'status': 'success'}
            },
            {
                'step_num': 2,
                'action': {'skill': 'click', 'params': {'selector': '#search'}},
                'result': {'status': 'error', 'message': 'Element #search not found'}
            },
            {
                'step_num': 3,
                'action': {'skill': 'type', 'params': {'text': 'áo khoác'}},
                'result': {'status': 'error', 'message': 'Timeout waiting for element'}
            }
        ]
    }
    
    traj_analysis = analyzer.analyze_trajectory(trajectory)
    print(f"Total errors: {traj_analysis['total_errors']}")
    print(f"Most common: {traj_analysis['most_common_category']}")
    print(f"Recovery suggestions:")
    for i, sug in enumerate(traj_analysis['recovery_suggestions'], 1):
        print(f"  {i}. {sug}")
    
    # Test 3: Statistics
    print("\n\nTest 3: Error Statistics")
    print("-" * 40)
    
    # Add more errors
    analyzer.analyze_error("Navigation failed: net::ERR_CONNECTION_REFUSED")
    analyzer.analyze_error("Element not clickable at point (100, 200)")
    
    stats = analyzer.get_error_statistics()
    print(f"Total errors analyzed: {stats['total_errors']}")
    print(f"Categories: {stats['by_category']}")
    
    # Test 4: Generate report
    print("\n\nTest 4: Generate Report")
    print("-" * 40)
    report = analyzer.generate_report()
    print(report)
    
    print("\nyes All Tests Completed!")
