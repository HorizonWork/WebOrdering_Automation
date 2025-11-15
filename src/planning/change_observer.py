"""
Change Observer - DOM Mutation Detection
Tracks DOM changes after actions using MutationObserver pattern
Based on Agent-E's change observation mechanism
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from playwright.async_api import Page
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DOMChange:
    """Single DOM change event"""
    change_type: str  # 'childList', 'attributes', 'characterData'
    target: str
    added_nodes: int
    removed_nodes: int
    attribute_name: Optional[str]
    timestamp: str


class ChangeObserver:
    """
    Observes and tracks DOM changes using MutationObserver API.
    
    **Purpose**:
        - Detect if action had effect
        - Provide feedback to planner
        - Enable error detection
        - Support backtracking on failures
    
    **How it works**:
        1. Inject MutationObserver JavaScript into page
        2. Listen for DOM mutations (add/remove nodes, attribute changes)
        3. Collect changes after action execution
        4. Analyze changes to determine action success
    
    **Change Types**:
        - childList: Nodes added/removed
        - attributes: Element attributes changed
        - characterData: Text content changed
    
    **Based on**: Agent-E paper (Section 3.3 - Change Observation)
    """
    
    def __init__(self):
        """Initialize change observer"""
        self.changes: List[DOMChange] = []
        self.observer_injected = False
        
        logger.info("ChangeObserver initialized")
    
    async def inject_observer(self, page: Page):
        """
        Inject MutationObserver into page.
        
        Args:
            page: Playwright page
        """
        if self.observer_injected:
            return
        
        logger.debug("Injecting MutationObserver...")
        
        # JavaScript to inject
        observer_script = """
        // Initialize change log
        window.__woaChangeLog = [];
        
        // Create MutationObserver
        window.__woaObserver = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                window.__woaChangeLog.push({
                    type: mutation.type,
                    target: mutation.target.tagName || 'unknown',
                    addedNodes: mutation.addedNodes.length,
                    removedNodes: mutation.removedNodes.length,
                    attributeName: mutation.attributeName,
                    timestamp: Date.now()
                });
            });
        });
        
        // Start observing
        window.__woaObserver.observe(document.body, {
            childList: true,      // Watch for node additions/removals
            subtree: true,        // Watch entire tree
            attributes: true,     // Watch attribute changes
            attributeOldValue: true,
            characterData: true,  // Watch text changes
            characterDataOldValue: true
        });
        
        // Function to retrieve changes
        window.__woaGetChanges = () => {
            const changes = window.__woaChangeLog;
            window.__woaChangeLog = [];  // Clear log
            return changes;
        };
        
        console.log('WOA MutationObserver injected');
        """
        
        try:
            await page.evaluate(observer_script)
            self.observer_injected = True
            logger.info("✓ MutationObserver injected successfully")
        except Exception as e:
            logger.error(f"Failed to inject observer: {e}")
    
    async def start_observing(self, page: Page):
        """
        Start observing changes.
        
        Args:
            page: Playwright page
        """
        # Inject if not already done
        await self.inject_observer(page)
        
        # Clear previous changes
        self.changes = []
        
        # Clear change log in browser
        try:
            await page.evaluate("window.__woaChangeLog = [];")
            logger.debug("Started observing changes")
        except Exception as e:
            logger.warning(f"Failed to clear change log: {e}")
    
    async def get_changes(self, page: Page, wait_ms: int = 1000) -> List[DOMChange]:
        """
        Get changes that occurred since start_observing().
        
        Args:
            page: Playwright page
            wait_ms: Wait time before collecting changes (ms)
            
        Returns:
            List of DOM changes
        """
        # Wait for changes to settle
        await asyncio.sleep(wait_ms / 1000)
        
        try:
            # Get changes from browser
            raw_changes = await page.evaluate("window.__woaGetChanges()")
            
            # Parse changes
            changes = []
            for change in raw_changes:
                dom_change = DOMChange(
                    change_type=change.get('type', 'unknown'),
                    target=change.get('target', 'unknown'),
                    added_nodes=change.get('addedNodes', 0),
                    removed_nodes=change.get('removedNodes', 0),
                    attribute_name=change.get('attributeName'),
                    timestamp=datetime.fromtimestamp(
                        change.get('timestamp', 0) / 1000
                    ).isoformat()
                )
                changes.append(dom_change)
            
            self.changes = changes
            logger.debug(f"Collected {len(changes)} DOM changes")
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to get changes: {e}")
            return []
    
    def analyze_changes(self, changes: Optional[List[DOMChange]] = None) -> Dict:
        """
        Analyze changes to determine impact.
        
        Args:
            changes: List of changes (uses self.changes if None)
            
        Returns:
            Analysis dict with status, message, metrics
        """
        if changes is None:
            changes = self.changes
        
        if not changes:
            return {
                'status': 'no_change',
                'confidence': 'high',
                'message': 'No DOM changes detected',
                'metrics': {
                    'total_changes': 0,
                    'nodes_added': 0,
                    'nodes_removed': 0,
                    'attributes_changed': 0
                }
            }
        
        # Calculate metrics
        total = len(changes)
        added = sum(c.added_nodes for c in changes)
        removed = sum(c.removed_nodes for c in changes)
        attrs = sum(1 for c in changes if c.change_type == 'attributes')
        
        # Classify impact
        if added > 20 or removed > 20:
            status = 'major_change'
            confidence = 'high'
            message = f'Major page change: +{added} nodes, -{removed} nodes'
        elif added > 5 or removed > 5:
            status = 'moderate_change'
            confidence = 'medium'
            message = f'Moderate change: +{added} nodes, -{removed} nodes'
        elif attrs > 10:
            status = 'ui_update'
            confidence = 'medium'
            message = f'UI updated: {attrs} attribute changes'
        elif total > 0:
            status = 'minor_change'
            confidence = 'low'
            message = f'Minor change: {total} mutations'
        else:
            status = 'no_change'
            confidence = 'high'
            message = 'No significant changes'
        
        return {
            'status': status,
            'confidence': confidence,
            'message': message,
            'metrics': {
                'total_changes': total,
                'nodes_added': added,
                'nodes_removed': removed,
                'attributes_changed': attrs
            }
        }
    
    def get_change_summary(self) -> str:
        """Get human-readable summary of changes"""
        analysis = self.analyze_changes()
        
        summary = f"""
DOM Change Summary:
  Status: {analysis['status']}
  Confidence: {analysis['confidence']}
  Message: {analysis['message']}
  
  Metrics:
    - Total changes: {analysis['metrics']['total_changes']}
    - Nodes added: {analysis['metrics']['nodes_added']}
    - Nodes removed: {analysis['metrics']['nodes_removed']}
    - Attributes changed: {analysis['metrics']['attributes_changed']}
"""
        return summary.strip()
    
    def did_action_succeed(self, expected_change: str = 'any') -> bool:
        """
        Determine if action succeeded based on changes.
        
        Args:
            expected_change: Expected change type ('any', 'major', 'moderate', 'minor')
            
        Returns:
            True if action likely succeeded
        """
        analysis = self.analyze_changes()
        status = analysis['status']
        
        if expected_change == 'any':
            return status != 'no_change'
        elif expected_change == 'major':
            return status == 'major_change'
        elif expected_change == 'moderate':
            return status in ['major_change', 'moderate_change']
        elif expected_change == 'minor':
            return status != 'no_change'
        
        return False
    
    def reset(self):
        """Reset observer state"""
        self.changes = []
        logger.debug("ChangeObserver reset")


# Test & Example
async def test_change_observer():
    """Test change observer with Playwright"""
    from src.execution.browser_manager import BrowserManager
    
    print("=" * 70)
    print("ChangeObserver - Test")
    print("=" * 70 + "\n")
    
    # Initialize
    observer = ChangeObserver()
    manager = BrowserManager(headless=False)
    
    try:
        # Create page
        page = await manager.new_page()
        print("✓ Page created\n")
        
        # Navigate to test page
        await page.goto("https://example.com")
        await asyncio.sleep(2)
        print("✓ Navigated to example.com\n")
        
        # Inject observer
        await observer.inject_observer(page)
        print("✓ Observer injected\n")
        
        # Start observing
        await observer.start_observing(page)
        print("✓ Started observing\n")
        
        # Simulate action: click a link
        print("🔄 Performing action: clicking link...")
        try:
            await page.click("a", timeout=5000)
        except:
            print("⚠️  No clickable link found, that's okay for demo\n")
        
        # Get changes
        print("📊 Collecting changes...")
        changes = await observer.get_changes(page, wait_ms=1500)
        print(f"✓ Collected {len(changes)} changes\n")
        
        # Analyze changes
        print("=" * 70)
        print("Analysis")
        print("=" * 70)
        print(observer.get_change_summary())
        
        # Check success
        success = observer.did_action_succeed('any')
        print(f"\n✅ Action succeeded: {success}")
        
    finally:
        await manager.close()
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_change_observer())
