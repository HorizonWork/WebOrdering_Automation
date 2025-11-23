
"""
JavaScript Event Recorder for Human Teleoperation.

This module provides the JS code to be injected into the browser to capture
user interactions (clicks, inputs) and store them for retrieval by the python agent.
"""

JS_EVENT_RECORDER = """
(function() {
    if (window._recorderInstalled) return;
    window._recorderInstalled = true;
    window._recordedEvents = [];

    function getSelector(el) {
        if (!el) return "";
        
        // 1. ID
        if (el.id) return "#" + el.id;
        
        // 2. Name (for inputs)
        if (el.name) return el.tagName.toLowerCase() + "[name='" + el.name + "']";
        
        // 3. Class (simplified)
        if (el.className && typeof el.className === 'string') {
            const classes = el.className.split(/\s+/).filter(c => c).slice(0, 2);
            if (classes.length > 0) {
                return el.tagName.toLowerCase() + "." + classes.join(".");
            }
        }
        
        // 4. Fallback: nth-of-type
        // This is expensive, so we do a simple version
        let parent = el.parentElement;
        if (parent) {
            let siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
            let index = siblings.indexOf(el) + 1;
            return el.tagName.toLowerCase() + ":nth-of-type(" + index + ")";
        }
        
        return el.tagName.toLowerCase();
    }

    function recordEvent(type, target, extra = {}) {
        const selector = getSelector(target);
        const event = {
            type: type,
            selector: selector,
            timestamp: Date.now(),
            ...extra
        };
        // Keep only last 50 events to avoid memory leak
        if (window._recordedEvents.length > 50) {
            window._recordedEvents.shift();
        }
        window._recordedEvents.push(event);
        console.log("[Recorder] Captured:", event);
    }

    // Capture Clicks
    document.addEventListener('click', function(e) {
        // Ignore clicks on recorder UI if we had one (we don't yet)
        recordEvent('click', e.target);
    }, true);

    // Capture Input Changes (change event fires on blur/enter)
    document.addEventListener('change', function(e) {
        const tag = e.target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') {
            recordEvent('change', e.target, {
                value: e.target.value,
                tagName: tag
            });
        }
    }, true);

    // Capture Scroll (Debounced)
    let scrollTimeout;
    document.addEventListener('scroll', function(e) {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(function() {
            const x = window.scrollX;
            const y = window.scrollY;
            // We record window scroll. Element scroll is harder to track generically.
            recordEvent('scroll', document.body, {
                scrollX: x,
                scrollY: y
            });
        }, 500); // 500ms debounce
    }, true);
    
    // Expose to Python
    window.getRecordedEvents = function() {
        const events = [...window._recordedEvents];
        window._recordedEvents = []; // Clear after reading
        return events;
    };
    
    console.log("Human Teleop Recorder Installed");
})();
"""
