# WOA Agent - Theoretical Foundations & Architecture

## Table of Contents
1. [Research Foundations](#research-foundations)
2. [Core Architecture](#core-architecture)
3. [Design Principles](#design-principles)
4. [Technical Deep-Dive](#technical-deep-dive)
5. [Model Selection](#model-selection)
6. [References](#references)

---

## Research Foundations

### Web Automation Landscape (2024-2025)

The WOA Agent synthesizes best practices from leading research in LLM-based web automation:

#### **WebVoyager (2024)**

**Key Contributions:**
- Multimodal perception combining screenshots with DOM/HTML
- Multi-encoder architecture for robust scene understanding
- ReAct reasoning pattern: Thought → Observation → Action

**How WOA Uses It:**
```
WebVoyager approach:
  Screenshot (vision) + DOM (text) → Shared embedding space
  
WOA implementation:
  PhoBERT encodes UI text labels + action descriptions (768-dim)
  Vision handled by OmniParser UI detection
  Combined in scene_representation.py for semantic matching
```

**References:**
- Multimodal encoding of annotated screenshots
- Sequential context maintenance through trajectory history
- ReAct loop for interpretable reasoning

---

#### **Agent-E (2024) - From Autonomous Web Navigation**

**Key Contributions:**
- Two-tier hierarchical architecture: Planner + Navigator
- Flexible DOM distillation with 3 modes
- Change observation via MutationObserver
- 73.2% success on WebVoyager (16% improvement over baselines)

**How WOA Uses It:**
```
Agent-E hierarchy:
  Planner Agent → high-level task decomposition
  Browser Navigation Agent → skill execution
  
WOA implementation:
  PlannerAgent (ViT5) → generates action sequences
  NavigatorAgent → manages Playwright skills
  Sub-agents → specialized handlers (login, payment)
```

**DOM Distillation Modes:**

| Mode | Use Case | Content |
|------|----------|---------|
| `full` | Complex pages | Complete DOM tree |
| `text_only` | General tasks | Text + structure only |
| `input_fields` | Forms, filtering | Interactive elements only |

**Change Observation Pattern:**
```javascript
// MutationObserver tracks:
new MutationObserver((mutations) => {
  mutations.forEach(m => {
    - Type: childList, attributes, characterData
    - Target: affected DOM element
    - Timestamp: when change occurred
  })
})
```

**References:**
- Agent-E: From Autonomous Web Navigation to Multiagent Collaboration (2024)
- Hierarchical agent design patterns
- 8 design principles from Agent-E paper

---

#### **AgentOccam (2025) - Simplicity in Web Agents**

**Key Contributions:**
- Observation/action space alignment critical as architecture
- 160%+ performance improvement by optimizing space sizing
- DOM element filtering removes irrelevant nodes
- Action set reduction + planning action addition

**How WOA Uses It:**
```
AgentOccam principle:
  LLM capacity ≠ large state space
  
WOA implementation:
  dom_distiller.py filters DOM to relevant elements
  Only keeps interactive elements by default
  Observation pruning: max 1000 chars, top 20 elements
  Action space: {goto, click, type, select, scroll, wait, complete}
  Added: "plan" action for reasoning steps
```

**References:**
- Simplicity bias: smaller spaces = better LLM utilization
- Information density over completeness
- Empirical validation on WebArena/WebVoyager

---

#### **Invisible Multi-Agent System**

**Key Contributions:**
- Human-inspired agent collaboration
- Dynamic hierarchy with specialist sub-agents
- Adaptive scene representation per task
- RAIL: Retrieval-Augmented Imitation Learning

**How WOA Uses It:**
```
Invisible approach:
  Main agent delegates to specialists
  Adaptive weighting of info sources
  
WOA implementation:
  AgentOrchestrator routes to:
    - LoginAgent (authentication)
    - PaymentAgent (checkout)
    - SearchAgent (finding products)
  
  Adaptive scene (src/perception/scene_representation.py):
    When searching: emphasize product listings
    When filling form: emphasize input fields
    When authenticating: emphasize login elements

  RAIL memory (src/learning/memory/rail.py):
    Store: (state_embedding, action, result)
    Retrieve: Similar states → apply learned actions
```

**References:**
- Human-inspired agent design
- Adaptive representation based on task context
- Memory-augmented learning

---

#### **OpenAI Operator (2024)**

**Key Contributions:**
- Safety-first design: human takeover, confirmation flow
- Self-correction capability
- GPT-4V for multimodal understanding

**How WOA Uses It:**
```
Operator safety patterns:
  Sensitive action confirmation
  Error recovery with backtracking
  
WOA implementation:
  safety_guardrails.py:
    - Deny access: banking, healthcare sites
    - Confirm before: payments, account changes
    - Timeout recovery: retry or backtrack
  
  navigator_agent.py:
    - Self-correcting ReAct loop
    - Backtracking on action failure
    - Max retry: 3 attempts
```

**References:**
- Guardrail systems for agent safety
- Human-in-the-loop design patterns
- Takeover mechanisms for user control

---

## Core Architecture

### 4-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│              USER QUERY (Vietnamese)                │
│         "Mua áo khoác giá dưới 500k"               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ LAYER 1: PERCEPTION (src/perception/)          │
│ ✓ Screenshot capture with bounding boxes       │
│ ✓ DOM extraction + flexible distillation       │
│ ✓ UI element detection (OmniParser)            │
│ ✓ PhoBERT embeddings (768-dim vectors)         │
│ ✓ Adaptive scene representation                │
│                                                 │
│ Outputs: {screenshot, dom, embeddings, elements}
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ LAYER 2: PLANNING (src/planning/)              │
│ ✓ Query embedding with PhoBERT                 │
│ ✓ ViT5 generates thought (ReAct reasoning)     │
│ ✓ ViT5 generates action (skill + params)       │
│ ✓ Change observation for feedback              │
│ ✓ Hierarchical agent routing                   │
│                                                 │
│ Output: {thought, action, confidence}          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ LAYER 3: EXECUTION (src/execution/)            │
│ ✓ Playwright browser control                   │
│ ✓ Skill primitives: click, type, scroll, etc.  │
│ ✓ Element finding with CSS/XPath               │
│ ✓ Timeout & error handling                     │
│ ✓ Observable state changes                     │
│                                                 │
│ Output: {status, changes, observation}         │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ LAYER 4: LEARNING (src/learning/)              │
│ ✓ Trajectory storage as vectors                │
│ ✓ RAIL memory: retrieve similar examples       │
│ ✓ LoRA fine-tuning on domain data              │
│ ✓ Error analysis & guardrail updates           │
│                                                 │
│ Output: Improved model for next iteration       │
└─────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### **Layer 1: Perception**

```python
# Current state extraction
def perceive(page) -> Dict:
    screenshot = page.screenshot()  # PNG bytes
    dom = page.content()             # HTML string
    embeddings = encoder.encode(dom) # 768-dim vectors
    elements = parser.detect()       # {id, tag, text, bbox}
    
    return {
        'visual': screenshot,
        'textual': dom,
        'semantic': embeddings,
        'interactive': elements
    }
```

**Key Technologies:**
- **Screenshot**: Playwright's browser automation
- **DOM**: HTML parsing with BeautifulSoup
- **Embeddings**: PhoBERT (vinai/phobert-base-v2)
- **UI Detection**: OmniParser or Detic/YOLO

**Output Format:**
```json
{
  "screenshot_bytes": "...",
  "dom_distilled": "...",
  "elements": [
    {"id": 0, "tag": "input", "text": "Tìm kiếm", "bbox": [10, 20, 100, 40]}
  ],
  "embeddings": [[0.1, 0.2, ..., 0.768]],
  "url": "https://shopee.vn"
}
```

---

#### **Layer 2: Planning**

```python
# Decision making with ReAct pattern
def plan(query, observation) -> Tuple[str, Dict]:
    # Step 1: Generate Thought (reasoning)
    thought = vit5.generate_thought(
        query=query,
        observation=observation,
        history=history
    )
    # Output: "Cần tìm kiếm sản phẩm trước"
    
    # Step 2: Generate Action
    action = vit5.generate_action(
        thought=thought,
        available_skills=skills,
        dom_state=observation['dom_distilled']
    )
    # Output: {"skill": "type", "params": {"selector": "search_box", "text": "áo khoác"}}
    
    return thought, action
```

**ReAct Loop:**
```
Thought: "Người dùng muốn tìm áo khoác màu đen"
Action: goto("https://shopee.vn")
Observation: Trang chủ Shopee đã tải
↓
Thought: "Giờ cần tìm kiếm sản phẩm"
Action: click(selector="#search-box")
Observation: Search box được focus
↓
Thought: "Gõ từ khóa tìm kiếm"
Action: type(selector="#search-box", text="áo khoác đen")
Observation: Text đã được nhập
↓
... (continues until task complete)
```

**Available Skills:**
- `goto(url)` - Navigate to URL
- `click(selector)` - Click element
- `type(selector, text)` - Type text
- `select(selector, value)` - Select dropdown
- `scroll(x, y)` - Scroll page
- `wait_for(selector)` - Wait for element
- `complete()` - Mark task done

---

#### **Layer 3: Execution**

```python
# Action execution through Playwright
async def execute(page, action):
    skill_name = action['skill']
    params = action['params']
    
    # Route to appropriate skill
    if skill_name == 'click':
        await page.click(params['selector'])
    elif skill_name == 'type':
        await page.fill(params['selector'], params['text'])
    elif skill_name == 'goto':
        await page.goto(params['url'])
    
    # Observe changes
    await asyncio.sleep(1)  # Wait for changes
    changes = await change_observer.get_changes()
    
    return {
        'status': 'success',
        'changes': changes,
        'new_observation': perceive(page)
    }
```

**Error Handling:**
```python
# Retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        result = await execute(page, action)
        return result
    except TimeoutError:
        if attempt < max_retries - 1:
            await page.reload()
        else:
            raise
    except SelectorError:
        # Try alternative selectors
        return backtrack()
```

---

#### **Layer 4: Learning**

```python
# Store successful trajectories
def learn(trajectory):
    # trajectory = [{step, thought, action, obs, result}]
    
    # Embed the trajectory
    state_embedding = encoder.encode(
        str(trajectory[-1]['observation'])
    )
    action_embedding = encoder.encode(
        str(trajectory[-1]['action'])
    )
    
    # Store in vector DB
    vector_store.store({
        'state': state_embedding,
        'action': action_embedding,
        'trajectory': trajectory,
        'success': True
    })
    
    # Optional: Fine-tune ViT5 on this trajectory
    if trajectory_quality > threshold:
        fine_tune_vit5_with_lora(
            training_data=[{
                'input': trajectory[-1]['observation'],
                'output': trajectory[-1]['action']
            }]
        )
```

**RAIL Memory:**
```python
def retrieve_similar(current_state):
    # Get current state embedding
    query_embedding = encoder.encode(current_state)
    
    # Find similar trajectories
    similar = vector_store.search(
        query_embedding,
        top_k=5
    )
    
    # Apply knowledge
    suggested_action = similar[0]['action']
    return suggested_action
```

---

## Design Principles

### Principle 1: Separation of Concerns

Each layer handles one responsibility:
- **Perception**: Observe world state
- **Planning**: Decide what to do
- **Execution**: Make it happen
- **Learning**: Improve from experience

**Benefit**: Easy to test, debug, and improve each component independently.

---

### Principle 2: Vietnamese Language-First Design

**PhoBERT for Encoding**
```python
# CORRECT: Use for embeddings
query = "Mua áo khoác"
embedding = phobert.encode(query)  # 768-dim vector
similarity = cosine_sim(embedding, ui_embedding)

# WRONG: Do NOT use for generation
# output = phobert.generate(query)  # ❌ Not supported
```

**ViT5 for Generation**
```python
# CORRECT: Use for generation
input_text = "Tìm kiếm sản phẩm áo khoác"
output = vit5.generate(input_text)  # "click(#search_box) type(#search, 'áo khoác')"

# NOT for embedding (use PhoBERT instead)
# embedding = vit5.encode(input_text)  # ❌ Wrong tool
```

---

### Principle 3: Adaptive Scene Representation

Different tasks need different levels of detail:

```python
# When searching: Focus on product list
mode = 'input_fields'  # Only interactive elements

# When reading product details: Need all text
mode = 'text_only'  # All text content

# When handling complex UI: Full detail
mode = 'full'  # Complete DOM
```

---

### Principle 4: Continuous Observation

After every action, observe what changed:

```python
# Without observation
action: click(button)
# Did button click? Did page navigate? Did popup appear?
# Unknown! 

# With change observation
action: click(button)
observation: {
    'added_nodes': 15,
    'removed_nodes': 0,
    'attribute_changes': 3,
    'status': 'major_change'
}
# Yes, button click triggered major page change
```

---

### Principle 5: Hierarchical Agent Structure

Complex tasks → Decompose → Delegate to specialists

```
MainAgent
├── LoginAgent (if login needed)
├── SearchAgent (for finding products)
├── FilterAgent (for applying filters)
├── CartAgent (for adding to cart)
└── PaymentAgent (for checkout)
```

**Example:**
```python
if task.requires_login:
    result = await login_agent.handle_login(page)
    
if task.requires_search:
    result = await search_agent.search(page, query)
    
if task.requires_payment:
    result = await payment_agent.checkout(page)
```

---

### Principle 6: Safety & Human Control

```python
# Sensitive operations require confirmation
if action.is_sensitive():  # e.g., payment > 1M VND
    user_confirmed = await get_user_confirmation(action)
    if not user_confirmed:
        return backtrack()

# Restricted websites
if url in RESTRICTED_SITES:  # banking, healthcare
    raise PermissionError(f"Access denied: {url}")

# Timeout recovery
if execution_time > max_time:
    await page.reload()  # Reset state
    return backtrack()  # Try alternative approach
```

---

## Technical Deep-Dive

### PhoBERT Integration

**Model Details:**
- **Name**: vinai/phobert-base-v2
- **Parameters**: ~135M
- **Embedding Dimension**: 768
- **Language**: Vietnamese
- **Pre-training**: BPE tokenization, MLM + NSP tasks

**Usage Pattern:**
```python
from src.models.phobert_encoder import PhoBERTEncoder

encoder = PhoBERTEncoder(model_name="vinai/phobert-base-v2")

# Encode Vietnamese text
queries = ["Tìm áo khoác", "Giày thể thao"]
embeddings = encoder.encode_text(queries, normalize=True)
# Output: (2, 768) array

# Find similar UI elements
ui_texts = ["Áo khoác nam", "Áo khoác nữ", "Quần jean"]
similarities = encoder.compute_similarity(
    query="Tìm áo khoác màu đen",
    candidates=ui_texts
)
# Output: [0.87, 0.85, 0.32]
```

---

### ViT5 Fine-Tuning

**Model Details:**
- **Name**: VietAI/vit5-base
- **Parameters**: ~310M
- **Architecture**: Encoder-Decoder (T5-style)
- **Pre-training**: Text-to-Text format on Vietnamese corpus

**LoRA Configuration:**
```python
# Memory-efficient fine-tuning
lora_config = {
    'r': 8,              # LoRA rank
    'lora_alpha': 16,    # Alpha multiplier
    'lora_dropout': 0.1, # Dropout rate
    'target_modules': ['q', 'v']  # Attention modules
}

# Training hyperparameters
training_config = {
    'learning_rate': 5e-5,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'num_epochs': 3,
    'warmup_ratio': 0.1,
    'fp16': True  # Mixed precision
}

# Training data format
training_data = [
    {
        'input': 'Trạng thái: Có search box. Tác vụ: tìm áo khoác',
        'output': 'type(#search-box, "áo khoác")'
    },
    # ... more examples
]
```

**Training Command:**
```bash
python scripts/train_vit5.py \
    --model_name VietAI/vit5-base \
    --data_path data/shopee_trajectories.json \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --batch_size 4
```

**GPU Requirements:**
- Base model: ~1.2 GB
- LoRA fine-tuning: ~7-8 hours (single V100/A100)
- Inference: ~2 seconds per action

---

### DOM Distillation Algorithm

**Input**: Full HTML DOM
**Output**: Simplified HTML focusing on relevant elements

**Algorithm:**

```
1. Parse HTML with BeautifulSoup
2. Based on mode:
   
   IF mode == 'full':
       - Remove: <script>, <style>, comments
       - Keep: Everything else
       
   IF mode == 'text_only':
       - Keep: <p>, <h1-h6>, <span>, <div>, <li>, <td>
       - Remove: Non-text tags
       - Filter attributes to: id, name, class, type, value
       
   IF mode == 'input_fields':
       - Keep: <input>, <button>, <select>, <a>, <form>
       - Remove: Content containers
       - Filter to interactive elements only

3. Clean attributes (remove inline styles, event handlers)
4. Truncate text content (max 100 chars per element)
5. Return simplified HTML
```

**Example:**

```html
<!-- BEFORE (97 lines) -->
<html>
  <head>
    <script>alert('test')</script>
    <style>.header { color: red; }</style>
  </head>
  <body>
    <div class="header">
      <h1>Shopee</h1>
      <nav>...</nav>
    </div>
    <div class="search-container">
      <input id="search" placeholder="Tìm kiếm" />
      <button onclick="search()">Tìm</button>
    </div>
    ...
  </body>
</html>

<!-- AFTER text_only mode (25 lines) -->
<html>
  <body>
    <div class="header">
      <h1>Shopee</h1>
    </div>
    <div class="search-container">
      <input id="search" placeholder="Tìm kiếm" />
      <button>Tìm</button>
    </div>
  </body>
</html>
```

**Size Reduction:**
- Full DOM: 50-200 KB typical
- Text-only: 5-20 KB (75-90% reduction)
- Input-fields: 2-5 KB (95%+ reduction)

---

### Change Observer Pattern

**JavaScript Injection:**
```javascript
// Injected into browser page
window.__woaChangeLog = [];

window.__woaObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        window.__woaChangeLog.push({
            type: mutation.type,  // 'childList' | 'attributes' | 'characterData'
            target: mutation.target.tagName,
            addedNodes: mutation.addedNodes.length,
            removedNodes: mutation.removedNodes.length,
            attributeName: mutation.attributeName,
            timestamp: Date.now()
        });
    });
});

window.__woaObserver.observe(document.body, {
    childList: true,    // Track node additions/removals
    subtree: true,      // Watch entire tree
    attributes: true,   // Track attribute changes
    characterData: true // Track text changes
});

// Retrieve changes and clear log
window.__woaGetChanges = () => {
    const changes = window.__woaChangeLog;
    window.__woaChangeLog = [];
    return changes;
};
```

**Change Analysis:**

```python
def analyze_changes(mutations):
    """Classify action result based on DOM changes"""
    
    if len(mutations) == 0:
        return {'status': 'no_change', 'confidence': 'high'}
    
    added = sum(m['addedNodes'] for m in mutations)
    removed = sum(m['removedNodes'] for m in mutations)
    attrs = sum(1 for m in mutations if m['type'] == 'attributes')
    
    if added > 10 or removed > 10:
        return {
            'status': 'major_change',
            'message': f'Page loaded: +{added} nodes, -{removed} nodes',
            'confidence': 'high'
        }
    elif attrs > 5:
        return {
            'status': 'ui_update',
            'message': f'UI updated: {attrs} attribute changes',
            'confidence': 'medium'
        }
    else:
        return {
            'status': 'minor_change',
            'message': f'Small change: {len(mutations)} mutations',
            'confidence': 'low'
        }
```

---

## Model Selection

### Why PhoBERT (Not RoBERTa)?

| Aspect | PhoBERT | RoBERTa | DistilBERT |
|--------|---------|---------|------------|
| **Vietnamese** | ✅ Trained on Vi text | ❌ English only | ❌ English only |
| **Pre-training** | Vietnamese corpus 20GB | English corpus | English corpus |
| **Tokenization** | BPE (Vietnamese aware) | BPE (English) | WordPiece |
| **Size** | 135M params | 355M params | 66M params |
| **Speed** | Fast | Slower | Fastest |
| **Embedding Quality** | ~0.85+ similarity | N/A | ~0.78 similarity |

**Decision**: PhoBERT optimal for Vietnamese text embedding

---

### Why ViT5 (Not mBART)?

| Aspect | ViT5 | mBART | XLMR | mT5 |
|--------|------|-------|------|-----|
| **Generation** | ✅ Full | ✅ Full | ❌ Encoder | ✅ Full |
| **Vietnamese** | ✅ Pre-trained | ✅ Supports | ✅ Supports | ✅ Supports |
| **Fine-tune** | ✅ LoRA ready | ⚠️ Large | ✅ Fast | ✅ Fast |
| **Parameters** | 310M (base) | 406M | 355M | 580M (base) |
| **Domain Adapt** | ✅ Easy | ⚠️ Harder | ✅ Easy | ⚠️ Larger |
| **Speed** | Fast | Medium | Medium | Slow |

**Decision**: ViT5 optimal for Vietnamese action generation with LoRA fine-tuning

---

### Why Playwright (Not Selenium)?

| Feature | Playwright | Selenium |
|---------|------------|----------|
| **Modern APIs** | ✅ Async/await | ❌ Synchronous |
| **Speed** | ✅ 2-3x faster | ⚠️ Slower |
| **Multi-browser** | ✅ Chrome, FF, Safari | ✅ Chrome, FF, Edge |
| **Debugging** | ✅ Built-in | ❌ External tools |
| **Screenshot** | ✅ Full page | ✅ Full page |
| **Network Interception** | ✅ Yes | ❌ Limited |
| **Context Isolation** | ✅ Yes | ❌ No |
| **Memory Usage** | ✅ Efficient | ⚠️ Higher |

**Decision**: Playwright optimal for fast, reliable automation

---

## References

### Core Papers
1. **WebVoyager (2024)** - Multimodal web agent with ReAct
2. **Agent-E (2024)** - Hierarchical architecture, DOM distillation
3. **AgentOccam (2025)** - Space alignment for LLM agents
4. **Invisible Multi-Agent** - RAIL memory, adaptive scenes

### Vietnamese Language Models
5. PhoBERT: Pre-trained Language Models for Vietnamese (VinAI)
6. ViT5: Pretrained Text-to-Text Transformer for Vietnamese (VietAI)
7. Vietnamese Text Normalization Standards

### Web Automation
8. Playwright Documentation & API Reference
9. WebDriver Standards (W3C)
10. MutationObserver Browser API

### Machine Learning
11. LoRA: Low-Rank Adaptation of Large Language Models
12. FAISS: Efficient Similarity Search
13. Retrieval-Augmented Generation (RAG) Patterns

---

## Architecture Diagrams

### System Flow Diagram

```
User Interface
     ↓
Input: "Mua áo khoác giá dưới 500k"
     ↓
┌─────────────────────────────────────┐
│ PERCEPTION LAYER                    │
│ - Screenshot capture                │
│ - DOM extraction                    │
│ - PhoBERT embedding (768-dim)       │
└──────────────┬──────────────────────┘
               ↓
        Current State
        {visual, textual, semantic}
               ↓
┌─────────────────────────────────────┐
│ PLANNING LAYER                      │
│ - ViT5 generates Thought            │
│ - ViT5 generates Action             │
│ - ReAct loop                        │
│ - Change observation                │
└──────────────┬──────────────────────┘
               ↓
        Decision: Action
        {skill, params, thought}
               ↓
┌─────────────────────────────────────┐
│ EXECUTION LAYER                     │
│ - Playwright skill execution        │
│ - Browser state management          │
│ - Error handling                    │
└──────────────┬──────────────────────┘
               ↓
        Result: State Change
        {screenshot, dom, changes}
               ↓
┌─────────────────────────────────────┐
│ LEARNING LAYER                      │
│ - Store trajectory as vectors       │
│ - RAIL memory retrieval             │
│ - LoRA fine-tuning                  │
└──────────────┬──────────────────────┘
               ↓
        Improved Models for Next Run
```

---

## Conclusion

The WOA Agent architecture represents a synthesis of latest research in LLM-based web automation, optimized specifically for Vietnamese e-commerce platforms. By carefully selecting Vietnamese language models (PhoBERT + ViT5), implementing proven architectural patterns (hierarchical agents, ReAct reasoning), and following design principles (separation of concerns, adaptive representation), the system achieves high reliability and performance while remaining maintainable and extensible.

The 4-layer pipeline (Perception → Planning → Execution → Learning) provides clear separation of concerns, making each component independently testable and improvable. The integration of change observation, RAIL memory, and LoRA fine-tuning enables continuous learning from experience.

**Key Success Factors:**
1. Correct model selection for Vietnamese
2. Hierarchical agent architecture
3. Continuous observation and feedback
4. Safety-first design with human control
5. Learning from successful trajectories

---

*Last Updated: November 15, 2025*
*Architecture Version: 1.0*
