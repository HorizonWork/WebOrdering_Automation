# 📚 WOA Agent - Tổng Quan Codebase

## 🎯 Giới thiệu Dự án

**WOA Agent** (Web Ordering Automation Agent) là một hệ thống **AI Agent tự động** được thiết kế để tự động hóa các tác vụ trên web, đặc biệt là các nền tảng thương mại điện tử Việt Nam (Shopee, Lazada).

### Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.10+
- **Framework Web Automation**: Playwright
- **Vietnamese NLP**: PhoBERT (vinai/phobert-base-v2)
- **Action Planning**: ViT5 (VietAI/vit5-base)
- **Vector Database**: FAISS / ChromaDB
- **Deep Learning**: PyTorch, Transformers

---

## 🏗️ Kiến trúc 4 Layers

Hệ thống được thiết kế theo kiến trúc **4 tầng** (4-layer pipeline):

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                               │
│         "Tìm áo khoác nam giá dưới 500k trên Shopee"        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: PERCEPTION (Nhận thức)                            │
│  - Capture DOM tree + Screenshot                            │
│  - Distill DOM (lọc bỏ noise)                              │
│  - Extract UI elements                                      │
│  - PhoBERT encoding → 768-dim vectors                       │
│  Output: Scene representation                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: PLANNING (Lập kế hoạch)                          │
│  - ReAct reasoning (Thought → Action → Observation)        │
│  - ViT5 generates action sequence                          │
│  - Navigator agent (high-level planning)                   │
│  - Sub-agents (search, login, payment)                     │
│  Output: Action plan                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: EXECUTION (Thực thi)                              │
│  - Browser manager (Playwright)                             │
│  - Skill executor (click, type, scroll...)                 │
│  - Change observer (detect page changes)                   │
│  - Safety guardrails                                        │
│  Output: Execution result                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: LEARNING (Học từ kinh nghiệm)                    │
│  - RAIL memory (vector store)                              │
│  - Trajectory buffer                                        │
│  - Self-improvement                                         │
│  - Error analysis                                           │
│  Output: Updated knowledge base                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc Thư mục

```
WOA-Agent/
│
├── 📄 README.md                    # Giới thiệu dự án
├── 📄 THEORY.md                    # Kiến trúc & nghiên cứu
├── 📄 SETUP.md                     # Hướng dẫn cài đặt
├── 📄 RUN-INSTRUCTIONS.md          # Hướng dẫn chạy (FILE NÀY!)
├── 📄 CODEBASE-OVERVIEW.md         # Tổng quan codebase
├── 📄 COMPLETE-DOCUMENTATION.md    # Tài liệu đầy đủ
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 pyproject.toml              # Project metadata
├── 📄 Makefile                     # Build & run commands
├── 📄 .env                         # Environment variables
│
├── 📂 src/                         # Source code chính
│   ├── 📂 models/                  # AI Models
│   │   ├── phobert_encoder.py     # PhoBERT (Vietnamese encoding)
│   │   └── vit5_planner.py        # ViT5 (Action generation)
│   │
│   ├── 📂 perception/              # Layer 1: Nhận thức
│   │   ├── dom_distiller.py       # DOM tree processing
│   │   ├── screenshot.py          # Screenshot capture
│   │   ├── ui_detector.py         # UI element detection
│   │   ├── embedding.py           # PhoBERT embeddings
│   │   └── scene_representation.py # Scene state
│   │
│   ├── 📂 planning/                # Layer 2: Lập kế hoạch
│   │   ├── react_engine.py        # ReAct reasoning loop
│   │   ├── planner_agent.py       # High-level planner
│   │   ├── navigator_agent.py     # Navigation logic
│   │   ├── change_observer.py     # Page change detection
│   │   └── sub_agents/            # Specialized agents
│   │       ├── base_agent.py      # Base class
│   │       ├── search_agent.py    # Search tasks
│   │       ├── login_agent.py     # Login tasks
│   │       ├── payment_agent.py   # Payment tasks
│   │       └── form_agent.py      # Form filling
│   │
│   ├── 📂 execution/               # Layer 3: Thực thi
│   │   ├── browser_manager.py     # Playwright wrapper
│   │   └── skill_executor.py      # Low-level actions
│   │
│   ├── 📂 learning/                # Layer 4: Học tập
│   │   ├── self_improvement.py    # Self-learning
│   │   ├── error_analyzer.py      # Error analysis
│   │   └── memory/                # Memory systems
│   │       ├── rail.py            # RAIL memory
│   │       ├── vector_store.py    # Vector DB wrapper
│   │       └── trajectory_buffer.py # Action history
│   │
│   ├── 📂 orchestrator/            # Điều phối trung tâm
│   │   ├── agent_orchestrator.py  # Main control loop
│   │   ├── state_manager.py       # State tracking
│   │   └── safety_guardrails.py   # Safety checks
│   │
│   └── 📂 utils/                   # Utilities
│       ├── logger.py              # Logging system
│       ├── metrics.py             # Performance tracking
│       ├── validators.py          # Input validation
│       └── vietnamese_processor.py # Vietnamese text processing
│
├── 📂 config/                      # Configuration
│   ├── settings.py                # Main settings
│   ├── models.yaml                # Model configs
│   ├── skills.yaml                # Skill definitions
│   ├── logging.yaml               # Logging config
│   └── data_catalog.yaml          # Data paths
│
├── 📂 scripts/                     # Utility scripts
│   ├── download_models.py         # Download models
│   ├── prepare_data.py            # Data preparation
│   ├── train_vit5.py              # ViT5 training
│   ├── train_phobert.py           # PhoBERT training
│   ├── collect_trajectories.py    # Collect training data
│   └── evaluate_agent.py          # Performance evaluation
│
├── 📂 tests/                       # Test suite
│   ├── unit/                      # Unit tests
│   │   ├── test_perception.py
│   │   ├── test_planning.py
│   │   ├── test_execution.py
│   │   └── test_learning.py
│   ├── integration/               # Integration tests
│   │   ├── test_agent_flow.py
│   │   └── test_shopee_workflow.py
│   ├── performance/               # Performance tests
│   │   └── test_gpu.py
│   └── full_pipeline_test.py      # Full pipeline test
│
├── 📂 data/                        # Runtime data (tự sinh)
│   ├── vector_store/              # FAISS/Chroma DB
│   ├── screenshots/               # Screenshots
│   ├── trajectories/              # Action history
│   └── logs/                      # Log files
│
├── 📂 checkpoints/                 # Model checkpoints (tự sinh)
│   ├── vit5/                      # ViT5 fine-tuned
│   └── phobert/                   # PhoBERT fine-tuned
│
├── 📂 docs/                        # Additional documentation
├── 📂 notebooks/                   # Jupyter notebooks (nếu có)
└── 📂 cache/                       # HuggingFace cache (tự sinh)
```

---

## 🧩 Components Chi tiết

### 1️⃣ **Layer 1: Perception (Nhận thức)**

**Mục đích**: Chuyển đổi trạng thái web page thành representation mà AI có thể hiểu.

#### `perception/dom_distiller.py`
- **Chức năng**: Lọc DOM tree, loại bỏ các thẻ không quan trọng
- **Input**: Raw HTML DOM
- **Output**: Distilled DOM (chỉ giữ các elements tương tác được)
- **Thuật toán**: 3 modes (simple, semantic, adaptive)

#### `perception/screenshot.py`
- **Chức năng**: Capture screenshot của page
- **Output**: PNG image + metadata

#### `perception/ui_detector.py`
- **Chức năng**: Detect các UI elements (buttons, inputs, links)
- **Method**: CSS selectors + heuristics

#### `perception/embedding.py`
- **Chức năng**: Encode text sang vectors bằng PhoBERT
- **Model**: vinai/phobert-base-v2 (768-dim)
- **Output**: Dense embeddings cho semantic matching

#### `perception/scene_representation.py`
- **Chức năng**: Tổng hợp tất cả thông tin perception
- **Output**: Scene state (DOM + screenshot + embeddings)

---

### 2️⃣ **Layer 2: Planning (Lập kế hoạch)**

**Mục đích**: Quyết định action nào cần thực hiện dựa trên observation.

#### `planning/react_engine.py`
- **Chức năng**: Implement ReAct reasoning loop
- **Pattern**: 
  ```
  Thought → Action → Observation → Thought → ...
  ```
- **Stopping criteria**: Goal achieved hoặc max steps

#### `planning/planner_agent.py`
- **Chức năng**: High-level task planning
- **Decompose**: Chia task phức tạp thành sub-tasks
- **Output**: Task plan với steps

#### `planning/navigator_agent.py`
- **Chức năng**: Navigation logic (page transitions)
- **Handles**: URL changes, redirects, popups

#### `planning/change_observer.py`
- **Chức năng**: Detect khi page thay đổi
- **Method**: MutationObserver pattern
- **Output**: Change events

#### `planning/sub_agents/`
Các specialized agents cho từng loại task:

- **`search_agent.py`**: Tìm kiếm sản phẩm
- **`login_agent.py`**: Đăng nhập tài khoản
- **`payment_agent.py`**: Xử lý thanh toán
- **`form_agent.py`**: Điền form

Mỗi agent có:
- `can_handle(task)`: Check xem có xử lý được task không
- `execute(task)`: Thực thi task

---

### 3️⃣ **Layer 3: Execution (Thực thi)**

**Mục đích**: Thực hiện các browser actions.

#### `execution/browser_manager.py`
- **Chức năng**: Quản lý Playwright browser lifecycle
- **Methods**:
  - `launch()`: Khởi động browser
  - `new_page()`: Tạo tab mới
  - `close()`: Đóng browser

#### `execution/skill_executor.py`
- **Chức năng**: Thực thi low-level browser actions
- **Skills**:
  - `click(selector)`: Click element
  - `type(selector, text)`: Nhập text
  - `scroll(direction)`: Scroll page
  - `wait(selector)`: Đợi element
  - `goto(url)`: Navigate to URL
  - `screenshot()`: Chụp màn hình
  - `extract_text(selector)`: Lấy text

---

### 4️⃣ **Layer 4: Learning (Học tập)**

**Mục đích**: Cải thiện performance qua thời gian.

#### `learning/memory/rail.py`
- **Chức năng**: RAIL (Retrieve-Augment-Improve-Learn) memory
- **Storage**: Vector database (FAISS/Chroma)
- **Workflow**:
  1. Retrieve: Tìm trajectories tương tự
  2. Augment: Bổ sung context
  3. Improve: Học từ successes/failures
  4. Learn: Update knowledge base

#### `learning/memory/vector_store.py`
- **Chức năng**: Wrapper cho FAISS/Chroma
- **Methods**:
  - `add(vectors, metadata)`: Thêm vectors
  - `search(query_vector, k)`: Tìm k nearest neighbors
  - `delete(ids)`: Xóa vectors

#### `learning/memory/trajectory_buffer.py`
- **Chức năng**: Lưu trữ action trajectories
- **Format**: `[(state, action, reward, next_state), ...]`

#### `learning/self_improvement.py`
- **Chức năng**: Self-learning từ experience
- **Methods**:
  - Analyze successes
  - Analyze failures
  - Update policies

#### `learning/error_analyzer.py`
- **Chức năng**: Phân tích lỗi để improve
- **Output**: Error patterns, root causes

---

### 🎛️ **Orchestrator (Điều phối)**

#### `orchestrator/agent_orchestrator.py`
- **Vai trò**: Main control loop, điều phối tất cả layers
- **Flow**:
  ```python
  while not goal_achieved and steps < max_steps:
      # 1. Perception
      scene = perceive(page)
      
      # 2. Planning
      action = plan(scene, history)
      
      # 3. Execution
      result = execute(action)
      
      # 4. Learning
      learn(scene, action, result)
  ```

#### `orchestrator/state_manager.py`
- **Chức năng**: Track agent state
- **State**: Current URL, history, variables

#### `orchestrator/safety_guardrails.py`
- **Chức năng**: Safety checks
- **Prevents**:
  - Malicious actions
  - Infinite loops
  - Sensitive data leakage

---

### 🤖 **Models**

#### `models/phobert_encoder.py`
- **Model**: vinai/phobert-base-v2
- **Params**: 135M
- **Output**: 768-dim embeddings
- **Use case**: Encode Vietnamese text
- **⚠️ KHÔNG dùng để generate text!**

#### `models/vit5_planner.py`
- **Model**: VietAI/vit5-base
- **Params**: 250M
- **Output**: Action sequences (text generation)
- **Use case**: Generate action plans
- **Fine-tuning**: LoRA (low-rank adaptation)

---

### 🛠️ **Utils**

#### `utils/logger.py`
- **Chức năng**: Logging system
- **Features**: Colored output, file rotation

#### `utils/metrics.py`
- **Chức năng**: Track performance metrics
- **Metrics**: Success rate, execution time, steps

#### `utils/validators.py`
- **Chức năng**: Validate inputs
- **Validates**: URLs, selectors, actions, queries

#### `utils/vietnamese_processor.py`
- **Chức năng**: Vietnamese text processing
- **Features**:
  - Remove diacritics
  - Normalize text
  - Extract keywords
  - Tokenization

---

## 🔄 Luồng Xử lý (Workflow)

### Ví dụ: "Tìm áo khoác nam giá dưới 500k trên Shopee"

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 0: Khởi tạo                                            │
│ - Load models (PhoBERT, ViT5)                              │
│ - Launch browser                                            │
│ - Initialize state manager                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Navigate to Shopee                                 │
│ Action: goto("https://shopee.vn")                          │
│ Executor: browser_manager.goto()                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Perception                                         │
│ - Capture DOM tree                                         │
│ - DOM distiller: Filter chỉ giữ interactive elements       │
│ - Screenshot: Capture màn hình                             │
│ - PhoBERT: Encode "áo khoác nam" → vector                  │
│ Output: Scene representation                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Planning                                           │
│ - ReAct: Thought = "Cần tìm search box"                   │
│ - UI detector: Tìm search input selector                   │
│ - ViT5: Generate action = 'click("#search-input")'        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Execution                                          │
│ - Skill: click("#search-input")                           │
│ - Playwright: page.click("#search-input")                 │
│ - Change observer: Detect input focused                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Planning (tiếp)                                    │
│ - ReAct: Thought = "Cần nhập từ khóa"                     │
│ - ViT5: Generate action = 'type("#search-input", "áo...")' │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Execution                                          │
│ - Skill: type("#search-input", "áo khoác nam")            │
│ - Playwright: page.fill("#search-input", "áo khoác nam")  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Planning & Execution                               │
│ - Click search button                                       │
│ - Wait for results                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: Apply filters                                      │
│ - Filter by price: < 500k                                  │
│ - Click filter button                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 9: Extract results                                    │
│ - Parse product cards                                       │
│ - Extract: name, price, rating, link                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 10: Learning                                          │
│ - Store trajectory in vector DB                            │
│ - Update success metrics                                    │
│ - Save for future similar tasks                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
                        ✅ DONE!
```

---

## 🧪 Testing Strategy

### Unit Tests (`tests/unit/`)
Test từng component riêng lẻ:
- `test_perception.py`: DOM distiller, screenshot, embeddings
- `test_planning.py`: ReAct engine, sub-agents
- `test_execution.py`: Browser manager, skill executor
- `test_learning.py`: RAIL memory, trajectory buffer

### Integration Tests (`tests/integration/`)
Test tương tác giữa các components:
- `test_agent_flow.py`: Full pipeline
- `test_shopee_workflow.py`: E2E Shopee workflow

### Performance Tests (`tests/performance/`)
- `test_gpu.py`: GPU utilization, throughput

---

## 📊 Data Flow

```
User Query (text)
    ↓
PhoBERT Encoder → Query Embedding (768-dim)
    ↓
Vector Store → Retrieve similar trajectories
    ↓
ViT5 Planner → Generate action sequence
    ↓
Skill Executor → Browser actions
    ↓
Page State → Observation
    ↓
Learning → Store trajectory
    ↓
Updated Knowledge Base
```

---

## ⚙️ Configuration Files

### `config/settings.py`
- Runtime settings (max_steps, headless, device)
- Environment variables

### `config/models.yaml`
```yaml
phobert:
  model_name: vinai/phobert-base-v2
  max_length: 256
  device: cuda

vit5:
  model_name: VietAI/vit5-base
  max_length: 512
  device: cuda
```

### `config/skills.yaml`
Định nghĩa các skills:
```yaml
skills:
  - name: click
    params: [selector]
  - name: type
    params: [selector, text]
  - name: scroll
    params: [direction]
```

---

## 🚀 Entry Points

### 1. Run Agent
```python
# File: src/orchestrator/agent_orchestrator.py
if __name__ == "__main__":
    agent = AgentOrchestrator(max_steps=30)
    result = asyncio.run(
        agent.execute_task(
            query="Tìm laptop Dell",
            start_url="https://shopee.vn"
        )
    )
```

### 2. Train Models
```bash
# scripts/train_vit5.py
python scripts/train_vit5.py --epochs 10 --batch-size 16
```

### 3. Evaluate
```bash
# scripts/evaluate_agent.py
python scripts/evaluate_agent.py --benchmark shopee
```

---

## 🔍 Debug Tips

### 1. Enable verbose logging
```python
from src.utils.logger import setup_logging
setup_logging(level="DEBUG")
```

### 2. Visualize browser
```python
agent = AgentOrchestrator(headless=False)
```

### 3. Inspect trajectories
```python
from src.learning.memory.trajectory_buffer import TrajectoryBuffer
buffer = TrajectoryBuffer()
buffer.load("data/trajectories/latest.json")
print(buffer.trajectories)
```

### 4. Check vector store
```python
from src.learning.memory.vector_store import VectorStore
store = VectorStore()
results = store.search(query_vector, k=5)
```

---

## 📦 Dependencies Quan trọng

### Core
- `torch`: Deep learning framework
- `transformers`: PhoBERT, ViT5
- `playwright`: Browser automation

### Vietnamese NLP
- `pyvi`: Vietnamese tokenization
- PhoBERT model từ VinAI

### Vector DB
- `faiss-cpu` hoặc `faiss-gpu`: Vector similarity search
- `chromadb`: Alternative vector DB

### Web
- `beautifulsoup4`: HTML parsing
- `lxml`: XML/HTML processing

---

## 🎓 Learning Resources

### Papers
- **WebVoyager** (2024): Multimodal web agents
- **Agent-E** (2024): Hierarchical planning, DOM distillation
- **AgentOccam** (2024): Simple agents work best

### Vietnamese NLP
- **PhoBERT**: https://github.com/VinAIResearch/PhoBERT
- **ViT5**: https://github.com/vietai/ViT5

### Web Automation
- **Playwright Docs**: https://playwright.dev/python/

---

## 🛣️ Roadmap

### ✅ Đã hoàn thành
- [x] 4-layer architecture
- [x] PhoBERT integration
- [x] ViT5 integration
- [x] ReAct reasoning
- [x] Playwright automation
- [x] RAIL memory

### 🚧 Đang phát triển
- [ ] Multi-page workflows
- [ ] Vision-language model (GPT-4V)
- [ ] Reinforcement learning
- [ ] Multi-agent coordination

### 🔮 Tương lai
- [ ] Support thêm platforms (Tiki, Sendo)
- [ ] Mobile app automation
- [ ] Voice interface

---

## 💡 Best Practices

### Code Style
- Follow PEP 8
- Type hints for all functions
- Docstrings (Google style)

### Testing
- 80%+ code coverage
- Integration tests for critical paths
- Performance benchmarks

### Git
- Feature branches
- Descriptive commit messages
- Pull requests for review

---

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Write tests
4. Submit PR

---

## 📞 Support

- **Documentation**: Xem các file `.md` trong repo
- **Issues**: GitHub Issues
- **Discord**: (Link nếu có)

---

## 📝 Glossary

- **DOM**: Document Object Model - Cấu trúc HTML tree
- **PhoBERT**: Vietnamese BERT model for encoding
- **ViT5**: Vietnamese T5 model for generation
- **ReAct**: Reasoning + Acting framework
- **RAIL**: Retrieve-Augment-Improve-Learn memory
- **Trajectory**: Sequence of (state, action, reward)
- **Skill**: Low-level browser action (click, type, etc.)
- **Sub-agent**: Specialized agent for specific task type

---

**Chúc bạn code vui vẻ! 🚀**
