# 🎯 WOA Agent - Plan Hiểu Codebase

## ✅ Tóm tắt Thay đổi

Tôi đã hoàn thành các thay đổi sau:

### 1. **Xóa Docker** ✅
- ❌ Đã xóa `Dockerfile`
- ❌ Đã xóa `docker-compose.yml`

### 2. **Cập nhật Makefile** ✅
- ✅ Sử dụng Python environment: `F:\WebOrdering_Automation\woa\python.exe`
- ✅ Thêm các commands hữu ích: install, run, test, train, evaluate...

### 3. **Cập nhật README.md** ✅
- ✅ Thay thế Docker instructions bằng local setup
- ✅ Hướng dẫn sử dụng Makefile với Python env của bạn

### 4. **Tạo RUN-INSTRUCTIONS.md** ✅
- ✅ Hướng dẫn chi tiết cài đặt và chạy
- ✅ Troubleshooting và debugging tips
- ✅ Ví dụ workflows cụ thể

### 5. **Tạo CODEBASE-OVERVIEW.md** ✅
- ✅ Giải thích kiến trúc 4 layers
- ✅ Mô tả chi tiết từng component
- ✅ Luồng xử lý workflow
- ✅ Data flow và testing strategy

---

## 📚 Plan Để Hiểu Codebase

Đây là **lộ trình từng bước** để bạn hiểu rõ toàn bộ codebase:

---

### 🎓 GIAI ĐOẠN 1: Hiểu Tổng Quan (30 phút)

#### Bước 1.1: Đọc Documentation
Đọc theo thứ tự:
1. ✅ **README.md** - Overview dự án
2. ✅ **CODEBASE-OVERVIEW.md** - Kiến trúc tổng thể (FILE MỚI!)
3. ✅ **THEORY.md** - Nghiên cứu và lý thuyết
4. ✅ **RUN-INSTRUCTIONS.md** - Cách chạy (FILE MỚI!)

#### Bước 1.2: Hiểu Kiến Trúc 4 Layers
```
User Query
    ↓
1️⃣ PERCEPTION (Nhận thức) - Capture DOM/Screenshot
    ↓
2️⃣ PLANNING (Lập kế hoạch) - ReAct reasoning
    ↓
3️⃣ EXECUTION (Thực thi) - Browser actions
    ↓
4️⃣ LEARNING (Học tập) - Store trajectory
```

#### Bước 1.3: Hiểu Data Flow
```
Text Query → PhoBERT (768-dim) → Vector Store
                ↓
            ViT5 Planner → Action Sequence
                ↓
            Playwright → Browser Actions
                ↓
            Learning → Updated KB
```

---

### 💻 GIAI ĐOẠN 2: Chạy Code (1 giờ)

#### Bước 2.1: Cài đặt
```powershell
cd F:\WebOrdering_Automation\WOA-Agent

# Cài đặt dependencies
make install

# Tải models
make download-models
```

#### Bước 2.2: Chạy Test Đơn Giản
```powershell
# Test import
F:\WebOrdering_Automation\woa\python.exe -c "import src; print('OK')"

# Test perception
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/unit/test_perception.py -v

# Test execution
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/unit/test_execution.py -v
```

#### Bước 2.3: Chạy Full Pipeline (headless=False để xem browser)
Tạo file `test_run.py`:
```python
import asyncio
from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def main():
    agent = AgentOrchestrator(
        max_steps=10,
        headless=False  # Hiển thị browser
    )
    
    result = await agent.execute_task(
        query="Tìm laptop Dell",
        start_url="https://shopee.vn"
    )
    
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")

asyncio.run(main())
```

Chạy:
```powershell
F:\WebOrdering_Automation\woa\python.exe test_run.py
```

---

### 🔍 GIAI ĐOẠN 3: Đọc Code Theo Luồng (2-3 giờ)

#### Bước 3.1: Entry Point - Orchestrator
**File**: `src/orchestrator/agent_orchestrator.py`

Đọc để hiểu:
- `__init__()`: Khởi tạo các components
- `execute_task()`: Main control loop
- Luồng: Perception → Planning → Execution → Learning

#### Bước 3.2: Layer 1 - Perception
**Đọc theo thứ tự**:

1. `src/perception/dom_distiller.py`
   - Hiểu cách filter DOM tree
   - 3 modes: simple, semantic, adaptive

2. `src/perception/screenshot.py`
   - Capture screenshot
   - Metadata extraction

3. `src/perception/embedding.py`
   - PhoBERT encoding
   - Vector generation (768-dim)

4. `src/perception/scene_representation.py`
   - Tổng hợp perception data

#### Bước 3.3: Layer 2 - Planning
**Đọc theo thứ tự**:

1. `src/planning/react_engine.py`
   - **QUAN TRỌNG**: ReAct reasoning loop
   - Thought → Action → Observation pattern

2. `src/planning/planner_agent.py`
   - High-level task planning
   - Task decomposition

3. `src/planning/sub_agents/search_agent.py`
   - Example sub-agent
   - `can_handle()` và `execute()`

4. `src/models/vit5_planner.py`
   - ViT5 action generation
   - JSON parsing và fallback

#### Bước 3.4: Layer 3 - Execution
**Đọc theo thứ tự**:

1. `src/execution/browser_manager.py`
   - Playwright wrapper
   - Browser lifecycle

2. `src/execution/skill_executor.py`
   - **QUAN TRỌNG**: Low-level skills
   - click, type, scroll, wait, goto...

#### Bước 3.5: Layer 4 - Learning
**Đọc theo thứ tự**:

1. `src/learning/memory/vector_store.py`
   - FAISS/Chroma wrapper
   - Vector operations

2. `src/learning/memory/rail.py`
   - RAIL memory system
   - Retrieve → Augment → Improve → Learn

3. `src/learning/memory/trajectory_buffer.py`
   - Store action history

---

### 🧪 GIAI ĐOẠN 4: Chạy và Debug (1-2 giờ)

#### Bước 4.1: Debug với Breakpoints
Sử dụng VSCode:

1. Tạo `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Run Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/test_run.py",
            "console": "integratedTerminal",
            "python": "F:\\WebOrdering_Automation\\woa\\python.exe"
        }
    ]
}
```

2. Set breakpoints tại:
   - `agent_orchestrator.py:execute_task()` (line ~150)
   - `react_engine.py:step()` (line ~80)
   - `skill_executor.py:execute()` (line ~100)

3. Press F5 → Debug step by step

#### Bước 4.2: Enable Debug Logs
```python
from src.utils.logger import setup_logging
setup_logging(level="DEBUG")
```

#### Bước 4.3: Inspect Vector Store
```python
from src.learning.memory.vector_store import VectorStore

store = VectorStore()
# Thêm một trajectory
store.add(embeddings, metadata)

# Search
results = store.search(query_vector, k=5)
print(results)
```

---

### 📊 GIAI ĐOẠN 5: Thử Nghiệm (1 giờ)

#### Bước 5.1: Modify một Sub-Agent
Ví dụ: Thêm logging vào `search_agent.py`:

```python
# Trong search_agent.py, method execute()
async def execute(self, task, observation):
    logger.info(f"🔍 SearchAgent: Executing task: {task}")
    
    # Existing code...
    
    logger.info(f"✅ SearchAgent: Found {len(results)} results")
    return results
```

#### Bước 5.2: Test với Task Mới
```python
result = await agent.execute_task(
    query="Tìm điện thoại iPhone 15 Pro Max",
    start_url="https://shopee.vn"
)
```

#### Bước 5.3: Analyze Trajectory
```python
# Sau khi chạy, xem trajectory
buffer = TrajectoryBuffer()
buffer.load("data/trajectories/latest.json")

for step in buffer.trajectories:
    print(f"State: {step['state']}")
    print(f"Action: {step['action']}")
    print(f"Reward: {step['reward']}")
```

---

### 🎓 GIAI ĐOẠN 6: Hiểu Sâu Models (2 giờ)

#### Bước 6.1: PhoBERT
**File**: `src/models/phobert_encoder.py`

Thử nghiệm:
```python
from src.models.phobert_encoder import PhoBERTEncoder

encoder = PhoBERTEncoder()

# Encode text
text = "Tìm áo khoác nam giá rẻ"
embedding = encoder.encode(text)
print(embedding.shape)  # (768,)

# Compute similarity
text1 = "áo khoác nam"
text2 = "jacket for men"
sim = encoder.compute_similarity(text1, text2)
print(f"Similarity: {sim}")
```

#### Bước 6.2: ViT5
**File**: `src/models/vit5_planner.py`

Thử nghiệm:
```python
from src.models.vit5_planner import ViT5Planner

planner = ViT5Planner()

# Generate action
observation = "Đang ở trang chủ Shopee, có search box"
thought = "Cần tìm kiếm laptop Dell"

action = planner.generate_action(observation, thought)
print(f"Action: {action}")
```

---

### 🔬 GIAI ĐOẠN 7: Training (Nâng cao - 3+ giờ)

#### Bước 7.1: Prepare Data
```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/prepare_data.py
```

#### Bước 7.2: Train ViT5
```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/train_vit5.py --epochs 3
```

#### Bước 7.3: Evaluate
```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/evaluate_agent.py
```

---

## 📋 Checklist Hiểu Codebase

Sau khi hoàn thành các giai đoạn trên, bạn nên có thể:

### Kiến trúc
- [ ] Giải thích được 4 layers và vai trò của mỗi layer
- [ ] Vẽ được data flow từ query → result
- [ ] Hiểu được ReAct reasoning loop

### Components
- [ ] Biết cách DOM distiller hoạt động
- [ ] Hiểu PhoBERT vs ViT5 khác nhau như thế nào
- [ ] Biết cách Playwright được sử dụng
- [ ] Hiểu RAIL memory system

### Code
- [ ] Có thể chạy agent với task mới
- [ ] Biết cách debug với breakpoints
- [ ] Có thể modify sub-agent
- [ ] Hiểu cách thêm skill mới

### Advanced
- [ ] Train được ViT5 trên data mới
- [ ] Phân tích được trajectories
- [ ] Optimize được performance
- [ ] Extend được cho platform mới

---

## 🎯 Lộ trình Học Tập Đề xuất

### Tuần 1: Foundation (5-10 giờ)
- ✅ Đọc documentation
- ✅ Cài đặt và chạy tests
- ✅ Hiểu kiến trúc 4 layers
- ✅ Debug đơn giản

### Tuần 2: Deep Dive (10-15 giờ)
- ✅ Đọc code từng layer
- ✅ Chạy với headless=False
- ✅ Modify sub-agents
- ✅ Test với tasks khác nhau

### Tuần 3: Advanced (10-20 giờ)
- ✅ Hiểu models (PhoBERT, ViT5)
- ✅ Training và evaluation
- ✅ Performance optimization
- ✅ Extend cho use cases mới

---

## 💡 Tips Học Hiệu Quả

### 1. **Hands-on > Chỉ Đọc**
Chạy code ngay khi đọc, đừng chỉ đọc documentation.

### 2. **Debug là Cách Học Tốt Nhất**
Set breakpoints và step through code để hiểu flow.

### 3. **Modify Code**
Thêm logging, thay đổi parameters, xem kết quả thay đổi như thế nào.

### 4. **Start Simple**
Chạy với `max_steps=5` trước, sau đó tăng dần.

### 5. **Visualize**
Dùng `headless=False` để xem browser actions.

### 6. **Ask Questions**
Khi không hiểu, search trong code hoặc documentation.

---

## 📞 Khi Gặp Khó Khăn

### Debugging Checklist
1. ✅ Check logs trong `data/logs/`
2. ✅ Set `log_level="DEBUG"`
3. ✅ Chạy với `headless=False`
4. ✅ Breakpoint tại điểm nghi ngờ
5. ✅ Print intermediate values

### Common Issues
- **Import errors**: Check PYTHONPATH
- **Model not found**: Run `make download-models`
- **Browser fails**: Reinstall Playwright browsers
- **CUDA error**: Switch to CPU with `device="cpu"`

---

## 🎉 Kết Luận

Bạn đã có:
1. ✅ **RUN-INSTRUCTIONS.md** - Hướng dẫn chạy chi tiết
2. ✅ **CODEBASE-OVERVIEW.md** - Tổng quan kiến trúc
3. ✅ **PLAN.md** (file này!) - Lộ trình học tập

**Bắt đầu từ Giai đoạn 1 và tiến dần!**

Chúc bạn học tốt! 🚀
