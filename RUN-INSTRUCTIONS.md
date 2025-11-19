# 🚀 Hướng dẫn Chạy WOA Agent - Môi trường Local

## 📋 Môi trường Python

Dự án này sử dụng Python environment tại:
```
F:\WebOrdering_Automation\woa\python.exe
```

## 🔧 Cài đặt ban đầu

### Bước 1: Cài đặt Dependencies

```powershell
# Di chuyển vào thư mục dự án
cd F:\WebOrdering_Automation\WOA-Agent

# Cài đặt các package Python cần thiết
F:\WebOrdering_Automation\woa\Scripts\pip.exe install -r requirements.txt
```

### Bước 2: Cài đặt Playwright Browsers

```powershell
# Cài đặt trình duyệt Chromium cho Playwright
F:\WebOrdering_Automation\woa\python.exe -m playwright install chromium

# (Tùy chọn) Cài đặt thêm Firefox hoặc WebKit
F:\WebOrdering_Automation\woa\python.exe -m playwright install firefox
```

### Bước 3: Tải các Models Vietnamese

```powershell
# Tải PhoBERT và ViT5 models
F:\WebOrdering_Automation\woa\python.exe scripts/training/download_models.py
```

### Bước 4: Kiểm tra cài đặt

```powershell
# Kiểm tra import thành công
F:\WebOrdering_Automation\woa\python.exe -c "import src; print('✓ Installation OK')"

# Kiểm tra Playwright
F:\WebOrdering_Automation\woa\python.exe -c "from playwright.sync_api import sync_playwright; print('✓ Playwright OK')"

# Kiểm tra transformers
F:\WebOrdering_Automation\woa\python.exe -c "import transformers; print('✓ Transformers OK')"
```

---

## ▶️ Chạy Agent

### Phương pháp 1: Sử dụng Makefile (Khuyến nghị)

```powershell
# Chạy agent
make run

# Hoặc các lệnh khác
make test              # Chạy tests
make test-unit         # Chạy unit tests
make test-integration  # Chạy integration tests
make format            # Format code
```

### Phương pháp 2: Chạy trực tiếp Python

```powershell
# Chạy agent orchestrator
F:\WebOrdering_Automation\woa\python.exe -m src.orchestrator.agent_orchestrator

# Chạy một test cụ thể
F:\WebOrdering_Automation\woa\python.exe tests/full_pipeline_test.py

# Chạy test Shopee workflow
F:\WebOrdering_Automation\woa\python.exe tests/integration/test_shopee_workflow.py
```

### Phương pháp 3: Sử dụng Python Script

Tạo file `run_agent.py`:

```python
import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def main():
    # Khởi tạo agent
    agent = AgentOrchestrator(
        max_steps=30,
        headless=False  # Hiển thị browser để debug
    )
    
    # Thực thi task
    result = await agent.execute_task(
        query="Tìm áo khoác nam giá dưới 500k trên Shopee",
        start_url="https://shopee.vn"
    )
    
    # In kết quả
    print(f"✅ Success: {result['success']}")
    print(f"📊 Steps: {result['steps']}")
    print(f"📝 History: {len(result['history'])} actions")

if __name__ == "__main__":
    asyncio.run(main())
```

Chạy:
```powershell
F:\WebOrdering_Automation\woa\python.exe run_agent.py
```

---

## 🧪 Chạy Tests

### Tất cả tests

```powershell
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/ -v
```

### Unit tests (nhanh)

```powershell
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/unit/ -v
```

### Integration tests (chậm hơn)

```powershell
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/integration/ -v
```

### Test một file cụ thể

```powershell
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/unit/test_perception.py -v
```

### Test với coverage

```powershell
F:\WebOrdering_Automation\woa\Scripts\pytest.exe tests/ --cov=src --cov-report=html
```

---

## 🔍 Debug và Development

### Chạy với debug mode

```python
# Trong file Python của bạn
import logging
from src.utils.logger import setup_logging

# Enable debug logging
setup_logging(level="DEBUG")

# Code của bạn...
```

### Hoặc set environment variable:

```powershell
# PowerShell
$env:AGENT_LOG_LEVEL = "DEBUG"
F:\WebOrdering_Automation\woa\python.exe -m src.orchestrator.agent_orchestrator
```

### Chạy browser ở chế độ hiển thị (không headless)

```python
agent = AgentOrchestrator(
    max_steps=30,
    headless=False  # Hiển thị browser
)
```

Hoặc:

```powershell
$env:AGENT_HEADLESS = "false"
F:\WebOrdering_Automation\woa\python.exe -m src.orchestrator.agent_orchestrator
```

---

## 📊 Training Models

### Chuẩn bị dữ liệu

```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/preprocessing/split_dataset.py
```

### Train ViT5 (Action Planner)

```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/training/train_controller.py
```

### Train PhoBERT (Encoder)

```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/preprocessing/build_embeddings.py
```

### Thu thập trajectories

```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/data_collection/collect_raw_trajectories.py
```

### Đánh giá Agent

```powershell
F:\WebOrdering_Automation\woa\python.exe scripts/evaluation/run_benchmark.py
```

---

## ⚙️ Configuration

### Environment Variables

Tạo file `.env` trong thư mục `WOA-Agent/`:

```env
# Agent Configuration
AGENT_MAX_STEPS=25
AGENT_HEADLESS=true
AGENT_BROWSER=chromium
AGENT_VIEWPORT_WIDTH=1280
AGENT_VIEWPORT_HEIGHT=720
AGENT_LOG_LEVEL=INFO
AGENT_DATA_DIR=data

# Device (cuda/cpu)
AGENT_DEVICE=cuda
CUDA_AVAILABLE=true

# Models
PHOBERT_MODEL=vinai/phobert-base-v2
VIT5_MODEL=VietAI/vit5-base

# Vector Database
VECTOR_DB_TYPE=faiss
VECTOR_DB_PATH=data/vector_store

# Learning
ENABLE_LEARNING=true
ENABLE_GUARDRAILS=true
```

### Load environment:

```python
from dotenv import load_dotenv
load_dotenv()

# Config sẽ tự động được load từ .env
from config.settings import settings
print(settings.max_steps)  # 25
```

---

## 📁 Cấu trúc Dữ liệu

Sau khi chạy, các thư mục sau sẽ được tạo:

```
WOA-Agent/
├── data/                    # Dữ liệu runtime
│   ├── vector_store/        # FAISS/Chroma DB
│   ├── screenshots/         # Screenshots tự động
│   ├── trajectories/        # Lịch sử actions
│   └── logs/                # Log files
├── checkpoints/             # Model checkpoints
│   ├── vit5/               # ViT5 fine-tuned
│   └── phobert/            # PhoBERT fine-tuned
└── cache/                   # Cache models từ HuggingFace
```

---

## 🐛 Troubleshooting

### Lỗi: "Module not found"

```powershell
# Đảm bảo đã cài đặt đầy đủ dependencies
F:\WebOrdering_Automation\woa\Scripts\pip.exe install -r requirements.txt
```

### Lỗi: "Playwright browser not found"

```powershell
# Cài đặt lại browsers
F:\WebOrdering_Automation\woa\python.exe -m playwright install chromium --force
```

### Lỗi: "CUDA out of memory"

Giảm batch size hoặc chuyển sang CPU:

```python
# Trong code
agent = AgentOrchestrator()
agent.device = "cpu"
```

Hoặc:

```powershell
$env:AGENT_DEVICE = "cpu"
```

### Lỗi: "Permission denied"

Chạy PowerShell với quyền Administrator.

### Lỗi: Model download fails

```powershell
# Download thủ công
F:\WebOrdering_Automation\woa\python.exe -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base-v2')"
F:\WebOrdering_Automation\woa\python.exe -c "from transformers import AutoModel; AutoModel.from_pretrained('VietAI/vit5-base')"
```

---

## 📝 Logs

### Xem logs realtime:

```powershell
# PowerShell
Get-Content data/logs/agent.log -Wait
```

### Logs được lưu tại:

```
data/logs/
├── agent.log           # Main agent log
├── perception.log      # DOM/Screenshot processing
├── planning.log        # ReAct reasoning
├── execution.log       # Browser actions
└── learning.log        # Memory/Learning
```

---

## 🎯 Ví dụ Workflows

### 1. Tìm kiếm sản phẩm trên Shopee

```python
result = await agent.execute_task(
    query="Tìm laptop Dell giá dưới 15 triệu",
    start_url="https://shopee.vn"
)
```

### 2. So sánh giá trên Lazada

```python
result = await agent.execute_task(
    query="So sánh giá iPhone 15 Pro Max",
    start_url="https://lazada.vn"
)
```

### 3. Thêm vào giỏ hàng

```python
result = await agent.execute_task(
    query="Thêm áo khoác nam màu đen size L vào giỏ hàng",
    start_url="https://shopee.vn"
)
```

---

## 📚 Tài liệu tham khảo

- [COMPLETE-DOCUMENTATION.md](./COMPLETE-DOCUMENTATION.md) - Tài liệu đầy đủ
- [THEORY.md](./THEORY.md) - Kiến trúc và research
- [SETUP.md](./SETUP.md) - Hướng dẫn setup chi tiết
- [CODEBASE-OVERVIEW.md](./CODEBASE-OVERVIEW.md) - Tổng quan codebase

---

## 💡 Tips

1. **Chạy với headless=False** khi develop để debug browser
2. **Enable DEBUG logs** để xem chi tiết quá trình xử lý
3. **Sử dụng pytest với -v** để xem output chi tiết
4. **Giảm max_steps** khi test nhanh
5. **Sử dụng Makefile** để đơn giản hóa commands

---

## ✅ Checklist Chạy Lần Đầu

- [ ] Đã cài đặt dependencies (`pip install -r requirements.txt`)
- [ ] Đã cài đặt Playwright browsers (`playwright install chromium`)
- [ ] Đã tải models Vietnamese (`python scripts/training/download_models.py`)
- [ ] Đã kiểm tra import thành công
- [ ] Đã tạo file `.env` với config phù hợp
- [ ] Đã test chạy agent với một task đơn giản
- [ ] Đã kiểm tra logs được tạo trong `data/logs/`

---

**Chúc bạn thành công! 🎉**
