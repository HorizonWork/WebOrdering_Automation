# WOA Agent - Setup & Requirements Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Development Setup](#development-setup)

---

## System Requirements

### Hardware Requirements

#### Minimum
- **CPU**: Intel i5 or equivalent (4 cores)
- **RAM**: 8 GB
- **Storage**: 20 GB (models + data)
- **GPU**: Optional (but recommended)

#### Recommended (Production)
- **CPU**: Intel i7/i9 or AMD Ryzen 7+ (8+ cores)
- **RAM**: 32 GB
- **Storage**: 100 GB SSD (models + training data)
- **GPU**: NVIDIA V100/A100 (16GB+ VRAM)
  - Or: NVIDIA RTX 3090/4090
  - Or: NVIDIA A6000
  - Cloud: AWS p3.2xlarge, GCP A100, Azure NC24

### Software Requirements

#### Core
- **Python**: 3.10, 3.11, or 3.12
- **pip**: Latest version
- **Git**: For version control

#### Operating System
- ✅ Ubuntu 20.04+ (Recommended for Linux)
- ✅ macOS 11+
- ✅ Windows 10/11 (with WSL2 for better performance)

#### CUDA (Optional but Recommended)
- **CUDA**: 11.8 or 12.1
- **cuDNN**: 8.7+
- **TensorRT**: 8.6+ (optional for inference optimization)

### Browser Requirements
- **Chrome/Chromium**: 100+
- **Firefox**: 95+
- **Safari**: 15+ (macOS only)

---

## Installation Methods

### Method 1: Local Setup (Recommended for Development)

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/WOA-Agent.git
cd WOA-Agent
```

#### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### Step 4: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt

# GPU support (PyTorch with CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2/M3 Macs)
# pip install torch::mps  # Automatically installed on macOS
```

#### Step 5: Download Playwright Browsers

```bash
playwright install chromium firefox
```

#### Step 6: Download Pre-trained Models

```bash
python scripts/download_models.py
# Downloads: PhoBERT, ViT5, and auxiliary models
# Size: ~2.5 GB
# Location: ./checkpoints/
```

---

### Method 2: Docker Setup (Recommended for Production)

#### Step 1: Build Docker Image

```bash
# Build with CUDA support
docker build -t woa-agent:latest --build-arg CUDA_VERSION=11.8 .

# Or: Build CPU-only
docker build -t woa-agent:cpu -f Dockerfile.cpu .
```

#### Step 2: Run with Docker Compose

```bash
# Start services (GPU)
docker-compose up -d

# View logs
docker-compose logs -f woa-agent

# Stop services
docker-compose down
```

#### Step 3: Verify Installation

```bash
docker-compose exec woa-agent python -c "from src.models import PhoBERTEncoder; print('✓ OK')"
```

#### Troubleshooting Docker

```bash
# Check GPU availability
docker-compose exec woa-agent nvidia-smi

# Run tests inside container
docker-compose exec woa-agent pytest tests/unit

# Interactive shell
docker-compose exec woa-agent /bin/bash
```

---

### Method 3: Conda Setup (Alternative)

```bash
# Create environment
conda create -n woa python=3.10
conda activate woa

# Install dependencies
pip install -r requirements.txt

# GPU with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## Configuration

### Environment Variables (.env)

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# ====== MODEL CONFIGURATION ======
PHOBERT_MODEL=vinai/phobert-base-v2
VIT5_MODEL=VietAI/vit5-base
DEVICE=cuda  # or 'cpu', 'mps' for Mac

# ====== PATHS ======
CHECKPOINT_DIR=./checkpoints
DATA_DIR=./data
LOG_DIR=./logs

# ====== EXECUTION SETTINGS ======
MAX_STEPS=30
HEADLESS=false  # Set to true for production
TIMEOUT=30000  # milliseconds
BROWSER=chromium  # chromium, firefox, webkit

# ====== GPU SETTINGS ======
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=4
NUM_WORKERS=4

# ====== LOGGING ======
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json, text

# ====== VECTOR DATABASE ======
VECTOR_DB_TYPE=faiss  # faiss, chroma, pinecone
VECTOR_DB_PATH=./data/vector_store
VECTOR_DB_DIMENSION=768

# ====== API KEYS (if using cloud services) ======
OPENAI_API_KEY=  # For fallback GPT-4 if needed
ANTHROPIC_API_KEY=  # Alternative LLM
PINECONE_API_KEY=  # For cloud vector DB

# ====== DEPLOYMENT ======
ENVIRONMENT=development  # development, staging, production
DEBUG=false
WORKERS=1
PORT=8000
```

### Configuration Files

#### `config/settings.py`

Global configuration object:

```python
from config.settings import Settings

settings = Settings()
print(settings.phobert_model)  # vinai/phobert-base-v2
print(settings.max_steps)  # 30
print(settings.device)  # cuda
```

#### `config/models.yaml`

Model-specific configurations:

```yaml
phobert:
  model_name: vinai/phobert-base-v2
  embedding_dim: 768
  max_length: 256
  batch_size: 32

vit5:
  model_name: VietAI/vit5-base
  max_input_length: 512
  max_output_length: 256
  num_beams: 4
  temperature: 0.7

skills:
  timeout: 10000  # ms
  retry_count: 3
  retry_delay: 500  # ms
```

#### `config/skills.yaml`

Skill definitions:

```yaml
goto:
  name: goto
  description: Navigate to URL
  params:
    url: string (required)
  timeout: 10000

click:
  name: click
  description: Click element
  params:
    selector: string (required)
  timeout: 5000

type:
  name: type
  description: Type text in element
  params:
    selector: string (required)
    text: string (required)
  timeout: 5000
```

---

## Verification

### Quick Verification

```bash
# 1. Check Python version
python --version  # Should be 3.10+

# 2. Check virtual environment
which python  # Should show venv path

# 3. Import core modules
python -c "
from src.models.phobert_encoder import PhoBERTEncoder
from src.models.vit5_planner import ViT5Planner
from src.execution.browser_manager import BrowserManager
print('✓ All imports successful')
"

# 4. Check GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"

# 5. List downloaded models
ls -lh checkpoints/
```

### Full Test Suite

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test category
pytest tests/unit/ -v  # Unit tests only
pytest tests/integration/ -v  # Integration tests

# Test with coverage report
pytest tests/ --cov=src --cov-report=html
# Open: htmlcov/index.html

# Test specific component
pytest tests/unit/test_perception.py::test_dom_distiller -v
```

### Component Verification

```bash
# 1. Test PhoBERT
python -c "
from src.models.phobert_encoder import PhoBERTEncoder
encoder = PhoBERTEncoder()
emb = encoder.encode_text('Tìm áo khoác')
print(f'✓ PhoBERT works. Embedding shape: {emb.shape}')
"

# 2. Test ViT5
python -c "
from src.models.vit5_planner import ViT5Planner
planner = ViT5Planner()
action = planner.generate_action('Cần tìm kiếm', ['click', 'type'])
print(f'✓ ViT5 works. Action: {action}')
"

# 3. Test Playwright
python -c "
import asyncio
from src.execution.browser_manager import BrowserManager
async def test():
    manager = BrowserManager()
    page = await manager.new_page()
    await page.goto('https://example.com')
    title = await page.title()
    await manager.close()
    print(f'✓ Playwright works. Page title: {title}')
asyncio.run(test())
"

# 4. Test Vector Store
python -c "
from src.learning.memory.vector_store import VectorStore
import numpy as np
store = VectorStore(dim=768)
vec = np.random.randn(768)
store.add(vec, metadata={'test': True})
results = store.search(vec, k=1)
print(f'✓ Vector store works. Results: {len(results)}')
"
```

---

## Troubleshooting

### Common Installation Issues

#### 1. CUDA/GPU Not Found

**Error**: `CUDA unavailable` or `torch: NVIDIA GPU not found`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# If NVIDIA driver not installed:
# Ubuntu
sudo apt-get install nvidia-driver-525

# macOS - Not needed (uses Metal Acceleration)

# Windows - Download from nvidia.com

# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Playwright Browsers Not Found

**Error**: `Browser executable not found`

**Solution**:
```bash
# Reinstall Playwright browsers
playwright install chromium

# Verify installation
playwright install-deps chromium
```

#### 3. Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solution**:
```bash
# Option 1: Reduce batch size
# In .env: BATCH_SIZE=2

# Option 2: Use CPU (slower but works)
# In .env: DEVICE=cpu

# Option 3: Use smaller models
# In .env: VIT5_MODEL=VietAI/vit5-small

# Option 4: Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 4. Model Download Issues

**Error**: `Connection timeout` or `Model not found on HuggingFace`

**Solution**:
```bash
# Manual download
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="vinai/phobert-base-v2",
    cache_dir="./checkpoints"
)

# Or use Hugging Face CLI
huggingface-cli download vinai/phobert-base-v2 --cache-dir ./checkpoints
```

#### 5. Port Already in Use

**Error**: `Address already in use: 0.0.0.0:8000`

**Solution**:
```bash
# Change port in .env
PORT=8001

# Or kill existing process
lsof -i :8000
kill -9 <PID>
```

### Performance Issues

#### Slow ViT5 Generation

**Issue**: Action generation takes > 5 seconds

**Solutions**:
```bash
# 1. Use GPU
# In .env: DEVICE=cuda

# 2. Use smaller model
# In .env: VIT5_MODEL=VietAI/vit5-small

# 3. Reduce beam search
# In config/models.yaml: num_beams: 1

# 4. Enable quantization (optional)
python scripts/quantize_models.py
```

#### High Memory Usage

**Issue**: Python process uses > 10GB RAM

**Solutions**:
```bash
# 1. Reduce batch size
# In .env: BATCH_SIZE=1

# 2. Use distributed inference
# Setup with Ray or vLLM

# 3. Profile memory
python -m memory_profiler scripts/identify_bottlenecks.py
```

#### Slow Screenshots

**Issue**: Screenshot capture takes > 2 seconds

**Solutions**:
```bash
# 1. Reduce screenshot quality
# In config/settings.py: SCREENSHOT_QUALITY=80

# 2. Use headless mode (faster)
# In .env: HEADLESS=true

# 3. Use different browser
# In .env: BROWSER=chromium  # Generally fastest
```

---

## Development Setup

### IDE Setup

#### VS Code

**Install Extensions**:
- Python (Microsoft)
- Pylance
- Black Formatter
- Pylint

**settings.json**:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  }
}
```

#### PyCharm

1. Open project
2. Configure interpreter: Settings → Project → Python Interpreter → Add Interpreter → Existing Environment
3. Select venv/bin/python
4. Configure VCS: Settings → Version Control → Git
5. Enable Black formatting: Settings → Tools → Black

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Initialize hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/PyCQA/pylint
    rev: pylint-2.17.0
    hooks:
      - id: pylint

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
```

### Debugging

**Debug with VSCode**:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}
```

**Debug with pdb**:

```python
import pdb; pdb.set_trace()
# Or in Python 3.7+:
breakpoint()
```

### Documentation Build

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html

# View in browser
open _build/html/index.html
```

---

## Next Steps After Setup

1. **Read THEORY.md** - Understand architecture
2. **Review examples** - See `notebooks/` folder
3. **Run tests** - `pytest tests/`
4. **Try quick example** - Follow README.md
5. **Start development** - Modify `src/` components

---

## Getting Help

- **Documentation**: [THEORY.md](THEORY.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/WOA-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/WOA-Agent/discussions)
- **Email**: your-email@example.com

---

*Last Updated: November 15, 2025*
