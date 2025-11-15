# WOA Agent - Web Automation for Vietnamese E-commerce

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-brightgreen)](https://playwright.dev/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

**WOA Agent** is a production-ready **Web Automation Agent** designed specifically for Vietnamese e-commerce platforms (Shopee, Lazada). It combines state-of-the-art LLM-based agent research with Vietnamese language models to automate complex web tasks autonomously.

## 🎯 Quick Start

### Project Overview

This is a **4-layer hierarchical agent system** that can:
- 📱 Automatically browse and interact with web pages
- 🔍 Understand Vietnamese user queries
- 🧠 Plan multi-step workflows using ReAct reasoning
- ⚙️ Execute browser actions (click, type, scroll, etc.)
- 📚 Learn and improve from experience using RAIL

**Example Workflow:**
```
User Query: "Mua áo khoác nam màu đen, giá dưới 500k trên Shopee"
    ↓
[Perception] Extract DOM + screenshot + PhoBERT embeddings
    ↓
[Planning] ViT5 generates: goto(shopee.vn) → search("áo khoác nam") → filter(color, price)
    ↓
[Execution] Playwright: click search box → type query → apply filters
    ↓
[Learning] Store trajectory in vector DB for future similar tasks
    ↓
Result: Add product to cart
```

### Installation (5 min)

#### Prerequisites
- **Python 3.10+**
- **CUDA 11.8+** (for GPU, optional but recommended)
- **Git**, **Docker** (optional)

#### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/WOA-Agent.git
cd WOA-Agent

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Playwright browsers
playwright install chromium
```

#### Option 2: Docker Setup (Recommended)

```bash
# Build Docker image
docker build -t woa-agent:latest .

# Run container with GPU
docker-compose up -d

# Verify installation
docker-compose exec woa-agent python -c "import src; print('✓ Installation OK')"
```

### Usage Example

```python
import asyncio
from src.orchestrator.agent_orchestrator import AgentOrchestrator

async def main():
    # Initialize agent
    agent = AgentOrchestrator(
        max_steps=30,
        headless=False  # Show browser
    )
    
    # Execute task
    result = await agent.execute_task(
        query="Tìm áo khoác nam giá dưới 500k",
        start_url="https://shopee.vn"
    )
    
    # Check result
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"History: {result['history']}")

asyncio.run(main())
```

## 📁 Project Structure

```
WOA-Agent/
│
├── README.md (this file)
├── THEORY.md (architecture & research foundations)
├── SETUP.md (detailed setup & requirements)
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py (global configuration)
│   ├── models.yaml (model configurations)
│   ├── skills.yaml (skill definitions)
│   └── logging.yaml (logging setup)
│
├── src/
│   ├── __init__.py
│   │
│   ├── perception/ (Layer 1: Observation)
│   │   ├── __init__.py
│   │   ├── screenshot.py (capture & process)
│   │   ├── dom_distiller.py (flexible HTML simplification)
│   │   ├── ui_detector.py (element detection)
│   │   ├── embedding.py (PhoBERT encoder)
│   │   └── scene_representation.py (adaptive scene)
│   │
│   ├── planning/ (Layer 2: Decision Making)
│   │   ├── __init__.py
│   │   ├── planner_agent.py (high-level ViT5 planner)
│   │   ├── navigator_agent.py (browser executor)
│   │   ├── react_engine.py (Thought → Action)
│   │   ├── change_observer.py (DOM change tracking)
│   │   └── sub_agents/
│   │       ├── __init__.py
│   │       ├── login_agent.py (auth handling)
│   │       └── payment_agent.py (checkout)
│   │
│   ├── execution/ (Layer 3: Action Execution)
│   │   ├── __init__.py
│   │   ├── browser_manager.py (Playwright wrapper)
│   │   ├── skill_executor.py (skill orchestration)
│   │   └── skills/
│   │       ├── __init__.py
│   │       ├── base_skill.py (abstract base)
│   │       ├── navigation.py (goto, wait_for, reload)
│   │       ├── interaction.py (click, type, select)
│   │       ├── observation.py (screenshot, get_dom)
│   │       └── validation.py (assert conditions)
│   │
│   ├── learning/ (Layer 4: Experience Storage & Learning)
│   │   ├── __init__.py
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py (embedding storage with FAISS)
│   │   │   ├── trajectory_buffer.py (experience replay)
│   │   │   └── rail.py (Retrieval-Augmented IL)
│   │   ├── self_improvement.py (fine-tuning pipeline)
│   │   └── error_analyzer.py (error classification)
│   │
│   ├── models/ (Model Wrappers)
│   │   ├── __init__.py
│   │   ├── phobert_encoder.py (Vietnamese text encoder)
│   │   ├── vit5_planner.py (action generation)
│   │   └── lora_trainer.py (LoRA fine-tuning)
│   │
│   ├── orchestrator/ (System Control)
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py (main loop)
│   │   ├── state_manager.py (context tracking)
│   │   └── safety_guardrails.py (constraints)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py (logging)
│       ├── metrics.py (performance metrics)
│       ├── vietnamese_processor.py (text normalization)
│       └── validators.py (data validation)
│
├── data/
│   ├── raw/ (raw training data)
│   ├── processed/ (preprocessed data)
│   ├── embeddings/ (cached embeddings)
│   └── trajectories/ (collected trajectories)
│
├── checkpoints/
│   ├── phobert/ (PhoBERT checkpoints)
│   └── vit5/ (ViT5 checkpoints)
│
├── logs/
│   ├── agent_runs/ (execution logs)
│   └── errors/ (error logs)
│
├── tests/
│   ├── __init__.py
│   ├── unit/ (component tests)
│   │   ├── test_perception.py
│   │   ├── test_planning.py
│   │   ├── test_execution.py
│   │   └── test_learning.py
│   ├── integration/ (end-to-end tests)
│   │   ├── test_agent_flow.py
│   │   └── test_shopee_workflow.py
│   └── fixtures/
│       ├── mock_dom.html
│       └── mock_screenshots/
│
├── scripts/
│   ├── train_phobert.py (PhoBERT fine-tuning)
│   ├── train_vit5.py (ViT5 fine-tuning)
│   ├── collect_trajectories.py (data collection)
│   ├── evaluate_agent.py (benchmark evaluation)
│   ├── deploy.sh (deployment script)
│   └── setup_db.py (vector DB initialization)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_validation.ipynb
│   └── 03_agent_debugging.ipynb
│
├── docs/
│   ├── architecture.md (system design)
│   ├── api_reference.md (API docs)
│   ├── setup_guide.md (detailed setup)
│   └── troubleshooting.md (common issues)
│
├── docker-compose.yml (Docker Compose config)
├── Dockerfile (Docker image)
└── .dockerignore
```

## 🏗️ Architecture Overview

### 4-Layer Pipeline

```
┌─────────────────────────────────────────────────┐
│  User Query (Vietnamese): "Mua áo khoác"        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  LAYER 1: PERCEPTION                │ ← dom_distiller, embedding.py
│  Screenshot + DOM + UI Elements     │
│  PhoBERT embeddings (768-dim)       │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  LAYER 2: PLANNING                  │ ← planner_agent.py (ViT5)
│  ReAct: Thought → Action            │   react_engine.py
│  ViT5 generates: skill_name(params) │   change_observer.py
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  LAYER 3: EXECUTION                 │ ← browser_manager.py
│  Playwright skills:                 │   skills/*.py
│  click, type, scroll, wait_for      │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  LAYER 4: LEARNING                  │ ← vector_store.py
│  Store trajectory + embeddings      │   rail.py
│  LoRA fine-tuning                   │   lora_trainer.py
└─────────────────────────────────────┘
```

### Key Design Principles

1. **PhoBERT for Encoding Only** (NOT generation)
   - Extracts Vietnamese text embeddings (768-dim)
   - Semantic matching with UI elements
   - Vector database storage

2. **ViT5 for Action Generation**
   - Generates Vietnamese action sequences
   - LoRA fine-tuning for domain adaptation
   - ReAct reasoning with explanations

3. **Hierarchical Multi-Agent**
   - **Planner**: High-level task decomposition
   - **Navigator**: Low-level Playwright actions
   - **Sub-agents**: Specialized handlers (login, payment, etc.)

4. **Change Observation** (MutationObserver)
   - Tracks DOM changes after each action
   - Provides feedback for error detection
   - Enables adaptive planning

5. **RAIL Memory System**
   - Stores successful trajectories as vectors
   - Retrieves similar examples for few-shot learning
   - Enables continuous improvement

## 🚀 Quick Commands

### Development

```bash
# Create environment
make setup

# Run tests
make test          # All tests
make test-unit     # Unit tests only
make test-int      # Integration tests only

# Format code
make format        # Auto-format with Black
make lint          # Check with Pylint

# Logs
make logs          # Follow agent logs
make clear-logs    # Clear all logs
```

### Training & Evaluation

```bash
# Fine-tune ViT5 on Shopee domain
make train-vit5 DATA=data/shopee_trajectories.json

# Fine-tune PhoBERT for semantic matching
make train-phobert DATA=data/ui_elements.json

# Evaluate on benchmark
make evaluate BENCHMARK=webvoyager
```

### Deployment

```bash
# Docker build
make docker-build

# Docker run
make docker-run

# Docker clean
make docker-clean

# Deploy to cloud (example with Vercel)
make deploy-vercel
```

## 📊 Performance Targets

Based on Agent-E and WebVoyager benchmarks:

| Metric | Target | Current |
|--------|--------|---------|
| Task Success Rate | > 75% | - |
| Action Accuracy | > 90% | - |
| Execution Time | < 2 min | - |
| PhoBERT Embedding NDCG | > 0.85 | - |
| ViT5 Generation BLEU | > 50 | - |
| Error Recovery Rate | > 80% | - |

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Models
PHOBERT_MODEL=vinai/phobert-base-v2
VIT5_MODEL=VietAI/vit5-base

# Paths
CHECKPOINT_DIR=./checkpoints
DATA_DIR=./data
LOG_DIR=./logs

# Execution
MAX_STEPS=30
HEADLESS=false
TIMEOUT=30000  # ms

# GPU
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=4

# Database
VECTOR_DB_TYPE=faiss  # or chroma, pinecone
VECTOR_DB_PATH=./data/vector_store
```

See `.env.example` for full options.

## 📚 Documentation

- **[THEORY.md](THEORY.md)** - Research foundations, architecture details, design principles
- **[SETUP.md](SETUP.md)** - Installation, requirements, configuration, troubleshooting

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_perception.py -v

# Run with verbose output
pytest tests/ -vv

# Run and show print statements
pytest tests/ -s
```

## 📖 Example Use Cases

### 1. E-commerce Shopping
```python
task = "Mua áo khoác nam, màu đen, kích thước L, giá dưới 500k trên Shopee"
result = await agent.execute_task(task, "https://shopee.vn")
```

### 2. Price Comparison
```python
task = "So sánh giá áo khoác trên Shopee và Lazada"
result = await agent.execute_task(task, "https://shopee.vn")
```

### 3. Form Filling
```python
task = "Điền form đăng ký tài khoản ngân hàng"
result = await agent.execute_task(task, "https://bank.com")
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Create Pull Request

## ⚠️ Safety & Limitations

### Safety Measures
- ✅ Human-in-the-loop confirmation for sensitive actions
- ✅ Guardrails on sensitive website access (banking, health)
- ✅ Action validation before execution
- ✅ Error recovery and backtracking

### Known Limitations
- ⚠️ JavaScript-heavy websites may have limited support
- ⚠️ CAPTCHA/2FA requires manual intervention
- ⚠️ Some dynamic content may not be captured correctly

## 📝 License

MIT License - see LICENSE file for details

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/WOA-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/WOA-Agent/discussions)
- **Email**: your-email@example.com

## 🎓 References

This project synthesizes research from:
- WebVoyager (2024) - Multimodal perception, ReAct reasoning
- Agent-E (2024) - Hierarchical architecture, DOM distillation, change observation
- AgentOccam (2025) - Observation/action space alignment
- Invisible Multi-Agent - RAIL memory, adaptive scene representation
- OpenAI Operator - Safety, human-in-loop design

See [THEORY.md](THEORY.md) for detailed references.

## 🗺️ Roadmap

- [x] Project setup & structure
- [ ] Phase 1: Perception layer (Week 1)
- [ ] Phase 2: Planning layer with ViT5 (Week 2)
- [ ] Phase 3: Execution with Playwright (Week 3)
- [ ] Phase 4: Change observer & error handling (Week 4)
- [ ] Phase 5: Learning layer & RAIL (Week 5)
- [ ] Phase 6: Integration & deployment (Week 6)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/WOA-Agent&type=Date)](https://star-history.com/#yourusername/WOA-Agent&Date)

---

**Made with ❤️ for Vietnamese e-commerce automation**

Last Updated: November 15, 2025
