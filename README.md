# WebOrdering_Automation - Web Automation for Vietnamese E-commerce

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-brightgreen)](https://playwright.dev/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

**WebOrdering_Automation** is a production-ready **Web Automation Agent** designed specifically for Vietnamese e-commerce platforms (Shopee, Lazada). It combines state-of-the-art LLM-based agent research with Vietnamese language models to automate complex web tasks autonomously.

## 🎯 Project Goals and Purpose

The primary goal of this project is to create an intelligent automation system that can navigate and interact with Vietnamese e-commerce websites on behalf of users. The system aims to handle complex tasks such as searching for products, comparing prices, filling forms, and completing purchases with minimal human intervention. Key objectives include:


- **Enhanced User Experience**: Reduce the time and effort required to complete e-commerce tasks
- **Localization**: Support Vietnamese language queries and understand local e-commerce patterns
- **Reliability**: Provide robust automation that can handle various edge cases and website changes
- **Scalability**: Design the system to work across multiple e-commerce platforms with minimal modifications
- **Learning Capability**: Improve performance over time through experience and feedback

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
- **Python 3.10+** (Đã có sẵn tại: `F:\WebOrdering_Automation\woa\python.exe`)
- **CUDA 11.8+** (cho GPU, tùy chọn nhưng khuyến nghị)
- **Git**

### Installation

#### Prerequisites
- **Python 3.10+** (Đã có sẵn tại: `F:\WebOrdering_Automation\woa\python.exe`)
- **CUDA 11.8+** (cho GPU, tùy chọn nhưng khuyến nghị)
- **Git**

#### Local Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/WebOrdering_Automation.git
cd WebOrdering_Automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium

# Download Vietnamese models
python scripts/training/download_models.py

# Verify installation
python -c "import src; print('✓ Installation OK')"
```

#### Using Makefile (Simpler)

```bash
# Install all dependencies
make install

# Download models
make download-models

# Run the agent
make run
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
WebOrdering_Automation/
│
├── README.md (this file)
├── CODEBASE-OVERVIEW.md (detailed architecture overview)
├── COMPLETE-DOCUMENTATION.md (comprehensive project documentation)
├── LICENSE
├── PLAN.md (project development plan)
├── SETUP.md (detailed setup guide)
├── RUN-INSTRUCTIONS.md (instructions for running the system)
├── THEOREY.md (research foundations and methodology)
├── pyproject.toml (project configuration)
├── requirements.txt (Python dependencies)
├── run_agent.py (main execution script)
├── .env.example (environment variables template)
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── data_catalog.yaml (data source definitions)
│   ├── logging.yaml (logging configuration)
│   ├── models.yaml (model configurations)
│   ├── selectors.yaml (DOM selectors configuration)
│   ├── settings.py (application settings)
│   └── skills.yaml (skill definitions)
│
├── src/
│   ├── __init__.py
│   ├── execution/ (Layer 3: Action Execution)
│   │   ├── __init__.py
│   │   ├── browser_manager.py (Playwright wrapper)
│   │   ├── omni_passer.py (communication between layers)
│   │   ├── skill_executor.py (skill orchestration)
│   │   ├── skills_executor.py (alternative skill executor)
│   │   └── skills/ (individual action implementations)
│   │       ├── __init__.py
│   │       ├── base_skill.py (abstract base skill)
│   │       ├── interaction.py (click, type, select)
│   │       ├── navigation.py (goto, wait_for, reload)
│   │       ├── observation.py (screenshot, get_dom)
│   │       ├── validation.py (assert conditions)
│   │       └── wait.py (waiting utilities)
│   │
│   ├── learning/ (Layer 4: Experience Storage & Learning)
│   │   ├── __init__.py
│   │   ├── error_analyzer.py (error classification)
│   │   ├── memory/ (memory and learning components)
│   │   │   ├── __init__.py
│   │   │   ├── rail.py (Retrieval-Augmented IL)
│   │   │   ├── trajectory_buffer.py (experience replay)
│   │   │   └── vector_store.py (embedding storage with FAISS)
│   │   ├── self_improvement.py (fine-tuning pipeline)
│   │   └── README.md (learning module documentation)
│   │
│   ├── orchestrator/ (System Control)
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py (main agent controller)
│   │   ├── safety_guardrails.py (safety constraints)
│   │   └── state_manager.py (context tracking)
│   │
│   ├── perception/ (Layer 1: Observation)
│   │   ├── __init__.py
│   │   ├── dom_distiller.py (HTML simplification)
│   │   ├── embedding.py (PhoBERT encoder)
│   │   ├── scene_representation.py (adaptive scene)
│   │   ├── screenshot.py (capture & process)
│   │   ├── ui_detector.py (element detection)
│   │   └── vision_enhancer.py (visual enhancement)
│   │
│   ├── planning/ (Layer 2: Decision Making)
│   │   ├── __init__.py
│   │   ├── change_observer.py (DOM change tracking)
│   │   ├── navigator_agent.py (browser executor)
│   │   ├── planner_agent.py (high-level ViT5 planner)
│   │   ├── react_engine.py (Thought → Action)
│   │   ├── rule_policy.py (rule-based policy)
│   │   └── sub_agents/ (specialized agents)
│   │       ├── __init__.py
│   │       ├── base_agent.py (abstract base agent)
│   │       ├── form_agent.py (form handling)
│   │       ├── login_agent.py (authentication handling)
│   │       ├── payment_agent.py (checkout handling)
│   │       ├── search_agent.py (search handling)
│   │       └── README.md (sub-agents documentation)
│   │
│   └── utils/ (utility functions)
│       ├── __init__.py
│       ├── logger.py (logging utilities)
│       ├── metrics.py (performance metrics)
│       ├── validators.py (data validation)
│       └── vietnamese_processor.py (Vietnamese text processing)
│
├── data/
│   ├── raw/ (raw training data)
│   ├── processed/ (preprocessed data)
│   ├── embeddings/ (cached embeddings)
│   └── trajectories/ (collected trajectories)
│
├── docs/
│   ├── api_reference.md (API documentation)
│   ├── architecture.md (system architecture)
│   ├── CHROME_PROFILE_GUIDE.md (Chrome profile setup guide)
│   ├── README.md (documentation overview)
│   ├── setup_guide.md (installation guide)
│   ├── TEST_EXECUTION_GUIDE.md (testing guide)
│   └── troubleshooting.md (troubleshooting guide)
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py (metric helpers)
│   ├── baselines/ (baseline agent implementations)
│   │   ├── __init__.py
│   │   ├── gemini_agent.py (Gemini baseline)
│   │   ├── gpt4_agent.py (GPT-4 baseline)
│   │   ├── rule_based_agent.py (Rule-based baseline)
│   │   └── README.md (baselines documentation)
│   ├── benchmarks/ (benchmark tasks)
│   │   └── README.md (benchmark documentation)
│   └── results/ (evaluation results)
│       └── README.md (results documentation)
│
├── experiments/
│   ├── README.md (experiment overview)
│   ├── exp_001_baseline_gemini_teacher/ (experiment 1)
│   ├── exp_002_ablation_no_thought/ (experiment 2)
│   └── exp_003_ablation_no_gemini/ (experiment 3)
│
├── notebooks/
│   ├── README.md (notebook overview)
│   ├── 01_data_collection/ (data collection notebooks)
│   ├── 02_annotation/ (annotation notebooks)
│   ├── 03_preprocessing/ (preprocessing notebooks)
│   ├── 04_training/ (training notebooks)
│   └── 05_evaluation/ (evaluation notebooks)
│
├── paper/
│   └── README.md (paper documentation)
│
├── scripts/
│   ├── __init__.py
│   ├── deploy.sh (deployment script)
│   ├── annotation/ (annotation scripts)
│   │   ├── __init__.py
│   │   ├── batch_annotate.py (batch annotation)
│   │   ├── gemini_annotator.py (Gemini annotation)
│   │   ├── quality_control.py (quality control)
│   │   ├── validate_annotations.py (annotation validation)
│   │   └── prompts/ (annotation prompts)
│   ├── data_collection/ (data collection scripts)
│   │   ├── __init__.py
│   │   ├── collect_raw_trajectories.py (collect trajectories)
│   │   ├── README.md (data collection documentation)
│   │   ├── validate_raw.py (validate raw data)
│   │   └── tasks/ (task definitions)
│   ├── evaluation/ (evaluation scripts)
│   │   ├── __init__.py
│   │   ├── compute_metrics.py (compute metrics)
│   │   ├── error_analysis.py (analyze errors)
│   │   ├── README.md (evaluation documentation)
│   │   ├── run_ablation.py (run ablation studies)
│   │   └── run_benchmark.py (run benchmarks)
│   ├── paper/ (paper generation scripts)
│   │   ├── __init__.py
│   │   ├── export_results.py (export results)
│   │   ├── generate_figures.py (generate figures)
│   │   ├── generate_tables.py (generate tables)
│   │   └── README.md (paper scripts documentation)
│   ├── preprocessing/ (preprocessing scripts)
│   │   ├── __init__.py
│   │   ├── build_controller_dataset.py (build controller data)
│   │   ├── build_embeddings.py (build embeddings)
│   │   ├── build_planner_dataset.py (build planner data)
│   │   ├── compute_statistics.py (compute statistics)
│   │   ├── README.md (preprocessing documentation)
│   │   └── split_dataset.py (split dataset)
│   └── training/ (training scripts)
│       ├── __init__.py
│       ├── download_models.py (download models)
│       ├── evaluate_model.py (evaluate models)
│       ├── README.md (training documentation)
│       ├── train_controller.py (train controller)
│       └── train_planner.py (train planner)
│
├── tests/
│   ├── __init__.py
│   ├── path_setup.py (test path setup)
│   ├── README.md (test documentation)
│   ├── full_pipeline_test.py (full pipeline test)
│   ├── test_browser_with_settings.py (browser settings test)
│   ├── test_chrome_profile.py (Chrome profile test)
│   ├── test_execution_quick.py (quick execution test)
│   ├── test_execution_stepbystep.py (step-by-step execution test)
│   ├── test_models.py (model tests)
│   ├── fixtures/ (test fixtures)
│   │   ├── __init__.py
│   │   ├── mock_dom.html (mock DOM for testing)
│   │   └── README.md (fixtures documentation)
│   ├── integration/ (integration tests)
│   │   ├── __init__.py
│   │   ├── test_agent_flow.py (agent flow test)
│   │   ├── test_shopee_workflow.py (Shopee workflow test)
│   │   └── README.md (integration tests documentation)
│   └── performance/ (performance tests)
│       ├── __init__.py
│       └── README.md (performance tests documentation)
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
- **[RUN-INSTRUCTIONS.md](RUN-INSTRUCTIONS.md)** - Detailed instructions for running the system
- **[CODEBASE-OVERVIEW.md](CODEBASE-OVERVIEW.md)** - Comprehensive overview of the codebase structure
- **[COMPLETE-DOCUMENTATION.md](COMPLETE-DOCUMENTATION.md)** - Complete project documentation
- **[docs/](docs/)** - Additional documentation including API reference, architecture, setup guide, and troubleshooting

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

- **Issues**: [GitHub Issues](https://github.com/teswy/WOA-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/teswy/WOA-Agent/discussions)
- **Email**: huy40580@gmail.com

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

[![Star History Chart](https://api.star-history.com/svg?repos=teswy/WOA-Agent&type=Date)](https://star-history.com/#teswy/WOA-Agent&Date)

---

**Made with ❤️ for Vietnamese e-commerce automation**

Last Updated: November 15, 2025
