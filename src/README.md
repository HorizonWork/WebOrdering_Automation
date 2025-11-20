# Source Code

This directory contains the core source code for the WebOrdering_Automation system. The code is organized into modules that represent different functional components of the system, following a 4-layer hierarchical architecture designed for web automation tasks. Each module has a specific role in processing user requests and executing web interactions autonomously.

## Contents

- `__init__.py` - Python package initialization
- `execution/` - Components for executing actions in web environments (Layer 3)
- `learning/` - Components for learning and self-improvement (Layer 4)
- `orchestrator/` - Components for agent orchestration and state management
- `perception/` - Components for processing visual and DOM information (Layer 1)
- `planning/` - Components for decision making and planning (Layer 2)
- `utils/` - Utility functions and common components

## Architecture Overview

The system implements a 4-layer hierarchical architecture that processes user queries through distinct functional layers. Each layer has a specific responsibility in the automation pipeline, allowing for modular development and clear separation of concerns.

### Layer 1: Perception (`perception/`)
- **Purpose**: Extract relevant information from the web environment
- **Components**: DOM distillation, screenshot capture, UI element detection, embedding generation
- **Key files**: `dom_distiller.py`, `ui_detector.py`, `embedding.py`, `screenshot.py`

### Layer 2: Planning (`planning/`)
- **Purpose**: Generate action sequences based on current state and goals
- **Components**: Decision-making algorithms, ReAct reasoning, specialized sub-agents
- **Key files**: `planner_agent.py`, `react_engine.py`, `sub_agents/`

### Layer 3: Execution (`execution/`)
- **Purpose**: Execute browser actions and interact with web elements
- **Components**: Browser management, skill execution, action validation
- **Key files**: `browser_manager.py`, `skill_executor.py`, `skills/`

### Layer 4: Learning (`learning/`)
- **Purpose**: Store experiences, improve performance, and handle errors
- **Components**: Memory systems, error analysis, self-improvement mechanisms
- **Key files**: `memory/`, `error_analyzer.py`, `self_improvement.py`

## Key Modules

- **execution**: Handles browser automation, skill execution, and interaction with web elements. This module translates high-level actions into specific browser commands using Playwright.

- **learning**: Implements memory systems (RAIL - Retrieval-Augmented Imitation Learning), self-improvement mechanisms, and error analysis to enable the system to learn from experience and improve over time.

- **orchestrator**: Manages agent coordination, state tracking, and safety guardrails. This module coordinates the flow between different layers and ensures safe execution of tasks.

- **perception**: Processes screenshots, DOM elements, and visual information to create a structured representation of the current web environment that can be understood by the planning layer.

- **planning**: Implements decision-making algorithms, navigation strategies, and task planning. Uses ReAct (Reasoning + Acting) methodology to generate action sequences based on current state and goals.

- **utils**: Provides logging, metrics, validation, and other utility functions that are used across different modules of the system.

## Architecture

The system follows an agent-based architecture where different specialized agents work together to complete web ordering tasks. The orchestrator coordinates these agents, while the perception module provides environmental awareness, the planning module makes decisions, and the execution module carries out actions. The learning module continuously improves the system's performance based on experience. This architecture allows for modularity, maintainability, and scalability of the automation system.