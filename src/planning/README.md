# Planning Module

This directory contains components for decision making and planning in the WebOrdering_Automation system. The planning module implements navigation, task planning, and decision-making algorithms that determine the sequence of actions to complete web ordering tasks.

## Contents

- `__init__.py` - Python package initialization
- `change_observer.py` - Component for observing and tracking changes in the environment
- `navigator_agent.py` - Component for navigation-specific planning
- `planner_agent.py` - Main planning agent component
- `react_engine.py` - Component implementing ReAct (Reasoning and Acting) engine
- `rule_policy.py` - Component for rule-based decision making
- `sub_agents/` - Submodule containing specialized planning sub-agents

## Overview

The planning module is responsible for high-level decision making and task planning. It determines what actions should be taken based on the current state and goal, coordinating with the perception module for environmental awareness and the execution module for action implementation.

## Components

- **Planner Agent**: Main component for overall task planning and decision making
- **Navigator Agent**: Specialized component for navigation-related planning
- **ReAct Engine**: Implements reasoning and acting cycles for decision making
- **Rule Policy**: Provides rule-based decision making capabilities
- **Change Observer**: Tracks and responds to changes in the environment
- **Sub Agents**: Specialized agents for specific planning tasks

## Architecture

The planning module uses a hierarchical approach with specialized agents for different aspects of planning, coordinated by the main planner agent. The ReAct engine enables reasoning about the current situation before taking actions.