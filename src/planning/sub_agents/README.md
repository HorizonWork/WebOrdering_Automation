# Planning Sub-Agents

This directory contains specialized planning sub-agents for the WebOrdering_Automation system. These agents handle specific aspects of the planning process, such as form handling, login procedures, payment processing, and search functionality.

## Contents

- `__init__.py` - Python package initialization
- `base_agent.py` - Base class for all sub-agents
- `form_agent.py` - Agent for handling form filling and validation
- `login_agent.py` - Agent for handling login and authentication
- `payment_agent.py` - Agent for handling payment processing
- `search_agent.py` - Agent for handling product search and filtering

## Overview

The sub-agents module implements specialized planning agents that handle specific types of tasks within the web ordering workflow. Each agent is designed to handle a particular aspect of the ordering process with expertise in that domain.

## Agent Categories

- **Form Agent**: Handles form filling, validation, and submission with proper error handling
- **Login Agent**: Manages login procedures, account access, and authentication flows
- **Payment Agent**: Handles payment processing, including form validation and security checks
- **Search Agent**: Manages product search, filtering, and result navigation

## Architecture

Each sub-agent inherits from a common base agent class that provides consistent interfaces and error handling. This allows for uniform integration with the main planning module while maintaining specialized functionality for each task type.