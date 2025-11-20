# Execution Module

This directory contains components for executing actions in web environments as part of the WebOrdering_Automation system. The execution module serves as Layer 3 in the 4-layer hierarchical architecture, handling browser automation, skill execution, and interaction with web elements. It translates high-level decisions from the planning module into specific interactions with web elements, manages browser sessions, and executes skill-based actions with proper error handling and validation. The execution module is critical for the system's ability to interact with Vietnamese e-commerce platforms like Shopee and Lazada in a reliable and robust manner.


## Contents

- `__init__.py` - Python package initialization
- `browser_manager.py` - Component for managing browser instances and sessions using Playwright
- `omni_passer.py` - Component for passing information between execution steps and other layers
- `skill_executor.py` - Component for executing individual skills with error handling and validation
- `skills_executor.py` - Component for executing sequences of skills with coordination and flow control
- `skills/` - Submodule containing individual skill implementations for various web interactions

## Overview

The execution module is responsible for carrying out the actions determined by the planning module. It serves as the interface between the high-level planning and the low-level browser interactions. The module handles all direct interactions with web pages, including clicking elements, typing text, scrolling, waiting for elements to appear, and validating the results of actions. It also manages browser state and sessions, ensuring that interactions are performed correctly within the context of the current page and session. The execution module includes comprehensive error handling and recovery mechanisms to deal with common web automation challenges such as dynamic content loading, element availability issues, and network timeouts. The module is designed to work specifically with Vietnamese e-commerce platforms, taking into account their unique UI patterns, dynamic content, and localization requirements.


## Components

### Browser Manager (`browser_manager.py`)
Handles browser instance creation, configuration, and session management using Playwright. This component manages browser lifecycle, tab management, and session persistence. It includes configuration for headless vs. visible operation, viewport settings, and user agent strings appropriate for Vietnamese e-commerce platforms. The browser manager also handles Chrome profile management and configuration for consistent automation behavior across runs.


### Skill Executor (`skill_executor.py`)
Executes individual skills with proper error handling, validation, and logging. This component validates action parameters before execution and handles exceptions that may occur during skill execution. It provides feedback to the planning module about the success or failure of executed actions, enabling the system to adapt its plans based on execution results. The skill executor also includes retry mechanisms for handling transient failures and timeout handling for operations that take too long.


### Omni Passer (`omni_passer.py`)
Facilitates information passing between execution steps and communication with other layers of the system. This component manages the flow of data between the execution layer and the planning, perception, and learning layers. It ensures that execution results, error conditions, and state changes are properly communicated to other system components for coordinated operation. The omni passer helps maintain consistency across the 4-layer architecture.


### Skills (`skills/`)
Contains specific implementations for different types of web interactions. Each skill is designed to perform a specific action type such as clicking, typing, selecting options, waiting, or validating conditions. The skills module includes specialized skills for e-commerce interactions like product selection, cart management, and checkout processes. Each skill is designed to be reusable and composable to build complex interaction sequences.


## Architecture

The execution module follows a skill-based architecture where complex actions are broken down into smaller, reusable skills. This allows for better error handling, validation, and reusability of common web interaction patterns. The architecture separates the concerns of browser management, skill execution, and inter-layer communication, making the module more maintainable and extensible. The skill-based approach enables the system to handle various types of web interactions while providing consistent error handling and logging across all actions. The module is designed to work in conjunction with the change observer in the planning layer to detect and respond to DOM changes that occur as a result of executed actions. This integration ensures that the system has accurate information about the current state of the web page after each action is executed. The execution module also includes safety mechanisms to prevent potentially harmful actions and includes human-in-the-loop capabilities for sensitive operations like payment processing.


## Skills Overview

The skills submodule contains the following specialized skill types:
- **Navigation skills**: Handle page navigation, URL loading, and page reloading
- **Interaction skills**: Handle clicking, typing, selecting, and other element interactions
- **Observation skills**: Handle screenshot capture and DOM state retrieval
- **Validation skills**: Handle condition checking and assertion validation
- **Wait skills**: Handle waiting for elements, conditions, or time-based delays