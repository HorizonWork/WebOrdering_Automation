# Data Collection Scripts

This directory contains executable scripts for collecting raw trajectory data for the WebOrdering_Automation project. These scripts implement the data collection pipeline that gathers examples of web ordering tasks on Vietnamese e-commerce platforms like Shopee and Lazada. The collected trajectories serve as the foundation for training and evaluation of the automation system, providing examples of successful task completion that can be used to train the planning and execution components of the system. The data collection process captures comprehensive information about user interactions, including DOM states, screenshots, action sequences, and timing information.


## Contents

- `__init__.py` - Python package initialization
- `collect_raw_trajectories.py` - Script for collecting raw trajectories from web ordering tasks
- `validate_raw.py` - Script for validating raw trajectory data for completeness and quality
- `tasks/` - Directory containing task definitions for data collection
  - `lazada_tasks.yaml` - Task definitions specific to Lazada platform
  - `shopee_tasks.yaml` - Task definitions specific to Shopee platform

## Overview

The data collection scripts implement the process of gathering raw trajectories from web ordering tasks on Vietnamese e-commerce platforms. These trajectories include comprehensive information about the interaction process, such as DOM snapshots, screenshots, action sequences, timing information, and success/failure outcomes. The collected data is essential for training the system's planning and execution components, as well as for establishing baseline performance metrics. The data collection process is designed to capture real-world usage patterns and challenges specific to Vietnamese e-commerce environments, including language-specific elements, platform-specific UI patterns, and common user workflows. The scripts are designed to work with the existing browser management infrastructure in the execution module to ensure consistent and reliable data collection.


## Process

The scripts in this directory implement a comprehensive data collection workflow that includes multiple stages:

1. **Task Execution**: Execute predefined tasks on e-commerce platforms while capturing comprehensive state information at each step
2. **Data Capture**: Record DOM snapshots, screenshots, action sequences, and timing information for each interaction
3. **Quality Control**: Validate collected data for completeness and quality, flagging any incomplete or problematic trajectories
4. **Storage**: Store trajectories in a structured format optimized for subsequent processing and training

The data collection process includes mechanisms for handling common challenges such as dynamic content loading, session management, and error recovery to ensure high-quality data collection. The scripts also include logging and monitoring capabilities to track collection progress and identify potential issues during the collection process.


## Usage

Run the scripts from the command line. For example:
- `python -m scripts.data_collection.collect_raw_trajectories` to collect trajectories
- `python -m scripts.data_collection.validate_raw` to validate collected data

The data collection script accepts various command-line arguments to customize the collection process, including platform selection (Shopee/Lazada), task selection from the task definition files, output directory specification, and browser configuration options. The collection process can be run in headless mode for automated collection or with visible browser windows for monitoring and debugging. The scripts also support resuming interrupted collection sessions and include checkpointing mechanisms to avoid losing collected data in case of interruptions.


## Dependencies

These scripts require the project dependencies to be installed as specified in `requirements.txt`. They also require access to specific websites for data collection and may need specific browser configurations or Chrome profiles to work correctly. The scripts depend on the execution module's browser management components and require Playwright to be properly installed with the required browser binaries. For optimal results, ensure that network connectivity is stable during data collection, as interruptions may result in incomplete trajectory data. The scripts may also require appropriate credentials or accounts on the target e-commerce platforms, depending on the specific tasks being collected. For privacy and legal compliance, ensure that data collection follows the terms of service of the target platforms and applicable privacy regulations.


## Task Definitions

The `tasks/` directory contains YAML files that define the specific tasks to be collected. Each task definition includes the starting URL, the goal description (in Vietnamese), success criteria, and any special handling instructions. The task definitions are designed to cover a representative sample of common e-commerce activities on Vietnamese platforms, including product search, filtering, cart management, and checkout processes. New task definitions can be added to expand the coverage of collected behaviors and improve the training data diversity.