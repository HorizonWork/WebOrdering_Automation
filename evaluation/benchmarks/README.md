# Benchmarks

This directory contains benchmark definitions and test cases used to evaluate the WebOrdering_Automation system. These benchmarks provide standardized tasks to assess different aspects of the system's capabilities, particularly in the context of Vietnamese e-commerce platforms like Shopee and Lazada. The benchmarks are designed to test specific capabilities of the web ordering automation system, with each benchmark representing a standardized task or scenario that can be used to evaluate performance consistently across different system configurations or implementations. The benchmarks follow a structured format that includes task descriptions, success criteria, and evaluation metrics to ensure reproducible and fair comparisons between different approaches and system versions.


## Contents

- `lazada_tasks.json` - Benchmark tasks specifically designed for Lazada platform testing
- `shopee_tasks.json` - Benchmark tasks specifically designed for Shopee platform testing
- `common_tasks.json` - Benchmark tasks that apply to multiple e-commerce platforms
- `schema.json` - Schema definition for benchmark task format
- `task_definitions.md` - Detailed documentation of benchmark task categories and formats

## Overview

The benchmarks are organized by e-commerce platform and task complexity to provide comprehensive coverage of the automation system's capabilities. Each benchmark includes detailed task descriptions in Vietnamese to ensure proper evaluation of the system's ability to understand and execute Vietnamese-language instructions. The benchmarks are designed to test the full 4-layer architecture of the system, from perception through learning, ensuring that each component functions correctly both individually and as part of the integrated system. The benchmarks also include edge cases and error scenarios to evaluate the system's robustness and error recovery capabilities in real-world conditions.


## Benchmark Categories

Benchmarks are organized into several categories that cover various aspects of web ordering tasks on Vietnamese e-commerce platforms:

### Basic Navigation Tasks
- Page navigation and URL handling
- Basic element identification and interaction
- Simple search operations
- Category browsing and filtering

### Form Interaction Tasks
- Login and account management
- Product search and filtering forms
- Shipping information forms
- Payment information entry

### Complex Shopping Tasks
- Multi-step purchase workflows
- Product comparison across pages
- Cart management and checkout
- Order tracking and management

### Error Handling Scenarios
- CAPTCHA and security challenges
- Network timeout recovery
- Invalid input handling
- Dynamic content loading issues

### Platform-Specific Tasks
- Shopee-specific features (coins, vouchers, promotions)
- Lazada-specific workflows (shipping options, payment methods)
- Platform-specific UI elements and interactions

## Usage

The benchmarks are used by the evaluation framework to systematically test the system's capabilities. Each benchmark should include clear success criteria and evaluation metrics. To run benchmarks, use the scripts in `scripts/evaluation/` directory, particularly `run_benchmark.py` which can be configured to run specific benchmark files against different system configurations. Results are automatically stored in the `evaluation/results/` directory with appropriate metadata for later analysis and comparison. The benchmark format follows the schema defined in `schema.json` and includes fields for task description, starting URL, success conditions, and evaluation metrics. When creating new benchmarks, follow the existing format and ensure that tasks are representative of real user scenarios on Vietnamese e-commerce platforms.