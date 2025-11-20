# Evaluation

This directory contains all components related to evaluating the WebOrdering_Automation system. It includes baseline implementations, benchmark definitions, evaluation scripts, and results storage. The evaluation framework is designed to systematically assess the performance of the automation system against various baselines and benchmarks using standardized metrics and evaluation procedures.

## Contents

- `__init__.py` - Python package initialization
- `metrics.py` - Definition of evaluation metrics and scoring functions
- `baselines/` - Baseline agent implementations for comparison
- `benchmarks/` - Benchmark definitions and test cases
- `results/` - Storage for evaluation results and experiment outputs

## Overview

The evaluation framework is designed to systematically assess the performance of the automation system against various baselines and benchmarks. It provides standardized metrics and evaluation procedures to ensure consistent and fair comparisons. The framework supports both quantitative metrics and qualitative analysis of system behavior and performance across different e-commerce platforms and task types.

## Metrics

The evaluation system uses several key metrics to assess system performance:
- **Task Success Rate**: Percentage of tasks completed successfully
- **Action Accuracy**: Accuracy of individual actions executed by the system
- **Execution Time**: Time taken to complete tasks
- **Error Recovery Rate**: Ability of the system to recover from errors
- **User Satisfaction**: Subjective measure of how well the system meets user expectations

For detailed information about each metric and how they are calculated, see `metrics.py` and the documentation in `docs/api_reference.md`.

## Baselines

The `baselines/` directory contains implementations of various baseline approaches for comparison with the main system implementation:
- **Gemini Agent** (`baselines/gemini_agent.py`): Uses Google's Gemini model for decision making
- **GPT-4 Agent** (`baselines/gpt4_agent.py`): Uses OpenAI's GPT-4 model for decision making
- **Rule-based Agent** (`baselines/rule_based_agent.py`): Implements heuristic-based approaches for comparison

These baselines provide reference points for measuring the effectiveness of the main system implementation and help identify areas for improvement.

## Benchmarks

The `benchmarks/` directory contains standardized test cases that evaluate different aspects of the automation system's capabilities. These benchmarks cover various e-commerce scenarios including product search, price comparison, form filling, checkout processes, and account management. Each benchmark includes detailed task descriptions, success criteria, and evaluation procedures to ensure consistent and fair assessment of system performance across different scenarios and platforms (Shopee, Lazada, etc.).

## Results

The `results/` directory stores outputs from evaluation runs, organized by timestamp and experiment configuration. Each evaluation run creates a timestamped subdirectory containing detailed logs, performance metrics, execution traces, and other relevant data for analysis. This organization allows for easy comparison of different system configurations and evaluation runs over time. The results can be analyzed using the notebooks in `notebooks/05_evaluation/` or using the analysis scripts in `scripts/evaluation/`.

## Running Evaluations

To run evaluations, use the scripts in `scripts/evaluation/` directory. The main evaluation script is `run_benchmark.py`, which can be configured to run specific benchmarks against different agent implementations and configurations. Results are automatically stored in the `results/` directory with appropriate metadata for later analysis and comparison.