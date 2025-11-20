# Experiment 01: Baseline with Gemini Teacher

This directory contains the configuration and results for the first baseline experiment using a Gemini-based teacher agent. This experiment serves as a foundational baseline for the WebOrdering_Automation system, establishing performance metrics against which other approaches can be compared. The experiment evaluates the effectiveness of using Google's Gemini model as the primary decision-making component in a teacher-forcing approach to web automation tasks on Vietnamese e-commerce platforms (Shopee, Lazada).

## Contents

- `analysis.md` - Analysis of experiment results, including performance metrics, error patterns, and comparison with other approaches
- `config.yaml` - Configuration settings for this experiment, including model parameters, evaluation settings, and execution parameters
- `results/` - Detailed results from the experiment run, organized by timestamp and task type

## Overview

This experiment represents a baseline approach where Google's Gemini model serves as the primary decision-making component in a teacher-forcing configuration. The approach uses the Gemini model to generate action sequences that guide the automation process, with the expectation that the model will provide high-quality, contextually appropriate actions for each step of the web ordering tasks. This baseline provides a reference point for evaluating the effectiveness of the hierarchical agent architecture implemented in the main system, particularly in the Vietnamese e-commerce context where language and cultural factors may affect performance. The experiment tests various task types including product search, filtering, and purchase completion across different e-commerce platforms to provide a comprehensive baseline assessment.

## Configuration

The experiment configuration in `config.yaml` includes parameters for model selection, prompting strategies, and evaluation criteria. Key configuration elements include:
- Model selection parameters for the Gemini model
- Prompt engineering strategies for web automation tasks in Vietnamese
- Evaluation metrics and success criteria for different task types
- Execution parameters such as maximum steps, timeout values, and retry mechanisms
- Platform-specific settings for Shopee and Lazada automation

The configuration is designed to provide consistent and fair evaluation conditions while allowing for comparison with other approaches implemented in subsequent experiments. Configuration parameters can be modified to test different settings and approaches while maintaining experimental consistency for comparison purposes.

## Results

The results directory contains detailed outputs from the experiment, including success rates, execution metrics, error analysis, and qualitative assessments of system behavior. Results are organized by task type, e-commerce platform, and execution date to enable detailed analysis and comparison. Key metrics include task success rate, action accuracy, execution time, and error recovery effectiveness. The results provide a comprehensive baseline for comparing more sophisticated approaches, including the hierarchical agent architecture implemented in the main system. Analysis of error patterns helps identify limitations of the baseline approach and guides improvements in subsequent system iterations. The results are also used to validate the evaluation framework and ensure consistent assessment across all experiments in the project. Each result entry includes execution logs, performance metrics, and detailed execution traces to enable thorough analysis and reproducibility of findings.