# Evaluation Scripts

This directory contains executable scripts for evaluating the WebOrdering_Automation system. These scripts run the evaluation framework, compute metrics, and analyze results.

## Contents

- `__init__.py` - Python package initialization
- `compute_metrics.py` - Script for computing evaluation metrics
- `error_analysis.py` - Script for analyzing errors and failures
- `run_ablation.py` - Script for running ablation studies
- `run_benchmark.py` - Script for running benchmark evaluations

## Overview

The evaluation scripts implement the process of assessing the trained models against various benchmarks and baselines. They provide comprehensive analysis of model performance, including success rates, error patterns, and comparative analysis.

## Process

The scripts in this directory:
1. Evaluate trained models on benchmark tasks
2. Compute various performance metrics
3. Conduct ablation studies to understand component contributions
4. Analyze error patterns and failure cases
5. Generate evaluation reports and visualizations

## Usage

Run the scripts from the command line. For example:
- `python -m scripts.evaluation.run_benchmark` to run benchmark evaluations
- `python -m scripts.evaluation.run_ablation` to run ablation studies
- `python -m scripts.evaluation.error_analysis` to analyze errors

## Dependencies

These scripts require the project dependencies to be installed and access to trained models and evaluation datasets.