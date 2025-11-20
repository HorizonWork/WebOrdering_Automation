# Experiment Log

This directory contains all experiment configurations, results, and analysis for the WebOrdering_Automation project. Each experiment is designed to test specific hypotheses about the system's performance and capabilities, and to compare different approaches to web automation tasks.

## Experiments Overview

- **exp_001_baseline_gemini_teacher**: Baseline experiment using Gemini as a teacher model to provide guidance for the automation system. This experiment establishes a performance baseline using a large language model as the primary decision-making component.

- **exp_002_ablation_no_thought**: Ablation study to evaluate the importance of planning and reasoning components. This experiment disables the planner's thought process to determine how much the reasoning layer contributes to overall performance.

- **exp_003_ablation_no_gemini**: Ablation study to evaluate the system's performance without Gemini model assistance. This experiment trains the system using only human demonstrations to understand the effectiveness of different training approaches.

## Experiment Structure

Each experiment directory contains the following components:
- `config.yaml`: Configuration settings for the experiment
- `analysis.md`: Analysis of experiment results
- `results/`: Directory containing detailed results and metrics

## Running Experiments

To run experiments, use the appropriate scripts in the `scripts/evaluation/` directory. Results will be saved in the corresponding experiment's results directory with timestamps for easy tracking and comparison.

## Contributing New Experiments

When adding new experiments, please follow this structure and update this README with a brief description of the new experiment's purpose and expected outcomes.
