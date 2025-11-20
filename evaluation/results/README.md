# Evaluation Results

This directory stores outputs from evaluation runs of the WebOrdering_Automation system. Results are organized by timestamp and experiment configuration to track performance over time and across different system variants.

## Contents

- `run_20251119_120000_baseline/` - Results from baseline experiment run
- `run_20251119_123000_vit5_gemini_labeled/` - Results from ViT-5 + Gemini labeled experiment run

## Overview

The results directory contains structured outputs from evaluation runs, including metrics, trajectories, logs, and other artifacts needed for analysis. Each run is stored in a timestamped directory that corresponds to a specific experiment configuration.

## Directory Structure

Each evaluation run directory contains:
- Configuration files used for the run
- Computed metrics and evaluation results
- Trajectory files showing the agent's actions and observations
- Log files with detailed execution information
- Any additional artifacts generated during evaluation

## Organization

Results are organized by timestamp to ensure each run has a unique directory. The naming convention is `run_YYYYMMDD_HHMMSS_description/` where the description indicates the specific configuration or experiment being run.