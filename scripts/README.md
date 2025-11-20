# Scripts

This directory contains executable scripts for various tasks in the WebOrdering_Automation project. These scripts automate common operations like data collection, annotation, preprocessing, training, evaluation, and paper generation. The scripts are organized by function to support the complete workflow of the WebOrdering_Automation system, from initial data collection through model training to final evaluation and result analysis.

## Contents

- `deploy.sh` - Deployment script for setting up the system in production environments
- `annotation/` - Scripts for data annotation and quality control
- `data_collection/` - Scripts for collecting raw trajectory data
- `evaluation/` - Scripts for running evaluations and computing metrics
- `paper/` - Scripts for generating paper figures and tables
- `preprocessing/` - Scripts for data preprocessing and preparation
- `training/` - Scripts for model training and evaluation

## Script Categories

### Annotation Scripts (`annotation/`)
Contains scripts for data annotation using various approaches, including automated annotation using LLMs (like Gemini) and quality control mechanisms to ensure annotation accuracy. These scripts are critical for creating training data for the system's planning and execution components.

### Data Collection Scripts (`data_collection/`)
Includes scripts for collecting raw trajectory data from web browsing sessions. These scripts automate the process of capturing user interactions with e-commerce websites, which can then be used for training and analysis purposes. The data collection process includes capturing DOM states, screenshots, and action sequences.

### Evaluation Scripts (`evaluation/`)
Contains scripts for running systematic evaluations of the system's performance against benchmarks and baselines. These scripts automate the execution of test scenarios and calculate various performance metrics to assess system effectiveness and compare different implementations or configurations.

### Paper Scripts (`paper/`)
Includes scripts for generating figures, tables, and other materials for research papers. These scripts typically process evaluation results and other data to create visualizations and summary statistics for publication purposes.

### Preprocessing Scripts (`preprocessing/`)
Contains scripts for preparing raw data for training and evaluation. These scripts handle tasks like data cleaning, format conversion, feature extraction, and dataset splitting to create properly formatted training data for the system's various components.

### Training Scripts (`training/`)
Includes scripts for training the various models used in the system, including the ViT5 planner and PhoBERT embedding models. These scripts handle model initialization, training loop execution, validation, and checkpoint management to create effective models for the automation system.

## Overview

The scripts are organized by function to support the complete workflow of the WebOrdering_Automation system. Each subdirectory contains specialized scripts for that particular stage of the process. The scripts follow consistent patterns for command-line argument handling, logging, and error reporting to make them easy to use and integrate into automated workflows.

## Usage

Most scripts can be executed directly from the command line. Many scripts accept command-line arguments to customize their behavior. For detailed usage information, run a script with the `--help` flag. Many scripts also support configuration through YAML files or environment variables for more complex settings. Common usage patterns include specifying input/output directories, model paths, and various hyperparameters for training or evaluation tasks.

## Dependencies

Scripts typically require the project dependencies to be installed. Some may require specific models or data files to be available. The main requirements are specified in `requirements.txt` and can be installed using `pip install -r requirements.txt`. Some scripts may also require additional tools like Docker for containerized execution or specific model files that can be downloaded using `scripts/training/download_models.py`.