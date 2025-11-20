# Jupyter Notebooks

This directory contains Jupyter notebooks for various stages of the WebOrdering_Automation project workflow. These notebooks serve as interactive tools for data collection, annotation, preprocessing, training, and evaluation activities. The notebooks provide a visual and interactive environment for understanding data, experimenting with models, and analyzing results throughout the development process of the automation system. They are particularly useful for exploratory data analysis, model development, and result visualization in the context of Vietnamese e-commerce automation tasks.

## Contents

- `01_data_collection/` - Notebooks for collecting raw trajectory data from web ordering tasks
- `02_annotation/` - Notebooks for data annotation and quality analysis using LLMs
- `03_preprocessing/` - Notebooks for data preprocessing and preparation for training
- `04_training/` - Notebooks for model training, validation, and analysis
- `05_evaluation/` - Notebooks for evaluation, results analysis, and comparison with baselines

## Overview

The notebooks are organized chronologically according to the project workflow, creating a comprehensive pipeline from raw data to final evaluation. Each stage builds upon the previous one, with outputs from one stage feeding into the next. This structure allows for iterative development and refinement of the automation system components. The notebooks are particularly valuable for understanding the characteristics of Vietnamese e-commerce data and for fine-tuning the system to handle local patterns and requirements effectively. They also serve as documentation of the development process and provide reproducible analysis of the system's performance and behavior at each stage of the pipeline. The notebook structure enables rapid experimentation and iteration, which is crucial for developing an effective automation system that can handle the complexities of Vietnamese e-commerce platforms like Shopee and Lazada.


### 01_data_collection/
Contains notebooks for collecting raw trajectory data from web ordering tasks. These notebooks demonstrate how to capture user interactions, DOM states, and action sequences from real browsing sessions. They include tools for data validation, cleaning, and initial analysis of collected trajectories. The notebooks are designed to work with the data collection scripts in the `scripts/data_collection/` directory and provide visual feedback on the quality and characteristics of collected data. They also include tools for identifying and filtering out low-quality trajectories that might negatively impact model training.


### 02_annotation/
Contains notebooks for data annotation and quality analysis using LLMs, particularly Gemini models for Vietnamese language understanding. These notebooks implement automated annotation processes that convert raw trajectories into structured training data for the system's planning and execution components. The notebooks include quality control mechanisms to ensure annotation accuracy and consistency. They also provide visualization tools for reviewing and validating annotation results, which is critical for maintaining high-quality training data. The annotation notebooks are essential for creating the training datasets needed to teach the system how to plan and execute web automation tasks effectively.


### 03_preprocessing/
Contains notebooks for data preprocessing and preparation for training. These notebooks implement various data transformation and feature extraction techniques to prepare raw collected data for model training. They include tools for handling missing values, normalizing data formats, and creating appropriate input representations for different system components. The preprocessing notebooks are crucial for ensuring that training data is in the correct format and quality for effective model training. They also include exploratory data analysis tools to understand data distributions and identify potential issues before training.


### 04_training/
Contains notebooks for model training, validation, and analysis. These notebooks provide interactive environments for training different components of the automation system, including the ViT5 planner and PhoBERT embedding models. They include tools for hyperparameter tuning, model validation, and performance monitoring during training. The training notebooks also provide visualization of training progress, loss curves, and validation metrics to help optimize model performance. They are particularly useful for fine-tuning models for Vietnamese e-commerce tasks and for experimenting with different architectures and training approaches.


### 05_evaluation/
Contains notebooks for evaluation, results analysis, and comparison with baselines. These notebooks provide comprehensive tools for analyzing the performance of trained models and comparing them against various baseline approaches. They include visualization tools for understanding model behavior, error analysis capabilities, and comparison frameworks for assessing different system configurations. The evaluation notebooks are essential for understanding how well the system performs on Vietnamese e-commerce tasks and for identifying areas for improvement. They also provide tools for generating the figures and tables used in research papers and documentation.


## Usage

Each notebook directory contains specific notebooks for that stage of the workflow. The notebooks are designed to be run in sequence, with outputs from one stage feeding into the next. Many notebooks include visualization and analysis components to help understand the data and model behavior at each stage. The notebooks typically include detailed explanations of the code and methodology, making them educational resources as well as practical tools. It's recommended to run the notebooks in the order of the project workflow (01-05) to understand the complete development pipeline. Each notebook should be run in the project's Python environment with all dependencies installed. The notebooks often include configuration options that can be adjusted for different experimental conditions or data sets.


## Dependencies

These notebooks require the project dependencies to be installed as specified in `requirements.txt`. Some notebooks may require specific models or data files to be available. The notebooks are designed to work with the same environment and dependencies as the main project code. Certain notebooks may require additional computational resources, particularly those in the training section, which may need GPU access for efficient execution. The notebooks also rely on data and configuration files from other parts of the project, so they should be run from the project root directory to ensure proper file path resolution. Some notebooks may require access to external APIs (e.g., for LLM-based annotation) and appropriate API keys configured in the environment.


## Best Practices

When working with these notebooks, consider the following best practices:
- Run notebooks in the intended sequence to ensure proper data flow between stages
- Save notebook outputs appropriately to avoid re-computation of expensive operations
- Use the notebooks for exploratory analysis to understand data characteristics and model behavior
- Update notebooks with findings and insights to maintain documentation of the development process
- Share insights from notebooks with the development team to guide system improvements