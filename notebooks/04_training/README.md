# Training Notebooks

This directory contains Jupyter notebooks for training machine learning models in the WebOrdering_Automation system. These notebooks implement the training process for controller and planner models.

## Contents

- `convergence_analysis.ipynb` - Notebook for analyzing model convergence
- `train_controller.ipynb` - Notebook for training the controller model
- `train_planner.ipynb` - Notebook for training the planner model

## Overview

The training notebooks implement the process of training machine learning models that power the WebOrdering_Automation system. These models learn to make decisions based on the preprocessed trajectory data.

## Process

The notebooks in this directory:
1. Train controller models that handle low-level interactions
2. Train planner models that handle high-level decision making
3. Analyze model convergence and training dynamics
4. Evaluate model performance during training

## Dependencies

These notebooks require the project dependencies to be installed and access to the preprocessed training data. They may also require specific hardware (e.g., GPUs) for efficient training.