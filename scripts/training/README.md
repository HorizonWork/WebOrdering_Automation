# Training Scripts

This directory contains executable scripts for training machine learning models in the WebOrdering_Automation system. These scripts implement the training process for controller and planner models.

## Contents

- `__init__.py` - Python package initialization
- `download_models.py` - Script for downloading pre-trained models
- `evaluate_model.py` - Script for evaluating trained models
- `train_controller.py` - Script for training the controller model
- `train_planner.py` - Script for training the planner model

## Overview

The training scripts implement the process of training machine learning models that power the WebOrdering_Automation system. These models learn to make decisions based on the preprocessed trajectory data.

## Process

The scripts in this directory:
1. Download pre-trained models as needed
2. Train controller models that handle low-level interactions
3. Train planner models that handle high-level decision making
4. Evaluate model performance during and after training

## Usage

Run the scripts from the command line. For example:
- `python -m scripts.training.train_controller` to train the controller model
- `python -m scripts.training.train_planner` to train the planner model
- `python -m scripts.training.evaluate_model` to evaluate a trained model

## Dependencies

These scripts require the project dependencies to be installed and access to the preprocessed training data. They may also require specific hardware (e.g., GPUs) for efficient training.