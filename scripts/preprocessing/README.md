# Preprocessing Scripts

This directory contains executable scripts for preprocessing collected and annotated trajectory data. These scripts prepare the data for training machine learning models in the WebOrdering_Automation system.

## Contents

- `__init__.py` - Python package initialization
- `build_controller_dataset.py` - Script for building controller training datasets
- `build_embeddings.py` - Script for building embedding representations
- `build_planner_dataset.py` - Script for building planner training datasets
- `compute_statistics.py` - Script for computing dataset statistics
- `split_dataset.py` - Script for splitting datasets into train/validation/test

## Overview

The preprocessing scripts implement the transformation of annotated trajectory data into formats suitable for training machine learning models. This includes feature extraction, data cleaning, and dataset construction.

## Process

The scripts in this directory:
1. Transform annotated trajectories into training-ready formats
2. Extract relevant features from the trajectory data
3. Build training datasets for different model components
4. Compute statistics about the processed datasets
5. Split datasets into appropriate train/validation/test splits

## Usage

Run the scripts from the command line. For example:
- `python -m scripts.preprocessing.build_controller_dataset` to build controller datasets
- `python -m scripts.preprocessing.split_dataset` to split datasets

## Dependencies

These scripts require the project dependencies to be installed and work with the annotated data generated in the annotation stage.