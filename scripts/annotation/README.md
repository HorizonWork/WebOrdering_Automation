# Annotation Scripts

This directory contains executable scripts for annotating trajectory data using Gemini models. These scripts automate the process of adding structured information to raw trajectories.

## Contents

- `__init__.py` - Python package initialization
- `batch_annotate.py` - Script for batch annotation of trajectory data
- `gemini_annotator.py` - Core module for Gemini-based annotation
- `quality_control.py` - Script for quality control of annotations
- `validate_annotations.py` - Script for validating annotation quality

## Overview

The annotation scripts implement the process of adding structured annotations to raw trajectory data using Gemini models. These annotations are essential for training supervised learning models that can understand and replicate the decision-making process in web ordering tasks.

## Process

The scripts in this directory:
1. Apply Gemini models to annotate raw trajectories with structured information
2. Validate the quality of generated annotations
3. Perform quality control checks on annotations
4. Process trajectories in batch mode for efficiency

## Usage

Run the scripts from the command line. For example:
- `python -m scripts.annotation.batch_annotate` to annotate trajectories in batch mode
- `python -m scripts.annotation.quality_control` to check annotation quality

## Dependencies

These scripts require access to Gemini models and the project dependencies to be installed. They work with the raw trajectory data collected in the data collection stage.