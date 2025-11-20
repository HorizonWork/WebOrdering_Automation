# Annotation Notebooks

This directory contains Jupyter notebooks for annotating collected trajectory data using Gemini models. These notebooks implement the annotation pipeline that adds structured information to raw trajectories.

## Contents

- `annotation_cost_analysis.ipynb` - Notebook for analyzing annotation costs
- `annotation_quality.ipynb` - Notebook for evaluating annotation quality
- `gemini_annotation_demo.ipynb` - Notebook demonstrating Gemini-based annotation

## Overview

The annotation notebooks implement the process of adding structured annotations to raw trajectory data using Gemini models. These annotations are essential for training supervised learning models that can understand and replicate the decision-making process in web ordering tasks.

## Process

The notebooks in this directory:
1. Apply Gemini models to annotate raw trajectories with structured information
2. Evaluate the quality of generated annotations
3. Analyze the cost and efficiency of the annotation process

## Dependencies

These notebooks require access to Gemini models and the project dependencies to be installed. They work with the raw trajectory data collected in the previous stage.