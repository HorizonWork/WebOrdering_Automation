# Paper Scripts

This directory contains executable scripts for generating paper figures, tables, and other materials used in the research paper describing the WebOrdering_Automation system.

## Contents

- `__init__.py` - Python package initialization
- `export_results.py` - Script for exporting evaluation results in paper format
- `generate_figures.py` - Script for generating paper figures
- `generate_tables.py` - Script for generating paper tables

## Overview

The paper scripts automate the process of creating visualizations and tables for the research paper. These scripts process evaluation results and other data to generate publication-ready figures and tables.

## Process

The scripts in this directory:
1. Process evaluation results to create summary statistics
2. Generate figures showing performance comparisons and results
3. Create tables with quantitative results and comparisons
4. Format outputs in the required style for the paper

## Usage

Run the scripts from the command line. For example:
- `python -m scripts.paper.generate_figures` to generate paper figures
- `python -m scripts.paper.generate_tables` to generate paper tables
- `python -m scripts.paper.export_results` to export results in paper format

## Dependencies

These scripts require the project dependencies to be installed and access to evaluation results and other data needed for the paper.