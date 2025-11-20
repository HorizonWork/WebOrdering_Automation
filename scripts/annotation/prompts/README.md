# Annotation Prompts

This directory contains prompt templates used for Gemini-based annotation of trajectory data. These prompts guide the annotation process to ensure consistent and accurate annotations.

## Contents

- `controller_annotation.txt` - Prompt template for controller action annotations
- `planner_annotation.txt` - Prompt template for planner decision annotations

## Overview

The annotation prompts provide structured templates that guide Gemini models in generating consistent and accurate annotations for trajectory data. These prompts are carefully designed to capture the essential information needed for training the WebOrdering_Automation system.

## Prompt Categories

- **Controller annotations**: Prompts for annotating low-level interaction actions
- **Planner annotations**: Prompts for annotating high-level decision-making processes

## Usage

These prompt files are used by the annotation scripts to guide the Gemini models in generating annotations. The prompts are loaded and potentially customized with specific examples or context before being sent to the model.

## Design Principles

The prompts are designed to:
- Extract structured information from trajectory data
- Maintain consistency across annotations
- Capture relevant context for training models
- Be specific enough to generate accurate annotations