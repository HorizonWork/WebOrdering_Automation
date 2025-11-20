# Baselines

This directory contains baseline agent implementations used for comparison with the main WebOrdering_Automation system. These baselines provide reference points to evaluate the effectiveness of our approach.

## Contents

- `__init__.py` - Python package initialization
- `gemini_agent.py` - Implementation using Google's Gemini model for decision making
- `gpt4_agent.py` - Implementation using OpenAI's GPT-4 model for decision making
- `rule_based_agent.py` - Rule-based approach implementation for comparison

## Overview

The baseline implementations represent different approaches to the web ordering automation task. These implementations serve as reference points to demonstrate the effectiveness of the main system approach.

## Baseline Approaches

- **Gemini Agent**: Uses Google's Gemini model to process observations and make decisions
- **GPT-4 Agent**: Uses OpenAI's GPT-4 model to process observations and make decisions
- **Rule-based Agent**: Implements heuristic-based approaches using predefined rules

## Usage

These baselines can be run using the evaluation framework to compare their performance against the main system. Each baseline follows the same interface as the main system for consistent evaluation.