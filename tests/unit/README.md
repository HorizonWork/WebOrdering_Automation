# Unit Tests

This directory contains unit tests for individual components of the WebOrdering_Automation system. These tests validate the functionality of specific functions, classes, and modules in isolation.

## Contents

- `test_dom_only.py` - Tests for DOM processing components
- `test_execution.py` - Tests for execution module components
- `test_execution_suite.py` - Comprehensive test suite for execution components

## Overview

The unit tests verify the correctness of individual components by testing them in isolation from other parts of the system. These tests focus on specific functionality and help ensure that each component behaves as expected.

## Test Categories

- **Execution Tests**: Validate the execution module's ability to perform actions
- **DOM Processing Tests**: Test the perception module's DOM processing capabilities
- **Utility Tests**: Verify utility functions and common components
- **Skill Tests**: Test individual skill implementations
- **Validation Tests**: Check input validation and error handling

## Architecture

Unit tests follow the principle of testing one component at a time with controlled inputs and expected outputs. They typically use mock objects to isolate the component being tested from external dependencies, ensuring that failures are due to issues in the component itself rather than its dependencies.