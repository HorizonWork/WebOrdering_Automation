# Tests

This directory contains all test suites for the WebOrdering_Automation system. The tests are organized to validate different aspects of the system, from individual units to full integration scenarios. The test suite is designed to ensure the reliability, correctness, and robustness of the automation system across various e-commerce platforms and usage scenarios. Tests are organized by scope and purpose, from unit tests that validate individual functions to integration tests that verify component interactions and end-to-end tests that validate complete workflows.

## Contents

- `__init__.py` - Python package initialization
- `full_pipeline_test.py` - End-to-end test of the complete automation pipeline
- `path_setup.py` - Test environment setup and path configuration
- `test_browser_with_settings.py` - Browser configuration and settings tests
- `test_chrome_profile.py` - Chrome profile setup and management tests
- `test_execution_quick.py` - Quick execution tests for basic functionality
- `test_execution_stepbystep.py` - Step-by-step execution tests
- `test_models.py` - Model loading and inference tests
- `fixtures/` - Test fixtures and mock data
- `integration/` - Integration tests for multi-component interactions
- `performance/` - Performance and stress tests
- `unit/` - Unit tests for individual components

## Overview

The test suite follows a comprehensive testing strategy that covers multiple levels of the system's functionality. It includes unit tests for individual components, integration tests for component interactions, and end-to-end tests that validate complete user workflows. The testing approach ensures that each layer of the 4-layer architecture functions correctly both in isolation and as part of the complete system. Additionally, the test suite includes performance and stress tests to validate the system's behavior under various load conditions and edge cases specific to e-commerce automation scenarios.

## Test Categories

### Unit Tests (`unit/`)
Focus on individual functions and classes in isolation. These tests validate the core logic of each component without dependencies on external systems. Unit tests cover the perception, planning, execution, and learning layers individually, ensuring each component functions as expected with various input scenarios and edge cases. These tests are typically fast to execute and provide immediate feedback during development. They are located in the `unit/` subdirectory and can be run with `pytest tests/unit/`.

### Integration Tests (`integration/`)
Validate interactions between multiple components and ensure that different modules work together correctly. These tests verify that data flows properly between the perception, planning, execution, and learning layers. Integration tests are particularly important for the WebOrdering_Automation system as they validate that the hierarchical architecture functions correctly and that information is properly passed between layers. These tests are located in the `integration/` subdirectory and can be run with `pytest tests/integration/`.

### Performance Tests (`performance/`)
Measure system performance under various conditions, including response times, memory usage, and throughput. These tests are crucial for the automation system to ensure that it can handle real-world usage patterns efficiently. Performance tests include browser resource usage, model inference times, and overall task completion times. These tests are located in the `performance/` subdirectory and can be run with `pytest tests/performance/`.

### End-to-End Tests (`full_pipeline_test.py` and other main tests)
Verify complete workflows from start to finish, simulating real user interactions with e-commerce platforms. These tests validate that the entire automation pipeline works correctly, from user query processing to task completion. End-to-end tests are particularly important for validating the system's ability to handle complex e-commerce scenarios like product search, filtering, and purchase completion across different platforms like Shopee and Lazada. These tests typically use mock or sandbox environments to avoid impacting real e-commerce sites during testing.

## Test Organization

### Fixtures (`fixtures/`)
Contains test fixtures and mock data used across different test types. This includes mock DOM structures, sample screenshots, and test scenarios that can be reused across multiple tests. The fixtures directory includes `mock_dom.html` and other resources that simulate various web page states for testing purposes without requiring actual web access. These fixtures are essential for creating consistent and repeatable test scenarios that don't depend on external systems or network connectivity.

### Test Configuration
Tests are configured to run in headless mode by default to enable automated execution in CI/CD environments. However, tests can be configured to run in visible mode for debugging purposes. The test configuration is managed through pytest configuration files and environment variables to allow for flexible test execution in different environments (development, CI, production-like testing environments).

## Running Tests

Tests can be run using pytest. For specific test categories, you can run:
- `pytest tests/unit/` - Run all unit tests
- `pytest tests/integration/` - Run all integration tests
- `pytest tests/performance/` - Run all performance tests
- `pytest tests/ -v` - Run all tests with verbose output
- `pytest tests/ --cov=src` - Run all tests with coverage reporting for the src directory

For more detailed information about test execution and configuration, see `TEST_EXECUTION_GUIDE.md` in the documentation directory. The test execution guide provides additional instructions for running specific test scenarios, debugging failing tests, and setting up test environments with proper configurations for e-commerce platform testing.