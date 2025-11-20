# Integration Tests

This directory contains integration tests for the WebOrdering_Automation system. These tests validate the interaction between multiple components and ensure the system works correctly as a whole.

## Contents

- `test_agent_flow.py` - Tests for complete agent workflows and interactions
- `test_shopee_workflow.py` - Tests for complete Shopee ordering workflows

## Overview

The integration tests verify that different modules of the system work together correctly. These tests typically involve multiple components such as perception, planning, and execution working in concert to complete tasks.

## Test Categories

- **Agent Flow Tests**: Validate the complete flow from perception through planning to execution
- **Workflow Tests**: Test complete end-to-end workflows on specific platforms (e.g., Shopee)
- **Module Integration**: Verify that different modules interact correctly with each other
- **State Consistency**: Ensure state is properly maintained across module boundaries

## Architecture

Integration tests are designed to test the interfaces between components rather than the internal implementation details. They focus on verifying that data flows correctly between modules and that the system behaves as expected when components work together.