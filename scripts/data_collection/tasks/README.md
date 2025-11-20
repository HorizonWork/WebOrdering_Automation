# Data Collection Tasks

This directory contains task definition files used for data collection in the WebOrdering_Automation project. These files define the specific web ordering tasks to be performed during data collection.

## Contents

- `lazada_tasks.yaml` - Task definitions for Lazada platform
- `shopee_tasks.yaml` - Task definitions for Shopee platform

## Overview

The task files define the specific web ordering scenarios to be collected during the data collection phase. These include details about the products to search for, the steps to complete an order, and any platform-specific considerations.

## Task Structure

Each task definition includes:
- Target platform (Lazada, Shopee, etc.)
- Product categories and search terms
- Steps to complete the ordering process
- Success criteria for task completion
- Any special instructions or constraints

## Usage

These task files are used by the data collection scripts to guide the collection process. The scripts read these definitions and execute the specified tasks to collect diverse trajectory examples.

## Platform Coverage

Currently, the system supports task definitions for:
- Lazada (Southeast Asian e-commerce platform)
- Shopee (Southeast Asian e-commerce platform)