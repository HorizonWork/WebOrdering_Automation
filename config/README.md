# Configuration

This directory contains all configuration files for the WebOrdering_Automation project. These files define settings, parameters, and schemas that control various aspects of the system's behavior. Configuration files are organized to separate different types of settings and make the system easier to customize and maintain.

## Contents

- `__init__.py` - Python package initialization
- `data_catalog.yaml` - Data source definitions and configurations
- `logging.yaml` - Logging configuration settings
- `models.yaml` - Model configurations and parameters
- `selectors.yaml` - DOM selectors configuration for element identification
- `settings.py` - General application settings
- `skills.yaml` - Skill definitions and configurations

## Configuration Files Overview

### data_catalog.yaml
Defines data sources, paths, and access parameters for various datasets used in the system. This includes paths to training data, model checkpoints, and evaluation datasets.

### logging.yaml
Configures the logging system, including log levels, output formats, and destinations. Controls how system events and errors are recorded and reported during execution.

### models.yaml
Specifies model configurations, including model names, paths, parameters, and settings for different AI models used in the system (e.g., PhoBERT, ViT5).

### selectors.yaml
Contains DOM selectors used for element identification and interaction. These selectors are used by the perception layer to identify UI elements on web pages. This is critical for the system's ability to interact with web interfaces effectively.

### settings.py
Contains general application settings, including default parameters, system behavior flags, and global configuration options that affect the overall system operation.

### skills.yaml
Defines the available skills and their configurations, including parameters and settings for different types of web interactions (click, type, scroll, etc.).

## Usage

The configuration files are loaded at application startup and provide the necessary settings for different components of the system. The `settings.py` file serves as the main entry point for accessing configuration values throughout the application. YAML files are typically used for static configuration that doesn't change during runtime, while Python files allow for more dynamic configuration options.

## Adding New Configurations

When adding new configuration options, consider:
1. Whether the option should be in YAML (for complex nested structures) or Python (for dynamic configurations)
2. Proper validation of configuration values
3. Default values for optional configurations
4. Clear documentation for each configuration option
5. The appropriate file type based on security and privacy requirements (sensitive information should be handled through environment variables)

## Environment Variables

Some configuration values can be overridden by environment variables. See `.env.example` for a list of supported environment variables and their descriptions. This allows for easy customization without modifying the configuration files directly, which is especially useful for deployment in different environments.