# Learning Memory

This directory contains memory and storage components for the WebOrdering_Automation system's learning module. These components provide storage and retrieval of experiences, trajectories, and learned patterns.

## Contents

- `__init__.py` - Python package initialization
- `rail.py` - Component for memory rail implementation
- `trajectory_buffer.py` - Component for storing and managing trajectory data
- `vector_store.py` - Component for vector-based storage and retrieval

## Overview

The memory submodule implements various storage mechanisms that enable the system to remember past experiences and use them for learning and decision making. These components support different types of memory needs within the learning system.

## Components

- **Rail**: Implements memory rail functionality for structured storage of experiences
- **Trajectory Buffer**: Manages collections of trajectories for training and analysis
- **Vector Store**: Provides vector-based storage for similarity search and retrieval

## Architecture

The memory system uses multiple storage approaches to handle different types of information:
- Short-term buffers for recent experiences
- Vector stores for similarity-based retrieval
- Structured storage for organized experience data