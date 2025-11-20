# Perception Module

This directory contains components for processing visual and DOM information in the WebOrdering_Automation system. The perception module handles screenshot processing, DOM element analysis, and visual information extraction.

## Contents

- `__init__.py` - Python package initialization
- `dom_distiller.py` - Component for extracting relevant information from DOM
- `embedding.py` - Component for generating embeddings from visual and text data
- `scene_representation.py` - Component for creating scene representations
- `screenshot.py` - Component for processing and analyzing screenshots
- `ui_detector.py` - Component for detecting UI elements in web pages
- `vision_enhancer.py` - Component for enhancing visual information processing

## Overview

The perception module provides the system with environmental awareness by processing visual information from web pages and extracting relevant structural and content information from the DOM. This enables the system to understand the current state of the web interface.

## Components

- **DOM Distiller**: Extracts relevant information from the page DOM structure
- **UI Detector**: Identifies and classifies UI elements in web pages
- **Screenshot Processor**: Analyzes visual content of web pages
- **Embedding Generator**: Creates vector representations of visual and textual content
- **Scene Representation**: Creates structured representations of the web interface state
- **Vision Enhancer**: Improves visual information processing capabilities

## Architecture

The perception module integrates multiple information sources (DOM structure, visual elements, text content) to create a comprehensive understanding of the web interface. This multi-modal approach allows for robust perception even when individual information sources are incomplete or noisy.