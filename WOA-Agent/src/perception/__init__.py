"""
Perception Package - Input Processing Layer
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .dom_distiller import DOMDistiller
from .embedding import PerceptionEmbedding
from .scene_representation import SceneRepresentation
from .ui_detector import UIDetector

__all__ = [
    'DOMDistiller',
    'PerceptionEmbedding',
    'SceneRepresentation',
    'UIDetector'
]
