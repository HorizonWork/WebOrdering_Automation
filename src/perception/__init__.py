"""
Perception Package - Input Processing Layer
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .dom_distiller import DOMDistiller  # noqa: E402
from .embedding import PerceptionEmbedding  # noqa: E402
from .scene_representation import SceneRepresentation  # noqa: E402
from .ui_detector import UIDetector  # noqa: E402

__all__ = [
    'DOMDistiller',
    'PerceptionEmbedding',
    'SceneRepresentation',
    'UIDetector'
]
