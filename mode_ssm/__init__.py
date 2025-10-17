"""
MODE-SSM: Mode-Aware State-Space Decoder for Brain-to-Text Translation

A neural decoder system for translating brain signals to text using state-space models
and mode-aware processing for silent and vocalized speech.
"""

__version__ = "0.1.0"
__author__ = "MODE-SSM Team"

# Import key components for easier access
from .models import *
from .checkpoint_manager import CheckpointManager
from .evaluation_metrics import EvaluationManager
from .training_stages import CurriculumTrainer

__all__ = [
    "CheckpointManager",
    "EvaluationManager",
    "CurriculumTrainer"
]