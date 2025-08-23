"""
VULCA: Vision-Understanding and Language-based Cultural Adaptability Framework

A comprehensive evaluation system for Multimodal Large Language Models (MLLMs) 
using Chinese painting critique as the evaluation domain.
"""

__version__ = "1.0.0"
__author__ = "VULCA Framework Contributors"

from .core.mllm_evaluator import MLLMEvaluator
from .core.persona_manager import PersonaManager
from .preprocessing.adaptive_window import AdaptiveWindowProcessor
from .analysis.semantic_analyzer import SemanticAnalyzer

__all__ = [
    "MLLMEvaluator",
    "PersonaManager", 
    "AdaptiveWindowProcessor",
    "SemanticAnalyzer"
]