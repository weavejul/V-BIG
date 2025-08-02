"""
Variance-Based Integrated Gradients (V-BIG) for Robust NLI

A package for training natural language inference models using
attribution-guided regularization based on Integrated Gradients.
"""

__version__ = "0.1.0"
__author__ = "Julian Weaver"

from .attribution import compute_ig_attributions, VarianceBasedLoss, AttributionAnalyzer
from .training import IGTrainer, VarianceTrainingArguments
from .data import prepare_dataset_nli, NLIDataProcessor
from .evaluation import compute_nli_metrics
from .visualization import visualize_attributions, compare_attributions

__all__ = [
    "compute_ig_attributions",
    "VarianceBasedLoss", 
    "IGTrainer",
    "VarianceTrainingArguments",
    "prepare_dataset_nli",
    "NLIDataProcessor",
    "compute_nli_metrics",
    "AttributionAnalyzer",
    "visualize_attributions",
    "compare_attributions",
]