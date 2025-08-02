"""Attribution computation and analysis for V-BIG."""

from .core import compute_ig_attributions, AttributionProcessor
from .losses import VarianceBasedLoss, compute_attribution_penalty
from .analyzer import AttributionAnalyzer

__all__ = [
    "compute_ig_attributions",
    "AttributionProcessor", 
    "VarianceBasedLoss",
    "compute_attribution_penalty",
    "AttributionAnalyzer",
]