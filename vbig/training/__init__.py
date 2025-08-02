"""Training components for V-BIG models."""

from .trainer import IGTrainer, VarianceTrainingArguments
from .callbacks import AttributionLoggingCallback

__all__ = [
    "IGTrainer",
    "VarianceTrainingArguments", 
    "AttributionLoggingCallback",
]