"""Custom trainer with IG regularization."""

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Any
import numpy as np

from ..attribution.core import compute_ig_attributions
from ..attribution.losses import VarianceBasedLoss


@dataclass
class VarianceTrainingArguments(TrainingArguments):
    """Training args with V-BIG specific parameters."""
    
    ig_mode: str = field(
        default='variance',
        metadata={'help': 'IG regularization mode: none, variance, stopwords, or both'}
    )
    lambda_reg: float = field(
        default=0.1,
        metadata={'help': 'Regularization strength for attribution penalty'}
    )
    alpha_variance: float = field(
        default=10.0,
        metadata={'help': 'Scaling factor for variance penalty'}
    )
    ig_steps: int = field(
        default=10,
        metadata={'help': 'Number of integration steps for IG computation'}
    )


class IGTrainer(Trainer):
    """
    A custom Trainer that integrates IG computation into the training loop.
    
    This trainer computes Integrated Gradients attributions during training
    and adds a variance-based penalty to encourage more distributed attention.
    """
    
    def __init__(
        self,
        ig_mode: str = 'none',
        lambda_reg: float = 0.1,
        alpha_variance: float = 10.0,
        ig_steps: int = 10,
        stopword_ids: Optional[Set[int]] = None,
        *args,
        **kwargs
    ):
        """
        Initialize the IG trainer.
        
        Args:
            ig_mode: Attribution penalty mode ('none', 'variance', 'stopwords', 'both')
            lambda_reg: Regularization strength
            alpha_variance: Variance scaling factor
            ig_steps: Number of IG integration steps
            stopword_ids: Set of stopword token IDs for stopword penalty
        """
        super().__init__(*args, **kwargs)
        self.ig_mode = ig_mode
        self.lambda_reg = lambda_reg
        self.alpha_variance = alpha_variance
        self.ig_steps = ig_steps
        self.stopword_ids = stopword_ids if stopword_ids is not None else set()
        
        # Initialize the variance-based loss function
        self.variance_loss = VarianceBasedLoss(
            lambda_reg=lambda_reg,
            alpha=alpha_variance,
            ig_mode=ig_mode,
            stopword_ids=stopword_ids
        )
        
        # Track loss components for logging
        self.loss_components = {'ce_loss': [], 'penalty': [], 'total_loss': []}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with optional IG-based regularization.
        
        Args:
            model: The model being trained
            inputs: Batch of input data
            return_outputs: Whether to return model outputs
            
        Returns:
            loss: Computed loss value
            outputs (optional): Model outputs if return_outputs=True
        """
        # Extract inputs
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels')

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        # If no IG regularization, use standard loss
        if self.ig_mode == 'none' or self.lambda_reg == 0:
            loss = outputs.loss
            
            # Track loss components
            self.loss_components['ce_loss'].append(loss.item())
            self.loss_components['penalty'].append(0.0)
            self.loss_components['total_loss'].append(loss.item())
        else:
            # Compute IG attributions
            with torch.enable_grad():
                token_attributions = compute_ig_attributions(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    n_steps=self.ig_steps
                )
            
            # Compute loss with variance penalty
            loss = self.variance_loss(
                logits=logits,
                labels=labels,
                token_attributions=token_attributions,
                input_ids=input_ids
            )
            
            # Track loss components for analysis
            components = self.variance_loss.get_loss_components(
                logits=logits,
                labels=labels,
                token_attributions=token_attributions,
                input_ids=input_ids
            )
            
            self.loss_components['ce_loss'].append(components['ce_loss'])
            self.loss_components['penalty'].append(components['penalty'])
            self.loss_components['total_loss'].append(components['total_loss'])

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with loss component tracking."""
        # Add average loss components to logs
        if self.loss_components['total_loss']:
            logs['avg_ce_loss'] = np.mean(self.loss_components['ce_loss'][-100:])  # Last 100 steps
            logs['avg_penalty'] = np.mean(self.loss_components['penalty'][-100:])
            logs['penalty_ratio'] = logs['avg_penalty'] / (logs['avg_ce_loss'] + 1e-8)
        
        super().log(logs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Enhanced evaluation with attribution analysis."""
        # Standard eval
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add loss component summaries to eval output
        if self.loss_components['total_loss']:
            output[f'{metric_key_prefix}_avg_ce_loss'] = np.mean(self.loss_components['ce_loss'])
            output[f'{metric_key_prefix}_avg_penalty'] = np.mean(self.loss_components['penalty'])
            output[f'{metric_key_prefix}_penalty_ratio'] = (
                output[f'{metric_key_prefix}_avg_penalty'] / 
                (output[f'{metric_key_prefix}_avg_ce_loss'] + 1e-8)
            )
        
        return output
    
    def get_loss_summary(self) -> Dict[str, float]:
        """Get summary statistics of loss components."""
        if not self.loss_components['total_loss']:
            return {}
            
        return {
            'mean_ce_loss': np.mean(self.loss_components['ce_loss']),
            'mean_penalty': np.mean(self.loss_components['penalty']),
            'mean_total_loss': np.mean(self.loss_components['total_loss']),
            'std_ce_loss': np.std(self.loss_components['ce_loss']),
            'std_penalty': np.std(self.loss_components['penalty']),
            'penalty_contribution': np.mean(self.loss_components['penalty']) / np.mean(self.loss_components['total_loss'])
        }
    
    def reset_loss_tracking(self):
        """Reset loss component tracking."""
        self.loss_components = {'ce_loss': [], 'penalty': [], 'total_loss': []}