"""Training callbacks for logging and monitoring."""

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments
import numpy as np
from typing import Dict, Any
import wandb
import logging

logger = logging.getLogger(__name__)


class AttributionLoggingCallback(TrainerCallback):
    """
    Callback to log attribution statistics during training.
    
    Tracks variance, concentration, and other attribution metrics
    to monitor the effect of the variance penalty.
    """
    
    def __init__(self, log_frequency: int = 100, use_wandb: bool = False):
        """
        Initialize the callback.
        
        Args:
            log_frequency: How often to compute and log attribution stats
            use_wandb: Whether to log to Weights & Biases
        """
        self.log_frequency = log_frequency
        self.use_wandb = use_wandb
        self.step_count = 0
        
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        trainer=None,
        **kwargs
    ):
        """Log attribution statistics at specified intervals."""
        self.step_count += 1
        
        # Only log at specified frequency
        if self.step_count % self.log_frequency != 0:
            return
            
        # Only log if trainer is IGTrainer with loss tracking
        if not hasattr(trainer, 'loss_components'):
            return
            
        try:
            # Get recent loss components
            recent_penalty = trainer.loss_components['penalty'][-self.log_frequency:]
            recent_ce_loss = trainer.loss_components['ce_loss'][-self.log_frequency:]
            
            if recent_penalty:
                attribution_stats = {
                    'attribution/mean_penalty': np.mean(recent_penalty),
                    'attribution/penalty_std': np.std(recent_penalty),
                    'attribution/penalty_ratio': np.mean(recent_penalty) / (np.mean(recent_ce_loss) + 1e-8),
                    'attribution/penalty_trend': self._compute_trend(recent_penalty),
                }
                
                # Log to trainer logs
                trainer.log(attribution_stats)
                
                # Log to wandb if enabled
                if self.use_wandb and wandb.run is not None:
                    wandb.log(attribution_stats, step=state.global_step)
                    
                logger.info(f"Step {state.global_step}: Attribution penalty = {attribution_stats['attribution/mean_penalty']:.6f}")
                
        except Exception as e:
            logger.warning(f"Failed to log attribution statistics: {e}")
    
    def _compute_trend(self, values: list) -> float:
        """Compute trend (slope) of recent values."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        return float(slope)


class EarlyStoppingWithAttribution(TrainerCallback):
    """
    Early stopping that considers both performance and attribution stability.
    
    Stops training when validation performance plateaus AND attribution
    variance has stabilized, indicating the regularization has converged.
    """
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        attribution_stability_threshold: float = 0.01,
        metric_for_best_model: str = "eval_accuracy"
    ):
        """
        Initialize early stopping with attribution monitoring.
        
        Args:
            early_stopping_patience: Number of evaluations to wait for improvement
            attribution_stability_threshold: Threshold for attribution variance stability
            metric_for_best_model: Metric to monitor for early stopping
        """
        self.early_stopping_patience = early_stopping_patience
        self.attribution_stability_threshold = attribution_stability_threshold
        self.metric_for_best_model = metric_for_best_model
        
        self.best_metric = None
        self.patience_counter = 0
        self.attribution_variances = []
        
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        trainer=None,
        logs=None,
        **kwargs
    ):
        """Check for early stopping conditions."""
        if logs is None:
            return
            
        current_metric = logs.get(self.metric_for_best_model)
        if current_metric is None:
            return
            
        # Track attribution variance if available
        if hasattr(trainer, 'loss_components') and trainer.loss_components['penalty']:
            current_variance = np.var(trainer.loss_components['penalty'][-100:])  # Variance of recent penalties
            self.attribution_variances.append(current_variance)
        
        # Check if current metric is better
        is_better = (
            self.best_metric is None or
            current_metric > self.best_metric  # Assuming higher is better (accuracy, etc.)
        )
        
        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Check attribution stability
        attribution_stable = self._is_attribution_stable()
        
        # Early stopping condition
        should_stop = (
            self.patience_counter >= self.early_stopping_patience and
            attribution_stable
        )
        
        if should_stop:
            control.should_training_stop = True
            logger.info(
                f"Early stopping triggered: "
                f"patience={self.patience_counter}, "
                f"attribution_stable={attribution_stable}"
            )
            
    def _is_attribution_stable(self) -> bool:
        """Check if attribution variance has stabilized."""
        if len(self.attribution_variances) < 3:
            return False
            
        # Check if recent attribution variances are stable
        recent_variances = self.attribution_variances[-3:]
        variance_of_variances = np.var(recent_variances)
        
        return variance_of_variances < self.attribution_stability_threshold