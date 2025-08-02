"""Loss functions incorporating attribution-based regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Set, Optional


def compute_attribution_penalty(
    token_attributions: torch.Tensor,
    input_ids: torch.Tensor,
    ig_mode: str = 'variance',
    stopword_ids: Optional[Set[int]] = None,
    alpha: float = 10.0
) -> torch.Tensor:
    """
    Compute an attribution-based penalty given the IG mode.
    
    Args:
        token_attributions: [batch, seq_len] attribution scores
        input_ids: [batch, seq_len] token IDs
        ig_mode: One of ['none', 'stopwords', 'variance', 'both']
        stopword_ids: Set of token IDs that correspond to stopwords
        alpha: Scaling factor for variance penalty
        
    Returns:
        penalty: A scalar tensor representing the penalty
    """
    penalty = torch.tensor(0.0, device=input_ids.device)

    # If using stopwords mode, penalize attributions on stopwords
    if ig_mode in ['stopwords', 'both']:
        if stopword_ids is not None:
            sw_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for w_id in stopword_ids:
                sw_mask |= (input_ids == w_id)
            penalty += torch.tanh(token_attributions[sw_mask].sum().abs())

    # If using variance mode, penalize high variance of attributions across tokens
    if ig_mode in ['variance', 'both']:
        var_per_example = torch.var(token_attributions, dim=-1)
        var_penalty = torch.tanh(var_per_example.mean() * alpha)
        penalty += var_penalty

    return penalty


class VarianceBasedLoss(nn.Module):
    """
    Loss function that combines cross-entropy with variance-based attribution penalty.
    
    This implements the core loss from the research:
    Total Loss = L_CE + λ * tanh(α * Var_attr)
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.1,
        alpha: float = 10.0,
        ig_mode: str = 'variance',
        stopword_ids: Optional[Set[int]] = None
    ):
        """
        Initialize the variance-based loss.
        
        Args:
            lambda_reg: Regularization strength (λ in the formula)
            alpha: Variance scaling factor (α in the formula) 
            ig_mode: Attribution penalty mode
            stopword_ids: Set of stopword token IDs for stopword penalty
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.ig_mode = ig_mode
        self.stopword_ids = stopword_ids
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_attributions: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the total loss with attribution penalty.
        
        Args:
            logits: Model output logits [batch, num_classes]
            labels: True labels [batch]
            token_attributions: IG attributions [batch, seq_len]
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            total_loss: Combined loss value
        """
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Attribution penalty
        if self.ig_mode != 'none' and self.lambda_reg > 0:
            penalty = compute_attribution_penalty(
                token_attributions=token_attributions,
                input_ids=input_ids,
                ig_mode=self.ig_mode,
                stopword_ids=self.stopword_ids,
                alpha=self.alpha
            )
            total_loss = ce_loss + self.lambda_reg * penalty
        else:
            total_loss = ce_loss
            
        return total_loss
    
    def get_loss_components(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_attributions: torch.Tensor,
        input_ids: torch.Tensor
    ) -> dict:
        """
        Get individual loss components for analysis.
        
        Returns:
            Dictionary with 'ce_loss', 'penalty', and 'total_loss'
        """
        ce_loss = self.ce_loss(logits, labels)
        
        if self.ig_mode != 'none' and self.lambda_reg > 0:
            penalty = compute_attribution_penalty(
                token_attributions=token_attributions,
                input_ids=input_ids,
                ig_mode=self.ig_mode,
                stopword_ids=self.stopword_ids,
                alpha=self.alpha
            )
            total_loss = ce_loss + self.lambda_reg * penalty
        else:
            penalty = torch.tensor(0.0, device=logits.device)
            total_loss = ce_loss
            
        return {
            'ce_loss': ce_loss.item(),
            'penalty': penalty.item(),
            'total_loss': total_loss.item()
        }