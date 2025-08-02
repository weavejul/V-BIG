"""Attribution analysis and comparison utils."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class AttributionStats:
    """Statistics for a set of attributions."""
    mean: float
    std: float
    variance: float
    max_attr: float
    min_attr: float
    concentration_ratio: float  # Ratio of top 20% vs bottom 80%
    
    def to_dict(self) -> dict:
        return {
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'max_attr': self.max_attr,
            'min_attr': self.min_attr,
            'concentration_ratio': self.concentration_ratio
        }


class AttributionAnalyzer:
    """Analyze and compare attribution patterns between models."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute_attribution_stats(self, attributions: torch.Tensor) -> AttributionStats:
        """
        Compute statistical measures for a set of attributions.
        
        Args:
            attributions: [seq_len] attribution scores
            
        Returns:
            AttributionStats object with computed statistics
        """
        attr_np = attributions.cpu().numpy() if torch.is_tensor(attributions) else attributions
        
        # Basic statistics
        mean_attr = float(np.mean(attr_np))
        std_attr = float(np.std(attr_np))
        var_attr = float(np.var(attr_np))
        max_attr = float(np.max(attr_np))
        min_attr = float(np.min(attr_np))
        
        # Concentration ratio: how much attribution is concentrated in top tokens
        sorted_attr = np.sort(np.abs(attr_np))[::-1]  # Sort descending by absolute value
        n_tokens = len(sorted_attr)
        top_20_pct = int(0.2 * n_tokens)
        
        if top_20_pct > 0:
            top_attr_sum = np.sum(sorted_attr[:top_20_pct])
            bottom_attr_sum = np.sum(sorted_attr[top_20_pct:])
            concentration_ratio = top_attr_sum / (bottom_attr_sum + 1e-8)
        else:
            concentration_ratio = 0.0
            
        return AttributionStats(
            mean=mean_attr,
            std=std_attr,
            variance=var_attr,
            max_attr=max_attr,
            min_attr=min_attr,
            concentration_ratio=concentration_ratio
        )
    
    def compare_attributions(
        self,
        baseline_attributions: torch.Tensor,
        variant_attributions: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compare two sets of attributions.
        
        Args:
            baseline_attributions: [seq_len] baseline model attributions
            variant_attributions: [seq_len] variant model attributions  
            input_ids: [seq_len] token IDs
            
        Returns:
            Dictionary of comparison metrics
        """
        # Convert to numpy
        baseline_np = baseline_attributions.cpu().numpy()
        variant_np = variant_attributions.cpu().numpy()
        
        # Correlation between attributions
        correlation = np.corrcoef(baseline_np, variant_np)[0, 1]
        
        # L2 difference
        l2_diff = np.sqrt(np.mean((baseline_np - variant_np) ** 2))
        
        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(baseline_np, variant_np)
        
        # Attribution shift: how much the "important" tokens changed
        baseline_top_indices = set(np.argsort(np.abs(baseline_np))[-5:])
        variant_top_indices = set(np.argsort(np.abs(variant_np))[-5:])
        overlap_ratio = len(baseline_top_indices & variant_top_indices) / 5.0
        
        # Variance comparison
        baseline_var = np.var(baseline_np)
        variant_var = np.var(variant_np)
        variance_ratio = variant_var / (baseline_var + 1e-8)
        
        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'l2_difference': float(l2_diff),
            'rank_correlation': float(rank_corr) if not np.isnan(rank_corr) else 0.0,
            'top_token_overlap': float(overlap_ratio),
            'variance_ratio': float(variance_ratio),
            'baseline_variance': float(baseline_var),
            'variant_variance': float(variant_var)
        }
    
    def analyze_prediction_attribution_alignment(
        self,
        attributions: torch.Tensor,
        input_ids: torch.Tensor,
        prediction_confidence: float,
        true_label: int,
        predicted_label: int
    ) -> Dict[str, float]:
        """
        Analyze how well attributions align with prediction confidence.
        
        Args:
            attributions: [seq_len] attribution scores
            input_ids: [seq_len] token IDs
            prediction_confidence: Model's confidence in its prediction
            true_label: Ground truth label
            predicted_label: Model's predicted label
            
        Returns:
            Dictionary of alignment metrics
        """
        attr_np = attributions.cpu().numpy()
        
        # Attribution magnitude vs confidence
        total_attr_magnitude = np.sum(np.abs(attr_np))
        attr_confidence_ratio = total_attr_magnitude / (prediction_confidence + 1e-8)
        
        # Attribution consistency (low variance suggests consistent reasoning)
        attribution_consistency = 1.0 / (np.var(attr_np) + 1e-8)
        
        # Prediction correctness
        is_correct = int(true_label == predicted_label)
        
        return {
            'attribution_magnitude': float(total_attr_magnitude),
            'attr_confidence_ratio': float(attr_confidence_ratio),
            'attribution_consistency': float(attribution_consistency),
            'prediction_confidence': float(prediction_confidence),
            'is_correct': is_correct,
            'confidence_calibration': float(prediction_confidence) if is_correct else float(1.0 - prediction_confidence)
        }
    
    def find_problematic_examples(
        self,
        attributions_list: List[torch.Tensor],
        input_ids_list: List[torch.Tensor],
        predictions: List[int],
        true_labels: List[int],
        confidences: List[float],
        threshold_variance: float = 0.1
    ) -> List[int]:
        """
        Find examples where the model might be relying on spurious correlations.
        
        High variance + high confidence + wrong prediction = likely spurious
        
        Args:
            attributions_list: List of attribution tensors
            input_ids_list: List of input ID tensors
            predictions: List of predicted labels
            true_labels: List of true labels
            confidences: List of prediction confidences
            threshold_variance: Variance threshold for flagging examples
            
        Returns:
            List of indices of problematic examples
        """
        problematic_indices = []
        
        for i, (attributions, input_ids, pred, true_label, conf) in enumerate(
            zip(attributions_list, input_ids_list, predictions, true_labels, confidences)
        ):
            stats = self.compute_attribution_stats(attributions)
            
            # High variance, high confidence, wrong prediction
            is_high_variance = stats.variance > threshold_variance
            is_high_confidence = conf > 0.8
            is_wrong = pred != true_label
            
            if is_high_variance and is_high_confidence and is_wrong:
                problematic_indices.append(i)
                
        return problematic_indices
    
    def export_attribution_analysis(
        self,
        analysis_results: Dict,
        output_path: str
    ) -> None:
        """Export attribution analysis results to JSON."""
        # Convert torch tensors to lists for JSON serialization
        def convert_for_json(obj):
            if torch.is_tensor(obj):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_for_json(obj)
        
        converted_results = recursive_convert(analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)