"""Evaluation metrics for NLI models."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List, Optional, Tuple
import torch


def compute_nli_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute standard NLI evaluation metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: True labels
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert logits to predicted labels
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
        # Convert to probabilities for AUC computation
        if predictions.shape[1] > 2:
            # Multi-class: use one-vs-rest AUC
            probs = softmax(predictions)
            auc_scores = []
            for class_idx in range(predictions.shape[1]):
                if class_idx in labels:  # Only compute AUC for classes present in labels
                    binary_labels = (labels == class_idx).astype(int)
                    try:
                        auc = roc_auc_score(binary_labels, probs[:, class_idx])
                        auc_scores.append(auc)
                    except ValueError:
                        continue  # Skip if only one class present
            roc_auc = np.mean(auc_scores) if auc_scores else 0.0
        else:
            # Binary classification
            probs = softmax(predictions)[:, 1]
            try:
                roc_auc = roc_auc_score(labels, probs)
            except ValueError:
                roc_auc = 0.0
    else:
        pred_labels = predictions
        roc_auc = 0.0  # Cannot compute AUC without probabilities
    
    # Compute metrics
    accuracy = accuracy_score(labels, pred_labels)
    
    # Precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred_labels, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist()
    }


def compute_detailed_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Compute detailed evaluation metrics including calibration and error analysis.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: True labels
        confidences: Prediction confidences (max probability)
        
    Returns:
        Dictionary of detailed metrics
    """
    metrics = compute_nli_metrics(predictions, labels)
    
    # Convert to predicted labels and probabilities
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
        probs = softmax(predictions)
        if confidences is None:
            confidences = np.max(probs, axis=1)
    else:
        pred_labels = predictions
        probs = None
    
    # Confusion matrix
    cm = confusion_matrix(labels, pred_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    metrics['accuracy_per_class'] = class_accuracies.tolist()
    
    # Calibration metrics (if probabilities available)
    if probs is not None and confidences is not None:
        calibration_metrics = compute_calibration_metrics(labels, pred_labels, confidences)
        metrics.update(calibration_metrics)
    
    # Error analysis by confidence
    if confidences is not None:
        error_analysis = analyze_errors_by_confidence(labels, pred_labels, confidences)
        metrics.update(error_analysis)
    
    return metrics


def compute_calibration_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE, etc.).
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        confidences: Prediction confidences
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = (true_labels[in_bin] == pred_labels[in_bin]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Calibration error for this bin
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            # Expected Calibration Error
            ece += prop_in_bin * calibration_error
            
            # Maximum Calibration Error
            mce = max(mce, calibration_error)
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce)
    }


def analyze_errors_by_confidence(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    confidence_thresholds: List[float] = [0.5, 0.7, 0.9]
) -> Dict[str, Dict]:
    """
    Analyze error patterns by confidence level.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        confidences: Prediction confidences
        confidence_thresholds: Thresholds for confidence analysis
        
    Returns:
        Dictionary of error analysis results
    """
    results = {}
    
    for threshold in confidence_thresholds:
        high_conf_mask = confidences >= threshold
        
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = (
                true_labels[high_conf_mask] == pred_labels[high_conf_mask]
            ).mean()
            
            results[f'accuracy_at_confidence_{threshold}'] = {
                'accuracy': float(high_conf_accuracy),
                'num_examples': int(high_conf_mask.sum()),
                'fraction_of_total': float(high_conf_mask.mean())
            }
    
    return {'confidence_analysis': results}


def compute_attribution_consistency_metrics(
    attributions_list: List[np.ndarray],
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics related to attribution consistency and reliability.
    
    Args:
        attributions_list: List of attribution arrays for each example
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Dictionary of attribution consistency metrics
    """
    # Convert predictions to labels if needed
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions
    
    # Compute attribution variance for each example
    attribution_variances = []
    attribution_concentrations = []
    
    for attrs in attributions_list:
        # Variance of attributions
        var = np.var(attrs)
        attribution_variances.append(var)
        
        # Concentration (top 20% vs bottom 80%)
        sorted_attrs = np.sort(np.abs(attrs))[::-1]
        n_tokens = len(sorted_attrs)
        top_20_pct = int(0.2 * n_tokens)
        
        if top_20_pct > 0:
            top_sum = np.sum(sorted_attrs[:top_20_pct])
            bottom_sum = np.sum(sorted_attrs[top_20_pct:])
            concentration = top_sum / (bottom_sum + 1e-8)
        else:
            concentration = 0.0
            
        attribution_concentrations.append(concentration)
    
    # Compute correlations with prediction correctness
    is_correct = (pred_labels == labels).astype(float)
    
    # Correlation between attribution variance and correctness
    var_corr = np.corrcoef(attribution_variances, is_correct)[0, 1]
    conc_corr = np.corrcoef(attribution_concentrations, is_correct)[0, 1]
    
    return {
        'mean_attribution_variance': float(np.mean(attribution_variances)),
        'std_attribution_variance': float(np.std(attribution_variances)),
        'mean_attribution_concentration': float(np.mean(attribution_concentrations)),
        'variance_correctness_correlation': float(var_corr) if not np.isnan(var_corr) else 0.0,
        'concentration_correctness_correlation': float(conc_corr) if not np.isnan(conc_corr) else 0.0
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)