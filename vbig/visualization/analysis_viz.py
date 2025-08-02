"""Analysis and metrics visualization utils."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_training_curves(
    training_history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training curves including loss components.
    
    Args:
        training_history: Dictionary with training metrics over time
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot total loss
    if 'total_loss' in training_history:
        axes[0, 0].plot(training_history['total_loss'], label='Total Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss components
    if 'ce_loss' in training_history and 'penalty' in training_history:
        axes[0, 1].plot(training_history['ce_loss'], label='Cross-Entropy Loss', color='blue')
        axes[0, 1].plot(training_history['penalty'], label='Attribution Penalty', color='orange')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot accuracy if available
    if 'accuracy' in training_history:
        axes[1, 0].plot(training_history['accuracy'], label='Accuracy', color='green')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot penalty ratio
    if 'penalty_ratio' in training_history:
        axes[1, 1].plot(training_history['penalty_ratio'], label='Penalty Ratio', color='purple')
        axes[1, 1].set_title('Penalty Contribution Ratio')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Penalty / Total Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix with proper formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (default: numeric)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    # Formatting
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot calibration curve showing reliability diagram.
    
    Args:
        y_true: True labels (binary)
        y_prob: Prediction probabilities
        n_bins: Number of bins for calibration
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Compute calibration curve
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_conf_in_bin = y_prob[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            bin_centers.append(avg_conf_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(count_in_bin)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(bin_centers, bin_accuracies, 'o-', label='Model')
    
    # Add error bars based on bin size
    if bin_counts:
        errors = [np.sqrt(acc * (1 - acc) / count) if count > 0 else 0 
                 for acc, count in zip(bin_accuracies, bin_counts)]
        ax1.errorbar(bin_centers, bin_accuracies, yerr=errors, fmt='o-', alpha=0.7)
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, density=True)
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison between multiple models across different metrics.
    
    Args:
        comparison_results: Dictionary mapping metric names to model scores
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(comparison_results).T
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    metrics = list(comparison_results.keys())
    
    for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
        ax = axes[i]
        
        model_names = list(comparison_results[metric].keys())
        scores = list(comparison_results[metric].values())
        
        # Bar plot
        bars = ax.bar(model_names, scores, alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{score:.4f}', ha='center', va='bottom')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 8:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_analysis(
    error_analysis_results: Dict[str, Any],
    title: str = "Error Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot error analysis results including error rates by different factors.
    
    Args:
        error_analysis_results: Dictionary with error analysis data
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Error rate by length
    if 'length_bins' in error_analysis_results:
        ax = axes[0, 0]
        bins = error_analysis_results['length_bins']
        error_rates = error_analysis_results['error_rate_by_length']
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
        
        ax.plot(bin_centers, error_rates, 'o-', color='red')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rate by Sequence Length')
        ax.grid(True, alpha=0.3)
    
    # Calibration analysis
    if 'bin_accuracies' in error_analysis_results:
        ax = axes[0, 1]
        bin_conf = error_analysis_results['bin_confidences']
        bin_acc = error_analysis_results['bin_accuracies']
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.plot(bin_conf, bin_acc, 'o-', color='blue', label='Model')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Confidence distribution
    if 'confidence_analysis' in error_analysis_results:
        ax = axes[1, 0]
        conf_analysis = error_analysis_results['confidence_analysis']
        
        thresholds = []
        accuracies = []
        
        for threshold_key, data in conf_analysis.items():
            if 'confidence' in threshold_key:
                threshold = float(threshold_key.split('_')[-1])
                thresholds.append(threshold)
                accuracies.append(data['accuracy'])
        
        if thresholds:
            ax.plot(thresholds, accuracies, 'o-', color='green')
            ax.set_xlabel('Confidence Threshold')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy vs Confidence Threshold')
            ax.grid(True, alpha=0.3)
    
    # Attribution variance distribution
    if 'attribution_variances' in error_analysis_results:
        ax = axes[1, 1]
        variances = error_analysis_results['attribution_variances']
        
        ax.hist(variances, bins=30, alpha=0.7, color='purple')
        ax.set_xlabel('Attribution Variance')
        ax.set_ylabel('Frequency')
        ax.set_title('Attribution Variance Distribution')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attribution_stability(
    stability_metrics: Dict[str, List[float]],
    title: str = "Attribution Stability Over Training",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot attribution stability metrics over training steps.
    
    Args:
        stability_metrics: Dictionary with lists of stability metrics over time
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric_name, values in stability_metrics.items():
        if values:  # Only plot if data exists
            ax.plot(values, label=metric_name.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Training Step / Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig