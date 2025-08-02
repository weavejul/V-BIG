"""Analysis utils for model comparison and performance evaluation."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from ..attribution.core import AttributionProcessor
from .metrics import compute_detailed_metrics, compute_attribution_consistency_metrics


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    attributions: List[np.ndarray]
    metrics: Dict[str, Any]
    runtime: float


class ModelComparator:
    """
    Compare multiple models on various metrics and attribution patterns.
    
    Useful for comparing baseline vs V-BIG models.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.attribution_processor = AttributionProcessor(tokenizer)
        self.results = {}
        
    def add_model_results(
        self,
        model_name: str,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        attributions: List[np.ndarray],
        probabilities: Optional[np.ndarray] = None,
        runtime: float = 0.0
    ):
        """
        Add results for a model to the comparison.
        
        Args:
            model_name: Name/identifier for the model
            predictions: Model predictions
            true_labels: Ground truth labels
            attributions: List of attribution arrays
            probabilities: Prediction probabilities (optional)
            runtime: Model runtime in seconds
        """
        # Compute comprehensive metrics
        metrics = compute_detailed_metrics(predictions, true_labels, 
                                         np.max(probabilities, axis=1) if probabilities is not None else None)
        
        # Add attribution-specific metrics
        if attributions:
            attr_metrics = compute_attribution_consistency_metrics(attributions, predictions, true_labels)
            metrics.update(attr_metrics)
        
        # Store results
        self.results[model_name] = ModelResults(
            model_name=model_name,
            predictions=predictions,
            probabilities=probabilities,
            attributions=attributions,
            metrics=metrics,
            runtime=runtime
        )
    
    def compare_metrics(self, metrics_to_compare: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare specific metrics across models.
        
        Args:
            metrics_to_compare: List of metric names to compare
            
        Returns:
            Dictionary mapping metric names to model scores
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['accuracy', 'roc_auc', 'macro_f1', 'expected_calibration_error']
        
        comparison = {}
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name, results in self.results.items():
                if metric in results.metrics:
                    comparison[metric][model_name] = results.metrics[metric]
        
        return comparison
    
    def analyze_attribution_differences(
        self,
        model1_name: str,
        model2_name: str,
        example_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze differences in attribution patterns between two models.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model  
            example_indices: Specific examples to analyze (default: all)
            
        Returns:
            Dictionary of attribution comparison results
        """
        if model1_name not in self.results or model2_name not in self.results:
            raise ValueError("Both models must be added to the comparator first")
        
        model1_attrs = self.results[model1_name].attributions
        model2_attrs = self.results[model2_name].attributions
        
        if example_indices is None:
            example_indices = list(range(len(model1_attrs)))
        
        # Compare attributions for specified examples
        correlations = []
        l2_differences = []
        variance_changes = []
        
        for idx in example_indices:
            if idx < len(model1_attrs) and idx < len(model2_attrs):
                attr1 = model1_attrs[idx]
                attr2 = model2_attrs[idx]
                
                # Correlation
                corr = np.corrcoef(attr1, attr2)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
                
                # L2 difference
                l2_diff = np.sqrt(np.mean((attr1 - attr2) ** 2))
                l2_differences.append(l2_diff)
                
                # Variance change
                var1 = np.var(attr1)
                var2 = np.var(attr2)
                variance_changes.append(var2 - var1)
        
        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'mean_l2_difference': np.mean(l2_differences),
            'mean_variance_change': np.mean(variance_changes),
            'variance_reduction_rate': np.mean([vc < 0 for vc in variance_changes])
        }
    
    def identify_improvement_cases(
        self,
        baseline_model: str,
        comparison_model: str,
        min_improvement: float = 0.1
    ) -> Tuple[List[int], List[int]]:
        """
        Identify examples where one model significantly outperforms another.
        
        Args:
            baseline_model: Name of baseline model
            comparison_model: Name of comparison model
            min_improvement: Minimum confidence improvement to be considered significant
            
        Returns:
            Tuple of (improved_indices, degraded_indices)
        """
        baseline_results = self.results[baseline_model]
        comparison_results = self.results[comparison_model]
        
        # Get prediction confidences
        baseline_conf = np.max(baseline_results.probabilities, axis=1) if baseline_results.probabilities is not None else None
        comparison_conf = np.max(comparison_results.probabilities, axis=1) if comparison_results.probabilities is not None else None
        
        improved_indices = []
        degraded_indices = []
        
        for i in range(len(baseline_results.predictions)):
            baseline_pred = baseline_results.predictions[i]
            comparison_pred = comparison_results.predictions[i]
            
            # Check if comparison model improved or degraded
            if baseline_conf is not None and comparison_conf is not None:
                conf_improvement = comparison_conf[i] - baseline_conf[i]
                
                # Improved: higher confidence and/or correct when baseline was wrong
                if conf_improvement > min_improvement:
                    improved_indices.append(i)
                elif conf_improvement < -min_improvement:
                    degraded_indices.append(i)
        
        return improved_indices, degraded_indices
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive text report comparing all models.
        
        Returns:
            Formatted comparison report
        """
        if not self.results:
            return "No model results available for comparison."
        
        report = "# Model Comparison Report\n\n"
        
        # Overall metrics comparison
        comparison = self.compare_metrics()
        
        report += "## Performance Metrics\n\n"
        for metric, scores in comparison.items():
            report += f"### {metric.replace('_', ' ').title()}\n"
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for model, score in sorted_models:
                report += f"- **{model}**: {score:.4f}\n"
            report += "\n"
        
        # Runtime comparison
        report += "## Runtime Comparison\n\n"
        for model_name, results in self.results.items():
            report += f"- **{model_name}**: {results.runtime:.2f}s\n"
        
        # Attribution analysis (if available)
        if len(self.results) >= 2:
            model_names = list(self.results.keys())
            if len(model_names) >= 2:
                attr_analysis = self.analyze_attribution_differences(model_names[0], model_names[1])
                report += f"\n## Attribution Analysis ({model_names[0]} vs {model_names[1]})\n\n"
                report += f"- **Mean Correlation**: {attr_analysis['mean_correlation']:.4f}\n"
                report += f"- **Mean L2 Difference**: {attr_analysis['mean_l2_difference']:.4f}\n"
                report += f"- **Mean Variance Change**: {attr_analysis['mean_variance_change']:.4f}\n"
                report += f"- **Variance Reduction Rate**: {attr_analysis['variance_reduction_rate']:.4f}\n"
        
        return report


class PerformanceAnalyzer:
    """
    Analyze model performance in detail, including error patterns and attribution quality.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def analyze_errors_by_length(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        input_lengths: List[int],
        length_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze how error rates change with input sequence length.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            input_lengths: Length of each input sequence
            length_bins: Number of bins to divide lengths into
            
        Returns:
            Dictionary of length-based analysis results
        """
        # Convert predictions to labels if needed
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions
        
        # Create length bins
        min_length, max_length = min(input_lengths), max(input_lengths)
        bin_edges = np.linspace(min_length, max_length, length_bins + 1)
        
        results = {
            'length_bins': bin_edges.tolist(),
            'accuracy_by_length': [],
            'error_rate_by_length': [],
            'count_by_length': []
        }
        
        for i in range(length_bins):
            # Find examples in this length bin
            in_bin = (np.array(input_lengths) >= bin_edges[i]) & (np.array(input_lengths) < bin_edges[i + 1])
            
            if i == length_bins - 1:  # Include the maximum value in the last bin
                in_bin = (np.array(input_lengths) >= bin_edges[i]) & (np.array(input_lengths) <= bin_edges[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(pred_labels[in_bin] == true_labels[in_bin])
                bin_error_rate = 1 - bin_accuracy
                bin_count = np.sum(in_bin)
            else:
                bin_accuracy = 0.0
                bin_error_rate = 0.0
                bin_count = 0
            
            results['accuracy_by_length'].append(bin_accuracy)
            results['error_rate_by_length'].append(bin_error_rate)
            results['count_by_length'].append(int(bin_count))
        
        return results
    
    def analyze_confidence_calibration(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Detailed analysis of model calibration across confidence levels.
        
        Args:
            probabilities: Model prediction probabilities
            true_labels: Ground truth labels
            n_bins: Number of confidence bins
            
        Returns:
            Dictionary of calibration analysis results
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Create confidence bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        results = {
            'bin_edges': bin_edges.tolist(),
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': [],
            'calibration_errors': []
        }
        
        for i in range(n_bins):
            # Find examples in this confidence bin
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            
            if i == n_bins - 1:  # Include 1.0 in the last bin
                in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(predictions[in_bin] == true_labels[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_count = np.sum(in_bin)
                calibration_error = abs(bin_confidence - bin_accuracy)
            else:
                bin_accuracy = 0.0
                bin_confidence = 0.0
                bin_count = 0
                calibration_error = 0.0
            
            results['bin_accuracies'].append(bin_accuracy)
            results['bin_confidences'].append(bin_confidence)
            results['bin_counts'].append(int(bin_count))
            results['calibration_errors'].append(calibration_error)
        
        # Overall calibration metrics
        results['expected_calibration_error'] = np.average(
            results['calibration_errors'],
            weights=results['bin_counts']
        )
        results['maximum_calibration_error'] = max(results['calibration_errors'])
        
        return results
    
    def find_high_attribution_variance_examples(
        self,
        attributions: List[np.ndarray],
        predictions: np.ndarray,
        true_labels: np.ndarray,
        variance_threshold: float = 0.1,
        top_k: int = 10
    ) -> List[Tuple[int, float, bool]]:
        """
        Find examples with highest attribution variance.
        
        Args:
            attributions: List of attribution arrays
            predictions: Model predictions
            true_labels: Ground truth labels
            variance_threshold: Minimum variance to consider
            top_k: Number of top examples to return
            
        Returns:
            List of (example_index, variance, is_correct) tuples
        """
        # Convert predictions to labels if needed
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions
        
        # Compute variance for each example
        variances = []
        for i, attrs in enumerate(attributions):
            variance = np.var(attrs)
            is_correct = pred_labels[i] == true_labels[i]
            variances.append((i, variance, is_correct))
        
        # Filter by threshold and sort
        high_variance = [(i, v, c) for i, v, c in variances if v >= variance_threshold]
        high_variance.sort(key=lambda x: x[1], reverse=True)
        
        return high_variance[:top_k]