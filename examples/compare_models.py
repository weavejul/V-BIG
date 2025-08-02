#!/usr/bin/env python3
"""
Example script for comparing baseline and V-BIG models systematically.

This script runs a comprehensive comparison including performance metrics,
attribution analysis, and generates detailed reports and visualizations.
"""

import argparse
import os
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vbig.evaluation import ModelComparator, PerformanceAnalyzer
from vbig.attribution import AttributionProcessor
from vbig.data import NLIDataProcessor
from vbig.visualization import plot_model_comparison, plot_attribution_statistics


def evaluate_model_on_dataset(model, tokenizer, dataset, attribution_processor):
    """Evaluate a model on a dataset and collect comprehensive results."""
    predictions = []
    probabilities = []
    attributions = []
    true_labels = []
    runtimes = []
    
    print(f"Evaluating model on {len(dataset)} examples...")
    
    for i, example in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        premise = example['premise']
        hypothesis = example['hypothesis']
        true_label = example['label']
        true_labels.append(true_label)
        
        # Time the inference
        start_time = time.time()
        
        # Get model prediction
        inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        predicted_label = torch.argmax(logits, dim=-1).item()
        predictions.append(predicted_label)
        probabilities.append(probs.squeeze(0).numpy())
        
        # Get attributions
        attrs, _, _ = attribution_processor.get_token_attributions(
            model=model,
            premise=premise,
            hypothesis=hypothesis,
            target_label=predicted_label,
            n_steps=10
        )
        attributions.append(attrs.numpy())
        
        inference_time = time.time() - start_time
        runtimes.append(inference_time)
    
    return {
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities),
        'attributions': attributions,
        'true_labels': np.array(true_labels),
        'avg_runtime': np.mean(runtimes)
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and V-BIG models")
    
    # Model paths
    parser.add_argument("--baseline_model", type=str, required=True,
                       help="Path to baseline model")
    parser.add_argument("--vbig_model", type=str, required=True,
                       help="Path to V-BIG model")
    
    # Evaluation settings
    parser.add_argument("--dataset", type=str, default="snli",
                       help="Dataset to use for evaluation")
    parser.add_argument("--num_examples", type=int, default=1000,
                       help="Number of examples to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for comparison results")
    
    # Analysis options
    parser.add_argument("--detailed_analysis", action="store_true",
                       help="Perform detailed error and calibration analysis")
    parser.add_argument("--save_examples", action="store_true",
                       help="Save interesting examples with visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    baseline_model, tokenizer = load_model_and_tokenizer(args.baseline_model)
    vbig_model, _ = load_model_and_tokenizer(args.vbig_model)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    data_processor = NLIDataProcessor(tokenizer, filter_invalid_labels=True)
    dataset = data_processor.load_dataset(args.dataset, eval_size=args.num_examples)
    eval_dataset = dataset['validation'].select(range(min(args.num_examples, len(dataset['validation']))))
    
    # Initialize processors
    attribution_processor = AttributionProcessor(tokenizer)
    comparator = ModelComparator(tokenizer)
    
    # Evaluate baseline model
    print("\\nEvaluating baseline model...")
    baseline_results = evaluate_model_on_dataset(
        baseline_model, tokenizer, eval_dataset, attribution_processor
    )
    
    comparator.add_model_results(
        model_name="Baseline",
        predictions=baseline_results['predictions'],
        true_labels=baseline_results['true_labels'],
        attributions=baseline_results['attributions'],
        probabilities=baseline_results['probabilities'],
        runtime=baseline_results['avg_runtime']
    )
    
    # Evaluate V-BIG model
    print("\\nEvaluating V-BIG model...")
    vbig_results = evaluate_model_on_dataset(
        vbig_model, tokenizer, eval_dataset, attribution_processor
    )
    
    comparator.add_model_results(
        model_name="V-BIG",
        predictions=vbig_results['predictions'],
        true_labels=vbig_results['true_labels'],
        attributions=vbig_results['attributions'],
        probabilities=vbig_results['probabilities'],
        runtime=vbig_results['avg_runtime']
    )
    
    # Generate comparison report
    print("\\nGenerating comparison report...")
    report = comparator.generate_comparison_report()
    
    with open(os.path.join(args.output_dir, "model_comparison_report.txt"), "w") as f:
        f.write(report)
    
    # Generate comparison metrics
    comparison_metrics = comparator.compare_metrics()
    
    # Save detailed metrics
    with open(os.path.join(args.output_dir, "comparison_metrics.json"), "w") as f:
        json.dump(comparison_metrics, f, indent=2, default=str)
    
    # Plot model comparison
    fig = plot_model_comparison(
        comparison_metrics,
        title="Baseline vs V-BIG Model Comparison",
        save_path=os.path.join(args.output_dir, "model_comparison.png")
    )
    
    # Plot attribution statistics
    attribution_dict = {
        "Baseline": baseline_results['attributions'],
        "V-BIG": vbig_results['attributions']
    }
    
    fig = plot_attribution_statistics(
        attribution_dict,
        save_path=os.path.join(args.output_dir, "attribution_statistics.png")
    )
    
    # Attribution analysis
    attr_analysis = comparator.analyze_attribution_differences("Baseline", "V-BIG")
    
    print("\\nAttribution Analysis:")
    print(f"  Mean correlation: {attr_analysis['mean_correlation']:.4f}")
    print(f"  Mean L2 difference: {attr_analysis['mean_l2_difference']:.4f}")
    print(f"  Mean variance change: {attr_analysis['mean_variance_change']:.4f}")
    print(f"  Variance reduction rate: {attr_analysis['variance_reduction_rate']:.4f}")
    
    # Find interesting examples
    improved_indices, degraded_indices = comparator.identify_improvement_cases("Baseline", "V-BIG")
    
    print(f"\\nExample Analysis:")
    print(f"  Examples where V-BIG improved: {len(improved_indices)}")
    print(f"  Examples where V-BIG degraded: {len(degraded_indices)}")
    
    # Detailed analysis if requested
    if args.detailed_analysis:
        print("\\nPerforming detailed analysis...")
        analyzer = PerformanceAnalyzer(tokenizer)
        
        # Length-based analysis
        input_lengths = [len(tokenizer.encode(ex['premise'] + " " + ex['hypothesis'])) 
                        for ex in eval_dataset]
        
        baseline_length_analysis = analyzer.analyze_errors_by_length(
            baseline_results['predictions'],
            baseline_results['true_labels'],
            input_lengths
        )
        
        vbig_length_analysis = analyzer.analyze_errors_by_length(
            vbig_results['predictions'],
            vbig_results['true_labels'],
            input_lengths
        )
        
        # Calibration analysis
        baseline_calibration = analyzer.analyze_confidence_calibration(
            baseline_results['probabilities'],
            baseline_results['true_labels']
        )
        
        vbig_calibration = analyzer.analyze_confidence_calibration(
            vbig_results['probabilities'],
            vbig_results['true_labels']
        )
        
        # Save detailed analysis
        detailed_analysis = {
            "attribution_analysis": attr_analysis,
            "improved_examples": improved_indices[:20],
            "degraded_examples": degraded_indices[:20],
            "baseline_length_analysis": baseline_length_analysis,
            "vbig_length_analysis": vbig_length_analysis,
            "baseline_calibration": baseline_calibration,
            "vbig_calibration": vbig_calibration
        }
        
        with open(os.path.join(args.output_dir, "detailed_analysis.json"), "w") as f:
            json.dump(detailed_analysis, f, indent=2, default=str)
    
    # Save interesting examples if requested
    if args.save_examples and improved_indices:
        from vbig.visualization import compare_attributions
        
        print("\\nSaving example visualizations...")
        
        # Save top 5 improved examples
        for i, idx in enumerate(improved_indices[:5]):
            example = eval_dataset[idx]
            
            # Get attributions for both models
            baseline_attrs, baseline_tokens, baseline_pred = attribution_processor.get_token_attributions(
                baseline_model, example['premise'], example['hypothesis']
            )
            
            vbig_attrs, vbig_tokens, vbig_pred = attribution_processor.get_token_attributions(
                vbig_model, example['premise'], example['hypothesis']
            )
            
            # Create comparison visualization
            fig = compare_attributions(
                tokens=baseline_tokens,
                baseline_attributions=baseline_attrs,
                variant_attributions=vbig_attrs,
                title=f"Improved Example {i+1}\\nPremise: {example['premise'][:50]}...",
                model_names=("Baseline", "V-BIG"),
                save_path=os.path.join(args.output_dir, f"improved_example_{i+1}.png")
            )
            
            # Save example details
            example_details = {
                "premise": example['premise'],
                "hypothesis": example['hypothesis'],
                "true_label": example['label'],
                "baseline_prediction": baseline_pred,
                "vbig_prediction": vbig_pred,
                "baseline_confidence": float(np.max(baseline_results['probabilities'][idx])),
                "vbig_confidence": float(np.max(vbig_results['probabilities'][idx]))
            }
            
            with open(os.path.join(args.output_dir, f"improved_example_{i+1}.json"), "w") as f:
                json.dump(example_details, f, indent=2)
    
    # Summary statistics
    summary = {
        "dataset": args.dataset,
        "num_examples": len(eval_dataset),
        "baseline_accuracy": float(comparator.results["Baseline"].metrics["accuracy"]),
        "vbig_accuracy": float(comparator.results["V-BIG"].metrics["accuracy"]),
        "accuracy_improvement": float(comparator.results["V-BIG"].metrics["accuracy"] - 
                                    comparator.results["Baseline"].metrics["accuracy"]),
        "baseline_runtime": baseline_results['avg_runtime'],
        "vbig_runtime": vbig_results['avg_runtime'],
        "runtime_overhead": vbig_results['avg_runtime'] / baseline_results['avg_runtime'],
        "attribution_correlation": attr_analysis['mean_correlation'],
        "variance_reduction_rate": attr_analysis['variance_reduction_rate'],
        "num_improved_examples": len(improved_indices),
        "num_degraded_examples": len(degraded_indices)
    }
    
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nComparison completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\\nSummary:")
    print(f"  Baseline accuracy: {summary['baseline_accuracy']:.4f}")
    print(f"  V-BIG accuracy: {summary['vbig_accuracy']:.4f}")
    print(f"  Accuracy improvement: {summary['accuracy_improvement']:.4f}")
    print(f"  Runtime overhead: {summary['runtime_overhead']:.2f}x")
    print(f"  Attribution correlation: {summary['attribution_correlation']:.4f}")
    print(f"  Variance reduction rate: {summary['variance_reduction_rate']:.4f}")


def load_model_and_tokenizer(model_path):
    """Helper function to load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    main()