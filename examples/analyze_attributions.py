#!/usr/bin/env python3
"""
Example script for analyzing attributions from trained models.

This script demonstrates how to compute and visualize attributions
for both baseline and V-BIG models, comparing their behavior.
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vbig.attribution import AttributionProcessor, AttributionAnalyzer
from vbig.visualization import visualize_attributions, compare_attributions
from vbig.data import NLIDataProcessor
from vbig.evaluation import ModelComparator


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def analyze_single_example(model, tokenizer, premise, hypothesis, target_label=None):
    """Analyze a single premise-hypothesis pair."""
    processor = AttributionProcessor(tokenizer)
    
    # Get attributions
    attributions, input_ids, predicted_label = processor.get_token_attributions(
        model=model,
        premise=premise,
        hypothesis=hypothesis,
        target_label=target_label,
        n_steps=50
    )
    
    # Decode tokens
    tokens = processor.decode_tokens(input_ids)
    
    # Get model prediction probabilities
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()
    
    return {
        'tokens': tokens,
        'attributions': attributions,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'probabilities': probs.squeeze(0).numpy()
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze model attributions")
    
    # Model arguments
    parser.add_argument("--baseline_model", type=str, required=True,
                       help="Path to baseline model")
    parser.add_argument("--vbig_model", type=str, default=None,
                       help="Path to V-BIG model (optional)")
    
    # Analysis arguments  
    parser.add_argument("--dataset", type=str, default="snli",
                       help="Dataset to analyze")
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to analyze")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    
    # Example-specific analysis
    parser.add_argument("--premise", type=str, default=None,
                       help="Specific premise to analyze")
    parser.add_argument("--hypothesis", type=str, default=None,
                       help="Specific hypothesis to analyze")
    parser.add_argument("--target_label", type=int, default=None,
                       help="Target label for attribution (0=entailment, 1=neutral, 2=contradiction)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load baseline model
    print(f"Loading baseline model from {args.baseline_model}")
    baseline_model, tokenizer = load_model_and_tokenizer(args.baseline_model)
    
    # Load V-BIG model if provided
    vbig_model = None
    if args.vbig_model:
        print(f"Loading V-BIG model from {args.vbig_model}")
        vbig_model, _ = load_model_and_tokenizer(args.vbig_model)
    
    # Analyze specific example if provided
    if args.premise and args.hypothesis:
        print(f"Analyzing specific example:")
        print(f"  Premise: {args.premise}")
        print(f"  Hypothesis: {args.hypothesis}")
        
        # Analyze with baseline model
        baseline_result = analyze_single_example(
            baseline_model, tokenizer, args.premise, args.hypothesis, args.target_label
        )
        
        print(f"\\nBaseline model:")
        print(f"  Predicted label: {baseline_result['predicted_label']}")
        print(f"  Confidence: {baseline_result['confidence']:.4f}")
        
        # Visualize baseline attributions
        fig = visualize_attributions(
            tokens=baseline_result['tokens'],
            attributions=baseline_result['attributions'],
            title="Baseline Model Attributions",
            save_path=os.path.join(args.output_dir, "baseline_attributions.png")
        )
        
        # Analyze with V-BIG model if available
        if vbig_model:
            vbig_result = analyze_single_example(
                vbig_model, tokenizer, args.premise, args.hypothesis, args.target_label
            )
            
            print(f"\\nV-BIG model:")
            print(f"  Predicted label: {vbig_result['predicted_label']}")
            print(f"  Confidence: {vbig_result['confidence']:.4f}")
            
            # Compare attributions
            fig = compare_attributions(
                tokens=baseline_result['tokens'],
                baseline_attributions=baseline_result['attributions'],
                variant_attributions=vbig_result['attributions'],
                title="Attribution Comparison",
                model_names=("Baseline", "V-BIG"),
                save_path=os.path.join(args.output_dir, "attribution_comparison.png")
            )
            
            # Analyze attribution differences
            analyzer = AttributionAnalyzer(tokenizer)
            comparison = analyzer.compare_attributions(
                baseline_result['attributions'],
                vbig_result['attributions'],
                torch.tensor(tokenizer.encode(args.premise + " " + args.hypothesis))
            )
            
            print(f"\\nAttribution comparison:")
            for key, value in comparison.items():
                print(f"  {key}: {value:.4f}")
        
        return
    
    # Dataset-wide analysis
    print(f"Performing dataset-wide analysis on {args.dataset}")
    
    # Load dataset
    data_processor = NLIDataProcessor(tokenizer, filter_invalid_labels=True)
    dataset = data_processor.load_dataset(args.dataset, eval_size=args.num_examples)
    eval_dataset = dataset['validation'].select(range(min(args.num_examples, len(dataset['validation']))))
    
    # Initialize model comparator
    comparator = ModelComparator(tokenizer)
    attribution_processor = AttributionProcessor(tokenizer)
    
    # Analyze baseline model
    print("Analyzing baseline model...")
    baseline_predictions = []
    baseline_probabilities = []
    baseline_attributions = []
    
    for i, example in enumerate(eval_dataset):
        if i % 50 == 0:
            print(f"  Processing example {i}/{len(eval_dataset)}")
        
        premise = example['premise']
        hypothesis = example['hypothesis']
        
        # Get model outputs
        inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = baseline_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        predicted_label = torch.argmax(logits, dim=-1).item()
        baseline_predictions.append(predicted_label)
        baseline_probabilities.append(probs.squeeze(0).numpy())
        
        # Get attributions
        attributions, _, _ = attribution_processor.get_token_attributions(
            model=baseline_model,
            premise=premise,
            hypothesis=hypothesis,
            target_label=predicted_label,
            n_steps=10  # Fewer steps for speed
        )
        baseline_attributions.append(attributions.numpy())
    
    # Add baseline results to comparator
    true_labels = [example['label'] for example in eval_dataset]
    comparator.add_model_results(
        model_name="Baseline",
        predictions=np.array(baseline_predictions),
        true_labels=np.array(true_labels),
        attributions=baseline_attributions,
        probabilities=np.array(baseline_probabilities)
    )
    
    # Analyze V-BIG model if provided
    if vbig_model:
        print("Analyzing V-BIG model...")
        vbig_predictions = []
        vbig_probabilities = []
        vbig_attributions = []
        
        for i, example in enumerate(eval_dataset):
            if i % 50 == 0:
                print(f"  Processing example {i}/{len(eval_dataset)}")
            
            premise = example['premise']
            hypothesis = example['hypothesis']
            
            # Get model outputs
            inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
            with torch.no_grad():
                logits = vbig_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            
            predicted_label = torch.argmax(logits, dim=-1).item()
            vbig_predictions.append(predicted_label)
            vbig_probabilities.append(probs.squeeze(0).numpy())
            
            # Get attributions
            attributions, _, _ = attribution_processor.get_token_attributions(
                model=vbig_model,
                premise=premise,
                hypothesis=hypothesis,
                target_label=predicted_label,
                n_steps=10
            )
            vbig_attributions.append(attributions.numpy())
        
        # Add V-BIG results to comparator
        comparator.add_model_results(
            model_name="V-BIG",
            predictions=np.array(vbig_predictions),
            true_labels=np.array(true_labels),
            attributions=vbig_attributions,
            probabilities=np.array(vbig_probabilities)
        )
    
    # Generate comparison report
    print("Generating analysis report...")
    report = comparator.generate_comparison_report()
    
    # Save report
    with open(os.path.join(args.output_dir, "analysis_report.txt"), "w") as f:
        f.write(report)
    
    print(f"Analysis report saved to {os.path.join(args.output_dir, 'analysis_report.txt')}")
    
    # Save detailed results
    if vbig_model:
        # Attribution analysis
        attr_analysis = comparator.analyze_attribution_differences("Baseline", "V-BIG")
        
        # Find improvement cases
        improved_indices, degraded_indices = comparator.identify_improvement_cases("Baseline", "V-BIG")
        
        detailed_results = {
            "attribution_analysis": attr_analysis,
            "improved_examples": improved_indices[:10],  # Save top 10
            "degraded_examples": degraded_indices[:10],
            "baseline_results": comparator.results["Baseline"].metrics,
            "vbig_results": comparator.results["V-BIG"].metrics
        }
        
        with open(os.path.join(args.output_dir, "detailed_analysis.json"), "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"Detailed analysis saved to {os.path.join(args.output_dir, 'detailed_analysis.json')}")
        
        # Find and analyze some interesting examples
        if improved_indices:
            print(f"\\nAnalyzing an improved example (index {improved_indices[0]}):")
            example = eval_dataset[improved_indices[0]]
            
            baseline_result = analyze_single_example(
                baseline_model, tokenizer, example['premise'], example['hypothesis']
            )
            vbig_result = analyze_single_example(
                vbig_model, tokenizer, example['premise'], example['hypothesis']
            )
            
            print(f"  Premise: {example['premise']}")
            print(f"  Hypothesis: {example['hypothesis']}")
            print(f"  True label: {example['label']}")
            print(f"  Baseline prediction: {baseline_result['predicted_label']} (conf: {baseline_result['confidence']:.3f})")
            print(f"  V-BIG prediction: {vbig_result['predicted_label']} (conf: {vbig_result['confidence']:.3f})")
            
            # Visualize this example
            fig = compare_attributions(
                tokens=baseline_result['tokens'],
                baseline_attributions=baseline_result['attributions'],
                variant_attributions=vbig_result['attributions'],
                title=f"Improved Example (Index {improved_indices[0]})",
                model_names=("Baseline", "V-BIG"),
                save_path=os.path.join(args.output_dir, "improved_example.png")
            )
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()