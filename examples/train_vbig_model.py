#!/usr/bin/env python3
"""
Example script for training a V-BIG model on the SNLI dataset.

This script demonstrates how to use the vbig package to train a model
with variance-based integrated gradients regularization.
"""

import argparse
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from vbig import IGTrainer, VarianceTrainingArguments
from vbig.data import NLIDataProcessor, create_stopword_ids
from vbig.evaluation import compute_nli_metrics
from vbig.training.callbacks import AttributionLoggingCallback


def main():
    parser = argparse.ArgumentParser(description="Train V-BIG model on NLI task")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="google/electra-small-discriminator",
                       help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="snli", choices=["snli", "multi_nli"],
                       help="Dataset to use for training")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum number of evaluation samples")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    
    # V-BIG specific arguments
    parser.add_argument("--ig_mode", type=str, default="variance", 
                       choices=["none", "variance", "stopwords", "both"],
                       help="IG regularization mode")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                       help="Regularization strength")
    parser.add_argument("--alpha_variance", type=float, default=10.0,
                       help="Variance scaling factor")
    parser.add_argument("--ig_steps", type=int, default=10,
                       help="Number of IG integration steps")
    
    # Experiment tracking
    parser.add_argument("--run_name", type=str, default=None,
                       help="Name for this experimental run")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=3  # SNLI has 3 classes
    )
    
    # Initialize data processor
    print(f"Loading dataset: {args.dataset}")
    data_processor = NLIDataProcessor(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        filter_invalid_labels=True
    )
    
    # Load and preprocess dataset
    dataset = data_processor.load_dataset(
        dataset_name=args.dataset,
        train_size=args.max_train_samples,
        eval_size=args.max_eval_samples
    )
    
    # Preprocess datasets
    train_dataset = data_processor.preprocess_dataset(dataset['train'])
    eval_dataset = data_processor.preprocess_dataset(dataset['validation'])
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Create stopword IDs for stopword penalty (if needed)
    stopword_ids = create_stopword_ids(tokenizer) if args.ig_mode in ["stopwords", "both"] else None
    
    # Set up training arguments
    training_args = VarianceTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
        # V-BIG specific parameters
        ig_mode=args.ig_mode,
        lambda_reg=args.lambda_reg,
        alpha_variance=args.alpha_variance,
        ig_steps=args.ig_steps,
    )
    
    # Initialize trainer
    trainer = IGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        ig_mode=args.ig_mode,
        lambda_reg=args.lambda_reg,
        alpha_variance=args.alpha_variance,
        ig_steps=args.ig_steps,
        stopword_ids=stopword_ids,
        compute_metrics=lambda eval_pred: compute_nli_metrics(
            eval_pred.predictions, eval_pred.label_ids
        ),
        callbacks=[AttributionLoggingCallback(log_frequency=100, use_wandb=args.use_wandb)]
    )
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Print loss summary
    if hasattr(trainer, 'get_loss_summary'):
        loss_summary = trainer.get_loss_summary()
        print("\nTraining loss summary:")
        for key, value in loss_summary.items():
            print(f"  {key}: {value:.6f}")
    
    # Save training configuration
    import json
    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "ig_mode": args.ig_mode,
        "lambda_reg": args.lambda_reg,
        "alpha_variance": args.alpha_variance,
        "ig_steps": args.ig_steps,
        "training_time": training_time,
        "eval_results": eval_results,
    }
    
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to {os.path.join(args.output_dir, 'training_config.json')}")


if __name__ == "__main__":
    main()