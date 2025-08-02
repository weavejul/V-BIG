#!/usr/bin/env python3
"""
Migration script to help transition from the old flat file structure to the new package structure.

This script provides utilities to:
1. Migrate existing training scripts to use the new vbig package
2. Convert old attribution analysis code
3. Provide examples of using the new API
"""

import os
import shutil
import argparse
from pathlib import Path


def migrate_old_files():
    """Show users what files from the old structure map to the new package."""
    
    migration_map = {
        "run_ig.py": [
            "vbig/attribution/core.py (compute_ig_attributions)",
            "vbig/attribution/losses.py (compute_attribution_penalty)",
            "vbig/training/trainer.py (IGTrainer)",
            "vbig/data/processors.py (prepare_dataset_nli)"
        ],
        "create_attributions.py": [
            "vbig/attribution/core.py (AttributionProcessor.get_token_attributions)",
            "vbig/visualization/attribution_viz.py (visualize_attributions)"
        ],
        "helpers.py": [
            "vbig/data/processors.py (prepare_dataset_nli)",
            "vbig/evaluation/metrics.py (compute_accuracy -> compute_nli_metrics)"
        ],
        "run.py": [
            "examples/train_vbig_model.py (updated to use new package)",
            "vbig/training/trainer.py (standard training without IG)"
        ]
    }
    
    print("Migration Guide: Old Files -> New Package Structure")
    print("=" * 60)
    
    for old_file, new_locations in migration_map.items():
        print(f"\\n{old_file}:")
        for location in new_locations:
            print(f"  -> {location}")
    
    print("\\n" + "=" * 60)
    print("Key Changes:")
    print("1. Attribution computation is now in vbig.attribution.core")
    print("2. Training logic is in vbig.training.trainer") 
    print("3. Data processing is in vbig.data.processors")
    print("4. Visualization functions are in vbig.visualization")
    print("5. Example scripts are in examples/ directory")


def show_api_migration():
    """Show examples of how old API calls map to new ones."""
    
    print("\\nAPI Migration Examples:")
    print("=" * 30)
    
    print("\\n1. Computing IG Attributions:")
    print("OLD:")
    print("  from run_ig import compute_ig_attributions")
    print("  attributions = compute_ig_attributions(model, input_ids, attention_mask, labels)")
    
    print("NEW:")
    print("  from vbig.attribution import compute_ig_attributions")  
    print("  attributions = compute_ig_attributions(model, input_ids, attention_mask, labels)")
    
    print("\\n2. Training with IG Regularization:")
    print("OLD:")
    print("  from run_ig import IGTrainer")
    print("  trainer = IGTrainer(ig_mode='variance', ...)")
    
    print("NEW:")
    print("  from vbig import IGTrainer, VarianceTrainingArguments")
    print("  training_args = VarianceTrainingArguments(ig_mode='variance', ...)")
    print("  trainer = IGTrainer(args=training_args, ...)")
    
    print("\\n3. Data Processing:")
    print("OLD:")
    print("  from helpers import prepare_dataset_nli")
    print("  dataset = dataset.map(lambda x: prepare_dataset_nli(x, tokenizer))")
    
    print("NEW:")
    print("  from vbig.data import NLIDataProcessor")
    print("  processor = NLIDataProcessor(tokenizer)")
    print("  dataset = processor.preprocess_dataset(dataset)")
    
    print("\\n4. Visualization:")
    print("OLD:")
    print("  # Custom matplotlib code in create_attributions.py")
    
    print("NEW:")
    print("  from vbig.visualization import visualize_attributions")
    print("  fig = visualize_attributions(tokens, attributions)")


def create_migration_example():
    """Create an example showing how to migrate old training code."""
    
    old_training_code = '''
# OLD TRAINING CODE (run_ig.py style)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from run_ig import IGTrainer, compute_ig_attributions
from helpers import prepare_dataset_nli
import datasets

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)

# Load data
dataset = datasets.load_dataset("snli")
train_dataset = dataset['train'].map(lambda x: prepare_dataset_nli(x, tokenizer), batched=True)

# Train
training_args = TrainingArguments(output_dir="./output")
trainer = IGTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    ig_mode='variance'
)
trainer.train()
'''

    new_training_code = '''
# NEW TRAINING CODE (using vbig package)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vbig import IGTrainer, VarianceTrainingArguments
from vbig.data import NLIDataProcessor

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)

# Load and process data
data_processor = NLIDataProcessor(tokenizer)
dataset = data_processor.load_dataset("snli")
train_dataset = data_processor.preprocess_dataset(dataset['train'])

# Train with V-BIG regularization
training_args = VarianceTrainingArguments(
    output_dir="./output",
    ig_mode='variance',
    lambda_reg=0.1,
    alpha_variance=10.0
)
trainer = IGTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
'''

    migration_example_path = "migration_example.py"
    with open(migration_example_path, "w") as f:
        f.write("# Migration Example: Old vs New API\\n")
        f.write('"""\\n')
        f.write("This file shows how to migrate from the old flat structure to the new vbig package.\\n")
        f.write('"""\\n\\n')
        f.write("# " + "="*50 + "\\n")
        f.write("# OLD CODE (don't use - for reference only)\\n") 
        f.write("# " + "="*50 + "\\n")
        f.write('"""\\n')
        f.write(old_training_code)
        f.write('"""\\n\\n')
        f.write("# " + "="*50 + "\\n")
        f.write("# NEW CODE (recommended)\\n")
        f.write("# " + "="*50 + "\\n")
        f.write(new_training_code)
    
    print(f"\\nMigration example saved to: {migration_example_path}")


def main():
    parser = argparse.ArgumentParser(description="V-BIG Migration Helper")
    parser.add_argument("--show-mapping", action="store_true", 
                       help="Show how old files map to new package structure")
    parser.add_argument("--show-api", action="store_true",
                       help="Show API migration examples")
    parser.add_argument("--create-example", action="store_true",
                       help="Create migration example file")
    parser.add_argument("--all", action="store_true",
                       help="Show all migration information")
    
    args = parser.parse_args()
    
    if args.all or args.show_mapping:
        migrate_old_files()
    
    if args.all or args.show_api:
        show_api_migration()
    
    if args.all or args.create_example:
        create_migration_example()
    
    if not any([args.show_mapping, args.show_api, args.create_example, args.all]):
        print("V-BIG Migration Helper")
        print("Use --help to see available options")
        print("Quick start: python migrate_from_old_structure.py --all")


if __name__ == "__main__":
    main()