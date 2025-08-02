# V-BIG: Variance-Based Integrated Gradients for Robust NLI

A Python package implementing Variance-Based Integrated Gradients regularization for training more robust Natural Language Inference (NLI) models.

## Abstract

Pre-trained language models can achieve high accuracy on standard natural language inference (NLI) tasks while still relying heavily on dataset artifacts rather than learning genuine semantic reasoning. In this work, I employ Integrated Gradients (IG) to examine the attributions of an ELECTRA-small model’s predictions on the Stanford Natural Language Inference (SNLI) dataset, uncovering its dependence on superficial patterns. To mitigate this issue, I introduce a variance-based penalty on token-level IG attributions, encouraging the model to distribute importance more evenly across relevant tokens and thereby reduce its reliance on spurious correlations. Quantitatively, this Integrated Gradients Variance model shows modest improvements in accuracy and ROC AUC over a baseline model. Qualitatively, it demonstrates more stable, context-sensitive attributions, better handling of complex examples, and improved calibration. Although the performance gains are incremental, these findings show the potential of attribution-guided regularization to foster more reliable, reasoning-oriented NLI models.

### Features

- **Attribution-guided training**: Uses Integrated Gradients to compute token-level attributions during training
- **Variance-based regularization**: Penalizes high attribution variance to reduce spurious correlations
- **Comprehensive evaluation**: Tools for analyzing model behavior and attribution patterns
- **Visualizations**: Built-in plotting functions for attribution analysis

## Installation

### From source

```bash
git clone https://github.com/weavejul/V-BIG.git
cd V-BIG
pip install -e .
```

### Dependencies

The package requires Python 3.8+ and the following dependencies:
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- Captum >= 0.5.0
- Datasets >= 2.0.0

See `requirements.txt` for the complete list.

## Quick Start

### Training a V-BIG Model

```python
from vbig import IGTrainer, VarianceTrainingArguments
from vbig.data import NLIDataProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator", num_labels=3
)

# Prepare data
data_processor = NLIDataProcessor(tokenizer)
dataset = data_processor.load_dataset("snli")
train_dataset = data_processor.preprocess_dataset(dataset['train'])
eval_dataset = data_processor.preprocess_dataset(dataset['validation'])

# Configure training with V-BIG regularization
training_args = VarianceTrainingArguments(
    output_dir="./vbig-model",
    num_train_epochs=3,
    ig_mode="variance",
    lambda_reg=0.1,
    alpha_variance=10.0,
)

# Train with IG regularization
trainer = IGTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Analyzing Attributions

```python
from vbig.attribution import AttributionProcessor
from vbig.visualization import visualize_attributions

# Create attribution processor
processor = AttributionProcessor(tokenizer)

# Compute attributions for an example
attributions, tokens, predicted_label = processor.get_token_attributions(
    model=model,
    premise="A woman is writing something on a post-it note.",
    hypothesis="The woman is writing a grocery list."
)

# Visualize the attributions
fig = visualize_attributions(
    tokens=processor.decode_tokens(tokens),
    attributions=attributions,
    title="Token Attributions"
)
```

### Comparing Models

```python
from vbig.evaluation import ModelComparator

# Compare baseline and V-BIG models
comparator = ModelComparator(tokenizer)

# Add results for both models
comparator.add_model_results("Baseline", baseline_predictions, true_labels, baseline_attributions)
comparator.add_model_results("V-BIG", vbig_predictions, true_labels, vbig_attributions)

# Generate comparison report
report = comparator.generate_comparison_report()
print(report)
```

## Command Line Interface

The code provides cli tools:

```bash
# Train a V-BIG model
vbig-train --output_dir ./my-vbig-model --ig_mode variance --lambda_reg 0.1

# Analyze attributions
vbig-analyze --baseline_model ./baseline --vbig_model ./vbig --output_dir ./analysis

# Compare models comprehensively
vbig-compare --baseline_model ./baseline --vbig_model ./vbig --output_dir ./comparison
```

## Package Structure

```
vbig/
├── attribution/          # Attribution computation and analysis
│   ├── core.py          # Core IG computation
│   ├── losses.py        # Variance-based loss functions
│   └── analyzer.py      # Attribution analysis utilities
├── training/            # Training components
│   ├── trainer.py       # Custom trainer with IG regularization
│   └── callbacks.py     # Training callbacks
├── data/                # Data processing utilities
│   ├── processors.py    # NLI data preprocessing
│   └── utils.py         # Utility functions
├── evaluation/          # Evaluation and metrics
│   ├── metrics.py       # Evaluation metrics
│   └── analysis.py      # Performance analysis
└── visualization/       # Plotting and visualization
    ├── attribution_viz.py    # Attribution visualizations
    └── analysis_viz.py       # Analysis plots
```

## Research Background

This implementation is based on research into making NLI models more robust by reducing their reliance on spurious correlations and dataset artifacts. The main idea is that models exploiting artifacts tend to have highly concentrated attribution patterns, while robust models tend to distribute attention more evenly across semantically relevant tokens.

The variance-based regularization loss is:

```
Total Loss = L_CE + λ * tanh(α * Var_attr)
```

Where:
- `L_CE` is the standard cross-entropy loss
- `Var_attr` is the variance of token-level attributions
- `λ` controls regularization strength
- `α` scales the variance for numerical stability

## Examples

See the `examples/` directory for scripts demonstrating:

- **Training**: `train_vbig_model.py` - Training pipeline
- **Analysis**: `analyze_attributions.py` - Attribution analysis and visualization
- **Comparison**: `compare_models.py` - Model comparisons

## Citation

If you use this code, please cite:

```bibtex
@article{weaver2024vbig,
  title={Variance-Based Integrated Gradients Regularization for Robust Natural Language Inference},
  author={Weaver, Julian},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull Request.