# V-BIG: Variance-Based Integrated Gradients for Robust NLI

A Python package implementing Variance-Based Integrated Gradients regularization for training more robust Natural Language Inference (NLI) models.

## Overview

In this repo, I attempted to train NLI models to be less susceptible to dataset artifacts by using attribution-guided regularization. The main idea was to add a variance-based penalty on Integrated Gradients attributions to encourage models to distribute attention more evenly across relevant tokens (models exploiting artifacts tend to have highly concentrated attribution patterns, while robust models distribute attention more evenly across semantically relevant tokens). There are some weird consequences of this in the small-scale tests I ran. See the PDF for a paper-style writeup.

The variance-based regularization loss is:

```
Total Loss = L_CE + λ * tanh(α * Var_attr)
```

Where:
- `L_CE` is the standard cross-entropy loss
- `Var_attr` is the variance of token-level attributions
- `λ` controls regularization strength
- `α` scales the variance for numerical stability

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd V-BIG

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.6+
- Transformers library
- Captum for attribution methods
- Other dependencies listed in `requirements.txt`

## Project Structure

```
V-BIG/
├── src/
│   ├── run.py                 # Main training script for standard NLI/QA models
│   ├── run_ig.py             # Training script with Integrated Gradients regularization
│   ├── helpers.py            # Utility functions for data preprocessing and training
│   ├── create_attributions.py # Script for computing and visualizing attributions
│   └── dataset_artifacts_test.ipynb # Jupyter notebook for experimentation
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── Variance-Based Integrated Gradients Regularization for Robust Natural Language Inference.pdf # Paper
```

## Usage

### Training with Integrated Gradients Regularization

```bash
# Train NLI model with variance-based regularization
python src/run_ig.py \
    --task nli \
    --model google/electra-small-discriminator \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --ig_mode variance \
    --lambda_reg 0.1 \
    --alpha_scale 1.0
```

### Standard Training (without regularization)

```bash
# Train standard NLI model
python src/run.py \
    --task nli \
    --model google/electra-small-discriminator \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3
```

### Computing Attributions

```bash
# Generate attribution visualizations
python src/create_attributions.py \
    --model_path ./output \
    --premise "Two women are embracing." \
    --hypothesis "The sisters are hugging." \
    --output_file attributions.png
```

## Regularization Modes

The `--ig_mode` parameter supports several regularization strategies:

- `none`: No regularization (standard training)
- `variance`: Variance-based penalty on attributions
- `stopwords`: Penalty on stopword attributions
- `both`: Combined variance and stopword penalties

## Hyperparameters

- `--lambda_reg`: Regularization strength (default: 0.1)
- `--alpha_scale`: Variance scaling factor (default: 1.0)
- `--ig_mode`: Regularization mode (default: 'none')

## Datasets

The project supports:
- **NLI**: SNLI, MNLI, and custom JSON datasets
- **QA**: SQuAD and custom question-answering datasets

## Citation

If you use this code in your research, please cite as:

```
@misc{vbig2024,
  title={Variance-Based Integrated Gradients Regularization for Robust Natural Language Inference},
  author={Julian Weaver},
  year={2024}
}
```

## License

This project is licensed under the MIT license.

## Contributing

Feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- Built off of the HuggingFace Transformers library
- Uses Captum for attribution methods