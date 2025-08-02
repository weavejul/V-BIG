"""Utility functions for data processing."""

import nltk
from nltk.corpus import stopwords
from typing import Set, List
from transformers import PreTrainedTokenizer


def download_stopwords():
    """Download NLTK stopwords if not already available."""
    try:
        stopwords.words('english')
    except LookupError:
        print("Stopwords resource not found. Downloading now...")
        nltk.download('stopwords')


def create_stopword_ids(tokenizer: PreTrainedTokenizer, language: str = 'english') -> Set[int]:
    """
    Create a set of token IDs corresponding to stopwords.
    
    Args:
        tokenizer: HuggingFace tokenizer
        language: Language for stopwords (default: 'english')
        
    Returns:
        Set of stopword token IDs
    """
    # Ensure stopwords are downloaded
    download_stopwords()
    
    # Get stopwords for the specified language
    stopwords_list = stopwords.words(language)
    
    # Convert to token IDs
    stopword_ids = set()
    for word in stopwords_list:
        # Tokenize the word (it might be split into subwords)
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        stopword_ids.update(ids)
    
    return stopword_ids


def filter_valid_labels(dataset, valid_labels: List[int] = [0, 1, 2]):
    """
    Filter dataset to only include examples with valid labels.
    
    Args:
        dataset: HuggingFace dataset
        valid_labels: List of valid label values
        
    Returns:
        Filtered dataset
    """
    return dataset.filter(lambda example: example['label'] in valid_labels)


def compute_class_weights(labels: List[int]) -> dict:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary mapping labels to weights
    """
    from collections import Counter
    import numpy as np
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    weights = {}
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    
    return weights


def create_balanced_subset(dataset, max_per_class: int = 1000):
    """
    Create a balanced subset with equal numbers of examples per class.
    
    Args:
        dataset: HuggingFace dataset with 'label' column
        max_per_class: Maximum number of examples per class
        
    Returns:
        Balanced subset dataset
    """
    from collections import defaultdict
    import random
    
    # Group examples by label
    examples_by_label = defaultdict(list)
    for i, example in enumerate(dataset):
        examples_by_label[example['label']].append(i)
    
    # Sample equal numbers from each class
    selected_indices = []
    for label, indices in examples_by_label.items():
        # Randomly sample up to max_per_class examples
        num_to_sample = min(len(indices), max_per_class)
        sampled_indices = random.sample(indices, num_to_sample)
        selected_indices.extend(sampled_indices)
    
    # Create balanced subset
    balanced_dataset = dataset.select(selected_indices)
    
    return balanced_dataset


def analyze_text_complexity(text: str) -> dict:
    """
    Analyze text complexity metrics.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary of complexity metrics
    """
    words = text.split()
    sentences = text.split('.')
    
    metrics = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'unique_words': len(set(word.lower() for word in words)),
        'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
    }
    
    return metrics