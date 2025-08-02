"""Data preprocessing utils for NLI tasks."""

import numpy as np
from typing import Dict, Any, Optional, Set, List
from transformers import PreTrainedTokenizer
import datasets
from dataclasses import dataclass


def prepare_dataset_nli(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: Optional[int] = None
) -> Dict[str, List]:
    """
    Preprocess an NLI dataset, tokenizing premises and hypotheses.
    
    Args:
        examples: Batch of examples with 'premise', 'hypothesis', and 'label' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (default: tokenizer's max length)
        
    Returns:
        Dictionary with tokenized inputs and labels
    """
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


@dataclass
class NLIExample:
    """Represents a single NLI example."""
    premise: str
    hypothesis: str
    label: int
    guid: Optional[str] = None


class NLIDataProcessor:
    """
    Processor for NLI datasets with enhanced functionality for V-BIG training.
    
    Handles data loading, preprocessing, filtering, and stopword identification.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        filter_invalid_labels: bool = True
    ):
        """
        Initialize the NLI data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length for tokenization
            filter_invalid_labels: Whether to filter out examples with invalid labels
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.filter_invalid_labels = filter_invalid_labels
        
    def load_dataset(
        self,
        dataset_name: str = 'snli',
        train_size: Optional[int] = None,
        eval_size: Optional[int] = None,
        cache_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """
        Load and preprocess an NLI dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            train_size: Maximum number of training examples
            eval_size: Maximum number of evaluation examples
            cache_dir: Directory for caching downloaded datasets
            
        Returns:
            Processed dataset dictionary
        """
        # Load raw dataset
        if dataset_name == 'snli':
            dataset = datasets.load_dataset('snli', cache_dir=cache_dir)
        elif dataset_name == 'multi_nli':
            dataset = datasets.load_dataset('multi_nli', cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Filter invalid labels if requested
        if self.filter_invalid_labels:
            dataset = dataset.filter(lambda example: example['label'] != -1)
        
        # Limit dataset sizes if specified
        if train_size is not None and 'train' in dataset:
            train_size = min(len(dataset['train']), train_size)
            dataset['train'] = dataset['train'].select(range(train_size))
            
        if eval_size is not None and 'validation' in dataset:
            eval_size = min(len(dataset['validation']), eval_size)
            dataset['validation'] = dataset['validation'].select(range(eval_size))
        
        return dataset
    
    def preprocess_dataset(
        self,
        dataset: datasets.Dataset,
        remove_columns: Optional[List[str]] = None
    ) -> datasets.Dataset:
        """
        Apply tokenization and preprocessing to a dataset.
        
        Args:
            dataset: Raw dataset to preprocess
            remove_columns: Columns to remove after preprocessing
            
        Returns:
            Preprocessed dataset
        """
        # Default columns to remove
        if remove_columns is None:
            remove_columns = ['premise', 'hypothesis']
            
        # Apply preprocessing
        processed_dataset = dataset.map(
            lambda examples: prepare_dataset_nli(
                examples, 
                self.tokenizer, 
                self.max_seq_length
            ),
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing dataset"
        )
        
        return processed_dataset
    
    def create_stopword_ids(self, stopwords: List[str]) -> Set[int]:
        """
        Convert stopwords to token IDs.
        
        Args:
            stopwords: List of stopword strings
            
        Returns:
            Set of token IDs corresponding to stopwords
        """
        stopword_ids = []
        for word in stopwords:
            # Tokenize each stopword and get IDs
            tokens = self.tokenizer.tokenize(word)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            stopword_ids.extend(ids)
        
        return set(stopword_ids)
    
    def analyze_dataset_statistics(self, dataset: datasets.Dataset) -> Dict[str, Any]:
        """
        Analyze dataset statistics for debugging and monitoring.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Label distribution
        if 'label' in dataset.column_names:
            labels = dataset['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            stats['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            stats['num_classes'] = len(unique_labels)
        
        # Sequence length statistics
        if 'input_ids' in dataset.column_names:
            lengths = [len(seq) for seq in dataset['input_ids']]
            stats['sequence_lengths'] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            }
        
        # Dataset size
        stats['num_examples'] = len(dataset)
        
        return stats
    
    def split_by_complexity(
        self,
        dataset: datasets.Dataset,
        complexity_threshold: int = 64
    ) -> Dict[str, datasets.Dataset]:
        """
        Split dataset by example complexity (sequence length).
        
        Args:
            dataset: Dataset to split
            complexity_threshold: Threshold for defining "complex" examples
            
        Returns:
            Dictionary with 'simple' and 'complex' dataset splits
        """
        if 'input_ids' not in dataset.column_names:
            raise ValueError("Dataset must be tokenized before complexity splitting")
        
        # Calculate sequence lengths (excluding padding)
        def get_length(example):
            # Count non-padding tokens
            length = sum(1 for token_id in example['input_ids'] if token_id != self.tokenizer.pad_token_id)
            return {'length': length}
        
        dataset_with_lengths = dataset.map(get_length)
        
        # Split by complexity
        simple_dataset = dataset_with_lengths.filter(lambda x: x['length'] <= complexity_threshold)
        complex_dataset = dataset_with_lengths.filter(lambda x: x['length'] > complexity_threshold)
        
        # Remove the length column
        simple_dataset = simple_dataset.remove_columns(['length'])
        complex_dataset = complex_dataset.remove_columns(['length'])
        
        return {
            'simple': simple_dataset,
            'complex': complex_dataset
        }
    
    def create_challenging_subset(
        self,
        dataset: datasets.Dataset,
        fraction: float = 0.1
    ) -> datasets.Dataset:
        """
        Create a subset of challenging examples for evaluation.
        
        Selects examples that are longer or have specific patterns that
        might reveal dataset artifacts.
        
        Args:
            dataset: Source dataset
            fraction: Fraction of examples to include in challenging subset
            
        Returns:
            Challenging subset dataset
        """
        if 'input_ids' not in dataset.column_names:
            raise ValueError("Dataset must be tokenized before creating challenging subset")
        
        # Calculate complexity scores
        def score_complexity(example):
            # Simple heuristic: longer sequences + presence of certain patterns
            length = sum(1 for token_id in example['input_ids'] if token_id != self.tokenizer.pad_token_id)
            
            # Decode to check for potential artifact patterns
            text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            
            # Penalize common artifact patterns (customize based on dataset)
            artifact_penalty = 0
            artifact_words = ['not', 'no', 'never', 'nobody', 'nothing']
            for word in artifact_words:
                if word in text.lower():
                    artifact_penalty += 1
            
            # Combined complexity score
            complexity = length + artifact_penalty * 5
            return {'complexity': complexity}
        
        dataset_with_scores = dataset.map(score_complexity)
        
        # Select top fraction by complexity
        sorted_dataset = dataset_with_scores.sort('complexity', reverse=True)
        num_challenging = int(len(sorted_dataset) * fraction)
        challenging_subset = sorted_dataset.select(range(num_challenging))
        
        # Remove the complexity column
        challenging_subset = challenging_subset.remove_columns(['complexity'])
        
        return challenging_subset