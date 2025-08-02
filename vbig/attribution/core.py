"""Attribution computation using Integrated Gradients."""

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from typing import Optional, Union, Tuple
import numpy as np


def compute_ig_attributions(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    n_steps: int = 10,
    baseline: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Integrated Gradients (IG) attributions for the given batch.
    
    Args:
        model: The model to attribute
        input_ids: Tensor of input token IDs [batch, seq_len]
        attention_mask: Tensor of attention masks [batch, seq_len]  
        labels: Tensor of correct labels [batch]
        n_steps: Number of integration steps (default: 10)
        baseline: Custom baseline embeddings (default: zeros)
        
    Returns:
        token_attributions: A [batch, seq_len] tensor containing attributions per token
    """
    # Get embeddings from input IDs
    embeddings = model.get_input_embeddings()(input_ids)

    # Define a forward function compatible with IG that also takes attention_mask
    def forward_func(emb, mask):
        outputs = model(inputs_embeds=emb, attention_mask=mask)
        return outputs.logits

    # Initialize IG with the forward function
    ig = IntegratedGradients(forward_func)

    # Use zero baseline if none provided
    if baseline is None:
        baseline = torch.zeros_like(embeddings)

    # Compute attributions for each target in labels
    attributions = ig.attribute(
        embeddings, 
        baselines=baseline,
        target=labels, 
        n_steps=n_steps,
        additional_forward_args=(attention_mask,)
    )
    
    # Sum over embedding dimensions to get a single attribution score per token
    token_attributions = attributions.sum(dim=-1)  # [batch, seq_len]

    return token_attributions


class AttributionProcessor:
    """Utility class for processing and analyzing attributions."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def get_token_attributions(
        self,
        model,
        premise: str,
        hypothesis: str,
        target_label: Optional[int] = None,
        n_steps: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute IG attributions for a single premise/hypothesis pair.
        
        Args:
            model: The model to analyze
            premise: Premise text
            hypothesis: Hypothesis text  
            target_label: Target label for attribution (default: predicted label)
            n_steps: Number of integration steps
            
        Returns:
            token_attributions: Attribution scores per token
            tokens: Token IDs
            predicted_label: Model's predicted label
        """
        # Tokenize input sentences
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Move to model device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Get model prediction
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Use predicted label if no target specified
        if target_label is None:
            target_label = predicted_label

        # Compute attributions
        labels = torch.tensor([target_label], device=device)
        token_attributions = compute_ig_attributions(
            model, input_ids, attention_mask, labels, n_steps=n_steps
        )
        
        return token_attributions.squeeze(0), input_ids.squeeze(0), predicted_label
    
    def decode_tokens(self, input_ids: torch.Tensor) -> list:
        """Decode token IDs to readable tokens."""
        return self.tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())
    
    def aggregate_subword_attributions(
        self, 
        token_attributions: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> Tuple[list, list]:
        """
        Aggregate subword token attributions to word level.
        
        Args:
            token_attributions: Attribution scores per token
            input_ids: Token IDs
            
        Returns:
            words: List of words
            word_attributions: List of aggregated attribution scores
        """
        tokens = self.decode_tokens(input_ids)
        attributions = token_attributions.cpu().numpy()
        
        words = []
        word_attributions = []
        current_word = ""
        current_attribution = 0.0
        
        for token, attr in zip(tokens, attributions):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                if current_word:
                    words.append(current_word)
                    word_attributions.append(current_attribution)
                    current_word = ""
                    current_attribution = 0.0
                continue
                
            # Handle subword tokens (starting with ##)
            if token.startswith('##'):
                current_word += token[2:]
                current_attribution += attr
            else:
                # New word - save previous if exists
                if current_word:
                    words.append(current_word)
                    word_attributions.append(current_attribution)
                
                current_word = token
                current_attribution = attr
        
        # Add final word
        if current_word:
            words.append(current_word)
            word_attributions.append(current_attribution)
            
        return words, word_attributions