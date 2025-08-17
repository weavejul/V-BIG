import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from dataclasses import dataclass, field
import evaluate
from helpers import (
    prepare_dataset_nli,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer
)
import os
import json
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from captum.attr import IntegratedGradients
import torch.nn.functional as F

# NLTK stopwords setup
import nltk
from nltk.corpus import stopwords
from typing import Optional

# Attempt to load stopwords; download if not available
try:
    stopwords_list = stopwords.words('english')
except LookupError:
    print("Stopwords resource not found. Downloading now...")
    nltk.download('stopwords')
    stopwords_list = stopwords.words('english')

NUM_PREPROCESSING_WORKERS = 2

def compute_ig_attributions(model, input_ids, attention_mask, labels):
    """
    Compute Integrated Gradients (IG) attributions for the given batch.
    By passing attention_mask as an additional_forward_arg, both embeddings and mask are scaled together.
    Args:
        model: The model to attribute.
        input_ids: Tensor of input token IDs [batch, seq_len].
        attention_mask: Tensor of attention masks [batch, seq_len].
        labels: Tensor of correct labels [batch].
    Returns:
        token_attributions: A [batch, seq_len] tensor containing attributions per token.
    """
    # Get embeddings from input IDs
    embeddings = model.get_input_embeddings()(input_ids)

    # Define a forward function compatible with IG that also takes attention_mask
    def forward_func(emb, mask):
        outputs = model(inputs_embeds=emb, attention_mask=mask)
        return outputs.logits

    # Initialize IG with the updated forward function
    ig = IntegratedGradients(forward_func)

    # Compute attributions for each target in labels
    # Pass attention_mask as additional_forward_args so IG can expand it along with embeddings
    baselines = torch.zeros_like(embeddings)  # Correct keyword is 'baselines'
    attributions = ig.attribute(
        embeddings, 
        baselines=baselines,  # Corrected here
        target=labels, 
        n_steps=10,
        additional_forward_args=(attention_mask,)
    )
    # Sum over embedding dimensions to get a single attribution score per token
    token_attributions = attributions.sum(dim=-1)  # [batch, seq_len]

    return token_attributions

def compute_attribution_penalty(token_attributions, input_ids, tokenizer, ig_mode, stopword_ids):
    """
    Compute an attribution-based penalty given the IG mode.
    Args:
        token_attributions: [batch, seq_len] attribution scores.
        input_ids: [batch, seq_len] token IDs.
        tokenizer: Tokenizer used to encode input.
        ig_mode: One of ['none', 'stopwords', 'variance', 'both'].
        stopword_ids: Set of token IDs that correspond to stopwords.
    Returns:
        penalty: A scalar tensor representing the penalty.
    """
    penalty = torch.tensor(0.0, device=input_ids.device)

    # If using stopwords mode, penalize attributions on stopwords
    if ig_mode in ['stopwords', 'both']:
        sw_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for w_id in stopword_ids:
            sw_mask |= (input_ids == w_id)
        penalty += torch.tanh(token_attributions[sw_mask].sum().abs())

    # If using variance mode, penalize high variance of attributions across tokens
    if ig_mode in ['variance', 'both']:
        var_per_example = torch.var(token_attributions, dim=-1)
        var_penalty = torch.tanh(var_per_example.mean() * 10)
        penalty += var_penalty

    return penalty

class IGTrainer(Trainer):
    """
    A custom Trainer subclass that integrates IG computation into the training loop.
    We override compute_loss to add an IG-based penalty if ig_mode != 'none'.
    """
    def __init__(self, ig_mode='none', processing_class=None, stopword_ids=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ig_mode = ig_mode
        self.processing_class = processing_class
        self.stopword_ids = stopword_ids if stopword_ids is not None else set()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels and ensure they are passed to the model
        labels = inputs.get("labels")
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ['labels']}
        outputs = model(**inputs_for_model, labels=labels)
        loss = outputs.loss

        if self.ig_mode == 'none':
            # No IG-based penalty, just return the normal loss
            return (loss, outputs) if return_outputs else loss

        # Compute IG attributions
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_attributions = compute_ig_attributions(model, input_ids, attention_mask, labels)

        # Compute penalty based on IG mode
        penalty = compute_attribution_penalty(token_attributions, input_ids, self.processing_class, self.ig_mode, self.stopword_ids)

        # Add penalty to the loss
        total_loss = loss + 0.1 * penalty
        return (total_loss, outputs) if return_outputs else total_loss

@dataclass
class MyTrainingArguments(TrainingArguments):
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a folder with a valid checkpoint for resuming training."},
    )

def main():
    # Initialize argument parser with custom TrainingArguments
    parser = HfArgumentParser((MyTrainingArguments, ))
    
    # Add additional arguments
    parser.add_argument('--model', type=str,
                        default='google/electra-small-discriminator',
                        help="Base model to fine-tune")
    parser.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                        help="Task: nli or qa")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Override default dataset")
    parser.add_argument('--max_length', type=int, default=128,
                        help="Max sequence length for inputs")
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help="Limit training examples")
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help="Limit eval examples")
    parser.add_argument('--ig_mode', type=str, choices=['none', 'stopwords', 'variance', 'both'],
                        default='none',
                        help="Which IG-based regularization mode to use")
    
    # Parse arguments
    training_args, args = parser.parse_args_into_dataclasses()

    # Load dataset
    if args.dataset is not None and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset_id = None
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else default_datasets[args.task]
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = datasets.load_dataset(*dataset_id)

    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    model_classes = {
        'qa': AutoModelForQuestionAnswering,
        'nli': AutoModelForSequenceClassification
    }
    model_class = model_classes[args.task]
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Ensure prepare_dataset_nli returns 'labels' field for NLI tasks.
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    else:
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)

    print("Preprocessing data...")
    if dataset_id == ('snli',):
        # Filter out no-label examples for SNLI
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            max_train = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    else:
        train_dataset_featurized = None

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            max_eval = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )
    else:
        eval_dataset_featurized = None

    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        if args.task == 'qa':
            # For QA, use SQuAD metric
            metric = evaluate.load('squad')
            predictions = eval_preds.predictions
            references = eval_preds.label_ids
            return metric.compute(predictions=predictions, references=references)
        else:
            # For NLI, compute accuracy and ROC AUC
            preds = eval_preds.predictions
            y_true = eval_dataset['label']

            # Compute Accuracy
            pred_labels = np.argmax(preds, axis=1)
            accuracy = (pred_labels == y_true).mean()

            # Compute ROC AUC
            probs = np.exp(preds)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            y_score = probs[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
            roc_auc = auc(fpr, tpr)
            return {'accuracy': accuracy, 'roc_auc': roc_auc}

    # Convert stopwords to IDs for IG penalty
    stopword_ids = tokenizer.convert_tokens_to_ids(stopwords_list)
    stopword_ids = set(stopword_ids)

    # Select trainer class and IG mode
    if args.task == 'qa':
        trainer_class = QuestionAnsweringTrainer
        ig_mode = 'none'
    else:
        trainer_class = IGTrainer
        ig_mode = args.ig_mode

    # Initialize trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        ig_mode=ig_mode,
        stopword_ids=stopword_ids,
    )

    # Run training and evaluation
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate()
        print('Evaluation results:')
        print(results)
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f)
        
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')

if __name__ == "__main__":
    main()