import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def forward_func(embeddings, attention_mask, model):
    """
    Custom forward function for Integrated Gradients.
    Accepts embeddings and attention mask, and returns logits from the model.
    """
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    return logits

def get_token_attributions(model, tokenizer, premise, hypothesis, target_label=None):
    """
    Compute Integrated Gradients for a given premise/hypothesis pair.
    """
    # Tokenize input sentences
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Extract embeddings from input IDs
    embeddings = model.get_input_embeddings()(input_ids)

    # Do forward pass to get preds
    with torch.no_grad():
        logits = forward_func(embeddings, attention_mask, model)
    preds = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(preds, dim=1).item()

    # No target label provided? Default to predicted label
    if target_label is None:
        target_label = pred_label

    # Init Integrated Gradients and define baseline
    ig = IntegratedGradients(lambda emb: forward_func(emb, attention_mask, model))
    baseline = torch.zeros_like(embeddings)

    # Compute attributions for target label
    attributions = ig.attribute(embeddings, baseline, target=target_label, n_steps=50)
    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Convert input IDs back to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, token_attributions, pred_label

def plot_token_attributions(tokens, attributions, title="Integrated Gradients Attributions", premise_length=0, 
                            max_width=80, width_per_char=10, base_spacing=0.3, line_spacing=1.0):
    """
    Create a multi-line, color-coded plot for token attributions with line wrapping and extra space.

    Parameters:
        tokens (list): List of tokens to display.
        attributions (list): List of attribution scores for the tokens.
        title (str): Title for the plot.
        premise_length (int): Number of tokens in the premise.
        max_width (int): Maximum number of characters (approx.) per line before wrapping.
        width_per_char (float): Horizontal space allocated per character.
        base_spacing (float): Additional base spacing around each token.
        line_spacing (float): Vertical spacing between lines of tokens.
    """
    # Normalize attribution scores for color mapping
    min_val = np.min(attributions)
    max_val = np.max(attributions)
    if min_val == max_val:
        min_val -= 1e-9
        max_val += 1e-9
    
    norm = Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap("RdYlGn")  # Diverging colormap

    # We'll arrange tokens in lines based on max_width
    lines = []
    current_line = []
    current_line_width = 0

    # We'll also keep track of where the premise ends to possibly mark it
    premise_line_index = None
    premise_token_in_line = None

    for i, tok in enumerate(tokens):
        token_length = len(tok)
        token_width = (token_length * width_per_char) + base_spacing

        # Check if adding this token would exceed the max_width
        if current_line_width + token_length > max_width and current_line:
            # Move to a new line
            lines.append(current_line)
            current_line = []
            current_line_width = 0

        # Add current token to the line
        current_line.append((tok, attributions[i]))
        current_line_width += token_length

        # Record premise/hypothesis boundary if needed
        if i == premise_length - 1:
            premise_line_index = len(lines)  # 0-based line index where premise ends
            premise_token_in_line = len(current_line) - 1  # token index in that line

    # Add the last line if not empty
    if current_line:
        lines.append(current_line)

    fig, ax = plt.subplots(figsize=(14, 3 + len(lines)*1))  # Increase figure height based on number of lines
    fig.suptitle(title, fontsize=14, x=0.5, y=1.05, ha='center')
    ax.axis('off')

    # Plot lines of tokens
    y_start = 0.5  # starting vertical position
    for line_index, line_tokens in enumerate(lines):
        # Compute the total width of the line for centering
        line_total_width = 0
        token_positions = []
        current_x = 0.0

        # First, compute positions for each token in this line
        for (tok, attr) in line_tokens:
            token_length = len(tok)
            token_width = (token_length * width_per_char) + base_spacing
            token_center = current_x + (token_width / 2.0)
            token_positions.append(token_center)
            current_x += token_width
            line_total_width += token_width

        # Center the line horizontally
        line_mid = (token_positions[0] + token_positions[-1]) / 2.0
        x_shift = -line_mid  # shift to center around 0
        y = y_start - (line_index * line_spacing)

        # Draw tokens
        for (tok, attr), x_pos in zip(line_tokens, token_positions):
            color = cmap(norm(attr))
            bbox_props = dict(facecolor=(color[0], color[1], color[2], 0.7),
                              edgecolor='none', boxstyle='round,pad=0.1')
            ax.text(x_pos + x_shift, y, tok,
                    fontsize=12, fontweight='bold',
                    bbox=bbox_props,
                    ha='center', va='center')

        # Draw premise-hypothesis divider if this line contains the boundary
        if premise_length > 0 and premise_length < len(tokens):
            if premise_line_index == line_index and premise_token_in_line is not None:
                # Find position after the premise token
                divider_x = token_positions[premise_token_in_line] + x_shift + (width_per_char / 2.0)
                ax.axvline(x=divider_x, color='gray', linestyle='--', linewidth=2, alpha=0.6)

                # Label premise and hypothesis
                # We'll place the labels above the line
                premise_positions = token_positions[:premise_token_in_line+1]
                hypothesis_positions = token_positions[premise_token_in_line+1:]
                premise_mid = (premise_positions[0] + premise_positions[-1]) / 2.0 + x_shift
                hypothesis_mid = (hypothesis_positions[0] + hypothesis_positions[-1]) / 2.0 + x_shift

                ax.text(premise_mid, y+0.4, "Premise", fontsize=12, fontweight='bold', color="blue", ha='center', va='center')
                ax.text(hypothesis_mid, y+0.4, "Hypothesis", fontsize=12, fontweight='bold', color="green", ha='center', va='center')

    # If no premise length or invalid, just label the whole input
    if not (premise_length > 0 and premise_length < len(tokens)):
        # Label everything as "Input" at the top line
        # Center label based on the first line
        if lines:
            first_line_tokens = lines[0]
            token_positions = []
            current_x = 0.0
            for (tok, attr) in first_line_tokens:
                token_length = len(tok)
                token_width = (token_length * width_per_char) + base_spacing
                token_center = current_x + (token_width / 2.0)
                token_positions.append(token_center)
                current_x += token_width
            line_mid = (token_positions[0] + token_positions[-1]) / 2.0
            ax.text(line_mid - line_mid, y_start+0.4, "Input", fontsize=12, fontweight='bold', color="blue", ha='center', va='center')

    # Determine axis limits
    # Roughly find the max width line
    max_line_width = 0
    for line_tokens in lines:
        current_x = 0.0
        for (tok, attr) in line_tokens:
            token_length = len(tok)
            token_width = (token_length * width_per_char) + base_spacing
            current_x += token_width
        max_line_width = max(max_line_width, current_x)
    
    # Center lines around zero
    left_lim = - (max_line_width / 2.0) - 1.0
    right_lim = (max_line_width / 2.0) + 1.0
    top_line_y = y_start + 1.0
    bottom_line_y = y_start - (len(lines) * line_spacing) - 0.5
    ax.set_xlim(left_lim, right_lim)
    ax.set_ylim(bottom_line_y, top_line_y)

    # Add colorbar below
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.07])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Attribution Score', fontsize=12)

    plt.subplots_adjust(top=0.7, bottom=0.2, left=0.05, right=0.95)
    return fig

def main(args=None, **kwargs):
    """
    Main function to handle token attribution and visualization.
    Accepts either an argparse.Namespace object (args) or keyword arguments.
    """
    # If no args are passed, assume kwargs were used
    if args is None:
        class Args:
            pass
        args = Args()
        # Define default values for all attributes
        defaults = {
            "model_dir": None,
            "premise": "",
            "hypothesis": "",
            "target_label": None,
            "create_vis": False,
            "save_fig": None,
        }
        for key, value in {**defaults, **kwargs}.items():
            setattr(args, key, value)
    
    # Ensure all required arguments are provided
    if not args.model_dir:
        raise ValueError("The 'model_dir' argument is required.")
    if not args.premise:
        raise ValueError("The 'premise' argument is required.")
    if not args.hypothesis:
        raise ValueError("The 'hypothesis' argument is required.")

    # Load tokenizer and model from specified directory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval().to(device)

    # Compute token attributions
    tokens, token_attributions, pred_label = get_token_attributions(
        model, tokenizer, args.premise, args.hypothesis, target_label=args.target_label
    )

    # Print results to console
    print("Predicted Label:", pred_label)
    print("Token Attributions:")
    for t, a in zip(tokens, token_attributions):
        print(f"{t}: {a:.4f}")

    # Create visualization only if --create_vis flag provided
    if args.create_vis:
        # Determine the length of the premise in tokens
        premise_inputs = tokenizer(args.premise, return_tensors='pt', truncation=True)
        premise_length = premise_inputs['input_ids'].shape[1]  # Number of tokens in the premise
        fig = plot_token_attributions(
            tokens, token_attributions, 
            title="Integrated Gradients Attributions",
            premise_length=premise_length
        )
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=300)
            print(f"Figure saved to {args.save_fig}")
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument('--premise', type=str, required=True, help="Premise sentence.")
    parser.add_argument('--hypothesis', type=str, required=True, help="Hypothesis sentence.")
    parser.add_argument('--target_label', type=int, default=None, help="Optional: target label index for attribution.")
    parser.add_argument('--create_vis', action='store_true', help="Flag to create and display/save visualization.")
    parser.add_argument('--save_fig', type=str, default=None, help="Path to save the figure (e.g. 'attributions.png'). If not provided, display it.")
    args = parser.parse_args()

    main(args)