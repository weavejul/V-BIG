"""Attribution visualization utils."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
import seaborn as sns


def visualize_attributions(
    tokens: List[str],
    attributions: Union[torch.Tensor, np.ndarray],
    premise_length: Optional[int] = None,
    title: str = "Token Attributions",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colormap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Visualize token attributions with color-coded importance.
    
    Args:
        tokens: List of token strings
        attributions: Attribution scores for each token
        premise_length: Length of premise (to separate premise/hypothesis)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        colormap: Matplotlib colormap name
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if torch.is_tensor(attributions):
        attributions = attributions.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize attributions for coloring
    attr_max = max(abs(attributions.min()), abs(attributions.max()))
    norm = Normalize(vmin=-attr_max, vmax=attr_max)
    cmap = plt.get_cmap(colormap)
    
    # Plot tokens with background colors representing attributions
    y_pos = 0.5
    x_start = 0.05
    x_spacing = 0.12
    
    for i, (token, attr) in enumerate(zip(tokens, attributions)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if token == '[SEP]' and premise_length is None:
                premise_length = i  # Auto-detect premise length
            continue
            
        x_pos = x_start + (i * x_spacing)
        
        # Create colored background rectangle
        color = cmap(norm(attr))
        rect = patches.Rectangle(
            (x_pos - 0.05, y_pos - 0.1), 0.1, 0.2,
            linewidth=0, facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add token text
        ax.text(x_pos, y_pos, token, ha='center', va='center', 
                fontsize=10, weight='bold')
        
        # Add attribution value below
        ax.text(x_pos, y_pos - 0.25, f'{attr:.3f}', ha='center', va='center',
                fontsize=8, style='italic')
    
    # Add premise/hypothesis separator if known
    if premise_length is not None:
        sep_x = x_start + premise_length * x_spacing
        ax.axvline(x=sep_x, color='black', linestyle='--', alpha=0.5)
        ax.text(sep_x, y_pos + 0.3, 'Premise | Hypothesis', ha='center', va='center',
                fontsize=12, weight='bold')
    
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                       shrink=0.6, aspect=30, pad=0.1)
    cbar.set_label('Attribution Score', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_attributions(
    tokens: List[str],
    baseline_attributions: Union[torch.Tensor, np.ndarray],
    variant_attributions: Union[torch.Tensor, np.ndarray],
    premise_length: Optional[int] = None,
    title: str = "Attribution Comparison",
    model_names: Tuple[str, str] = ("Baseline", "V-BIG"),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Compare attributions between two models side by side.
    
    Args:
        tokens: List of token strings
        baseline_attributions: Attribution scores for baseline model
        variant_attributions: Attribution scores for variant model
        premise_length: Length of premise
        title: Plot title
        model_names: Names of the two models being compared
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensors
    if torch.is_tensor(baseline_attributions):
        baseline_attributions = baseline_attributions.cpu().numpy()
    if torch.is_tensor(variant_attributions):
        variant_attributions = variant_attributions.cpu().numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                        gridspec_kw={'height_ratios': [1, 1, 0.8]})
    
    # Get unified color scale
    all_attrs = np.concatenate([baseline_attributions, variant_attributions])
    attr_max = max(abs(all_attrs.min()), abs(all_attrs.max()))
    norm = Normalize(vmin=-attr_max, vmax=attr_max)
    cmap = plt.get_cmap('RdBu_r')
    
    # Plot baseline attributions
    _plot_attribution_row(ax1, tokens, baseline_attributions, norm, cmap, 
                          f"{model_names[0]} Attributions", premise_length)
    
    # Plot variant attributions
    _plot_attribution_row(ax2, tokens, variant_attributions, norm, cmap,
                          f"{model_names[1]} Attributions", premise_length)
    
    # Plot difference (variant - baseline)
    attribution_diff = variant_attributions - baseline_attributions
    diff_max = max(abs(attribution_diff.min()), abs(attribution_diff.max()))
    diff_norm = Normalize(vmin=-diff_max, vmax=diff_max)
    _plot_attribution_row(ax3, tokens, attribution_diff, diff_norm, cmap,
                          f"Difference ({model_names[1]} - {model_names[0]})", premise_length)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', 
                       shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Attribution Score', fontsize=12)
    
    # Add difference colorbar
    sm_diff = plt.cm.ScalarMappable(cmap=cmap, norm=diff_norm)
    sm_diff.set_array([])
    cbar_diff = fig.colorbar(sm_diff, ax=ax3, orientation='vertical',
                            shrink=0.8, aspect=30, pad=0.02)
    cbar_diff.set_label('Attribution Difference', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_attribution_row(
    ax: plt.Axes,
    tokens: List[str],
    attributions: np.ndarray,
    norm: Normalize,
    cmap,
    title: str,
    premise_length: Optional[int] = None
):
    """Helper function to plot a single row of attributions."""
    y_pos = 0.5
    x_start = 0.05
    x_spacing = min(0.9 / len(tokens), 0.12)  # Adjust spacing based on number of tokens
    
    for i, (token, attr) in enumerate(zip(tokens, attributions)):
        # Skip special tokens for positioning but track for premise_length
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if token == '[SEP]' and premise_length is None:
                premise_length = i
            continue
            
        x_pos = x_start + (i * x_spacing)
        
        # Create colored background rectangle
        color = cmap(norm(attr))
        rect = patches.Rectangle(
            (x_pos - x_spacing/2 * 0.8, y_pos - 0.15), x_spacing * 0.8, 0.3,
            linewidth=0, facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add token text
        ax.text(x_pos, y_pos, token, ha='center', va='center',
                fontsize=9, weight='bold')
        
        # Add attribution value below
        ax.text(x_pos, y_pos - 0.3, f'{attr:.3f}', ha='center', va='center',
                fontsize=7, style='italic')
    
    # Add premise/hypothesis separator
    if premise_length is not None:
        sep_x = x_start + premise_length * x_spacing
        ax.axvline(x=sep_x, color='black', linestyle='--', alpha=0.5)
        ax.text(sep_x, y_pos + 0.35, 'P | H', ha='center', va='center',
                fontsize=10, weight='bold')
    
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    ax.axis('off')


def plot_attribution_statistics(
    attributions_dict: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot statistics comparing attributions across models.
    
    Args:
        attributions_dict: Dictionary mapping model names to attribution lists
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Compute statistics for each model
    model_stats = {}
    for model_name, attrs_list in attributions_dict.items():
        variances = [np.var(attrs) for attrs in attrs_list]
        means = [np.mean(np.abs(attrs)) for attrs in attrs_list]
        maxes = [np.max(np.abs(attrs)) for attrs in attrs_list]
        
        model_stats[model_name] = {
            'variances': variances,
            'mean_abs_attrs': means,
            'max_abs_attrs': maxes
        }
    
    # Plot variance distributions
    axes[0, 0].set_title('Attribution Variance Distribution')
    for model_name, stats in model_stats.items():
        axes[0, 0].hist(stats['variances'], alpha=0.7, label=model_name, bins=30)
    axes[0, 0].set_xlabel('Attribution Variance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Plot mean absolute attribution distributions
    axes[0, 1].set_title('Mean Absolute Attribution Distribution')
    for model_name, stats in model_stats.items():
        axes[0, 1].hist(stats['mean_abs_attrs'], alpha=0.7, label=model_name, bins=30)
    axes[0, 1].set_xlabel('Mean Absolute Attribution')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Box plot of variances
    axes[1, 0].set_title('Attribution Variance by Model')
    variance_data = [stats['variances'] for stats in model_stats.values()]
    axes[1, 0].boxplot(variance_data, labels=list(model_stats.keys()))
    axes[1, 0].set_ylabel('Attribution Variance')
    
    # Scatter plot: variance vs mean attribution
    axes[1, 1].set_title('Variance vs Mean Attribution')
    for model_name, stats in model_stats.items():
        axes[1, 1].scatter(stats['mean_abs_attrs'], stats['variances'], 
                          alpha=0.6, label=model_name)
    axes[1, 1].set_xlabel('Mean Absolute Attribution')
    axes[1, 1].set_ylabel('Attribution Variance')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_attribution_heatmap(
    attributions_matrix: np.ndarray,
    tokens_list: List[List[str]],
    title: str = "Attribution Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a heatmap showing attributions across multiple examples.
    
    Args:
        attributions_matrix: Matrix of attributions [n_examples, max_seq_len]
        tokens_list: List of token lists for each example
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attributions_matrix, cmap='RdBu_r', aspect='auto')
    
    # Set labels
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Example Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attribution Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig