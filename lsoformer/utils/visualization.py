# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Set the tick locations
    plt.xticks(np.arange(0, len(train_losses) + 1, 2))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_qor_trajectory(true_trajectory, pred_trajectory, design_name, save_path=None):
    """
    Plot QoR trajectory for a single design
    
    Args:
        true_trajectory: True QoR trajectory
        pred_trajectory: Predicted QoR trajectory
        design_name: Name of the design
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    steps = range(1, len(true_trajectory) + 1)
    
    plt.plot(steps, true_trajectory, 'b-o', label='True QoR')
    plt.plot(steps, pred_trajectory, 'r-o', label='Predicted QoR')
    
    plt.title(f'QoR Trajectory for {design_name}')
    plt.xlabel('Optimization Step')
    plt.ylabel('QoR (Levels)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_mape_by_circuit(mape_values, circuit_names, metric_name='Delay', save_path=None):
    """
    Plot MAPE values for each circuit
    
    Args:
        mape_values: MAPE values for each circuit
        circuit_names: Names of circuits
        metric_name: Name of the metric (Delay or Area)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Sort by MAPE values
    sorted_indices = np.argsort(mape_values)
    sorted_mape = [mape_values[i] for i in sorted_indices]
    sorted_names = [circuit_names[i] for i in sorted_indices]
    
    # Create bar plot
    bars = plt.bar(range(len(sorted_mape)), sorted_mape, color='skyblue')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{sorted_mape[i]:.2f}%', ha='center', va='bottom', rotation=0)
    
    plt.title(f'MAPE for {metric_name} by Circuit')
    plt.xlabel('Circuit')
    plt.ylabel('MAPE (%)')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_comparison_heatmap(model_results, metric='delay', save_path=None):
    """
    Plot heatmap comparing different models
    
    Args:
        model_results: Dictionary with model names as keys and lists of MAPE values as values
        metric: Metric name (delay or area)
        save_path: Path to save the plot
    """
    # Create DataFrame from results
    df = pd.DataFrame(model_results)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
    
    plt.title(f'Model Comparison - {metric.capitalize()} MAPE (%)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_visualization_metadata(config, timestamp, save_dir):
    """
    Save metadata about the visualization
    
    Args:
        config: Configuration object
        timestamp: Timestamp string
        save_dir: Directory to save metadata
    """
    metadata = {
        'timestamp': timestamp,
        'hidden_dim': config.HIDDEN_DIM,
        'transformer_dim': config.TRANSFORMER_DIM,
        'num_heads': config.NUM_HEADS,
        'num_layers': config.NUM_LAYERS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE
    }
    
    metadata_path = os.path.join(save_dir, f'metadata_{timestamp}.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')
