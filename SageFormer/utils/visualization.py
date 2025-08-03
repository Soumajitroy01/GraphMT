import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_predictions(predictions, targets, save_path=None):
    """
    Plot predictions vs targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(targets.cpu().numpy(), predictions.cpu().numpy(), alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(targets.cpu().numpy()), min(predictions.cpu().numpy()))
    max_val = max(max(targets.cpu().numpy()), max(predictions.cpu().numpy()))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_circuit_performance(results_df, circuit_name, save_path=None):
    """
    Plot performance for a specific circuit.
    
    Args:
        results_df: DataFrame with results
        circuit_name: Name of the circuit
        save_path: Path to save the plot
    """
    # Filter results for the specified circuit
    circuit_df = results_df[results_df['circuit'] == circuit_name]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(circuit_df['true'], circuit_df['pred'], alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(circuit_df['true']), min(circuit_df['pred']))
    max_val = max(max(circuit_df['true']), max(circuit_df['pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values for {circuit_name}')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def create_comparison_table(results, save_path=None):
    """
    Create a comparison table of different models.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the table
    """
    # Create DataFrame
    df = pd.DataFrame(columns=['Model', 'Target', 'MAE', 'MAPE'])
    
    for model_name, metrics in results.items():
        df = df.append({
            'Model': model_name,
            'Target': metrics['target'],
            'MAE': f"{metrics['mae']:.3f}",
            'MAPE': f"{metrics['mape']:.2f}%"
        }, ignore_index=True)
    
    # Sort by MAE
    df = df.sort_values('MAE')
    
    # Print table
    print(df.to_string(index=False))
    
    # Save to file if specified
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
