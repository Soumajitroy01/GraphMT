# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mape_loss(output, target, epsilon=1e-8):
    """
    Mean Absolute Percentage Error loss.
    
    Args:
        output: Predictions
        target: Ground truth values
        epsilon: Small constant for numerical stability
        
    Returns:
        MAPE loss value
    """
    return torch.mean(torch.abs((target - output) / (target + epsilon)))

def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Compute MAPE
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
