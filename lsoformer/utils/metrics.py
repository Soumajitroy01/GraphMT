# utils/metrics.py
import numpy as np
import torch

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse_loss(y_pred, y_true):
    """
    Calculate Mean Squared Error loss
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        MSE loss
    """
    return torch.mean((y_pred - y_true) ** 2)
