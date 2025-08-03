# utils/helpers.py
import os
import torch
import numpy as np
import pickle
import random
from datetime import datetime
from config import Config
from tqdm import tqdm

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_model(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path, device=None):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to checkpoint file
        device: Device to load model onto (default: None, uses current device)
    
    Returns:
        model, optimizer, epoch, loss
    """
    if device is None:
        device = Config.get_device()
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer state to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_losses(train_losses, val_losses, path):
    """Save training and validation losses"""
    losses = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    with open(path, 'wb') as f:
        pickle.dump(losses, f)

def load_losses(path):
    """Load training and validation losses"""
    with open(path, 'rb') as f:
        losses = pickle.load(f)
    return losses['train_loss'], losses['val_loss']

def move_batch_to_device(batch, device):
    """
    Move batch to device
    
    Args:
        batch: Dictionary of tensors or PyG data objects
        device: Device to move tensors to
    
    Returns:
        Batch with all tensors on specified device
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif hasattr(v, 'to'):
            # PyG data objects have a to() method
            result[k] = v.to(device)
        else:
            result[k] = v
    return result

def empty_cuda_cache():
    """Empty CUDA cache to free up memory"""
    if torch.cuda.is_available() and Config.CUDA_EMPTY_CACHE:
        torch.cuda.empty_cache()

# In utils/helpers.py
def reset_tqdm():
    """Reset all tqdm instances"""
    if hasattr(tqdm, '_instances'):
        for instance in list(tqdm._instances):
            instance.close()
