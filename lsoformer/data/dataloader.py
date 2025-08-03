# data/dataloader.py
import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from config import Config

def create_dataloaders(dataset, batch_size, train_ratio=0.7, val_ratio=0.15):
    """
    Create train, validation, and test dataloaders
    
    Args:
        dataset: LSODataset instance
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Calculate sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # CUDA-specific dataloader kwargs
    kwargs = {
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY,
        'persistent_workers': True if Config.NUM_WORKERS > 0 else False
    }
    
    # Create dataloaders
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader

def create_inductive_dataloaders(dataset, batch_size, test_designs):
    """
    Create dataloaders for inductive setting (circuit-wise)
    
    Args:
        dataset: LSODataset instance
        batch_size: Batch size for dataloaders
        test_designs: List of design names to use for testing
    
    Returns:
        train_loader, test_loader
    """
    # Split based on design names
    train_indices = [i for i, item in enumerate(dataset) 
                    if item['design_name'] not in test_designs]
    test_indices = [i for i, item in enumerate(dataset) 
                   if item['design_name'] in test_designs]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # CUDA-specific dataloader kwargs
    kwargs = {
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY
    }
    
    # Create dataloaders
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader
