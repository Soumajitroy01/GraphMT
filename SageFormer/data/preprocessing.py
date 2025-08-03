import torch
import numpy as np
from torch_geometric.data import Data

def normalize_node_features(x, method='standard'):
    """
    Normalize node features.
    
    Args:
        x: Node feature tensor
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Normalized feature tensor
    """
    if method == 'standard':
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std[std == 0] = 1.0  # Prevent division by zero
        return (x - mean) / std
    
    elif method == 'minmax':
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Prevent division by zero
        return (x - min_val) / range_val
    
    return x

def augment_graph(data, edge_dropout_prob=0.1, feature_noise_std=0.01):
    """
    Augment graph data with edge dropout and feature noise.
    
    Args:
        data: PyG Data object
        edge_dropout_prob: Probability of dropping an edge
        feature_noise_std: Standard deviation of feature noise
        
    Returns:
        Augmented Data object
    """
    # Create a copy of the data
    augmented_data = Data()
    
    # Copy all attributes
    for key in data.keys:
        augmented_data[key] = data[key]
    
    # Apply edge dropout
    if edge_dropout_prob > 0:
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges) > edge_dropout_prob
        augmented_data.edge_index = data.edge_index[:, mask]
    
    # Apply feature noise
    if feature_noise_std > 0:
        noise = torch.randn_like(data.x) * feature_noise_std
        augmented_data.x = data.x + noise
    
    return augmented_data
