# models/gnn/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class GCNEncoder(nn.Module):
    """GCN-based circuit encoder with hyperparameters from Table 1."""
    
    def __init__(self, input_dim=4, hidden_dim=64):
        super(GCNEncoder, self).__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers (Table 1: Layer1=64, Layer2=64)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Output projection (Table 1: Output=128 after Max+Mean pooling)
        self.output_projection = nn.Linear(hidden_dim * 2, 128)
        
    def forward(self, x, edge_index, batch):
        # Initial node embedding
        x = self.node_embedding(x)
        # print(f"After node embedding: {x.shape}")
        x = F.relu(x)
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        # print(f"After GCN layer 1: {x.shape}")
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        # print(f"After GCN layer 2: {x.shape}")
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global pooling (Table 1: Max+Mean)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        # print(f"After max pooling: {x_max.shape}, After mean pooling: {x_mean.shape}")
        x_combined = torch.cat([x_max, x_mean], dim=1)
        # print(f"After concatenating pooled features: {x_combined.shape}")
        
        # Output projection
        x_out = self.output_projection(x_combined)
        # print(f"After output projection: {x_out.shape}")
        
        return x_out
