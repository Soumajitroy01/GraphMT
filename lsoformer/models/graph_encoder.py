# models/graph_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Graph encoder for AIG graphs
        
        Args:
            input_dim: Input dimension (node features)
            hidden_dim: Hidden dimension
        """
        super(GraphEncoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

class LevelWisePooling(nn.Module):
    def __init__(self, hidden_dim):
        """
        Level-wise pooling for DAG structure
        
        Args:
            hidden_dim: Hidden dimension of node embeddings
        """
        super(LevelWisePooling, self).__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, node_embeddings, node_depths):
        """
        Forward pass
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            node_depths: Depth of each node [num_nodes] or [batch, num_nodes]
        
        Returns:
            Level embeddings [max_depth, 2*hidden_dim]
        """
        # Ensure node_depths has the right shape
        if node_depths.dim() > 1:
            node_depths = node_depths.squeeze(0)
        
        # Get maximum depth
        max_depth = int(node_depths.max().item()) + 1
        level_embeddings = []
        
        # Pool nodes at each level
        for level in range(max_depth):
            # Get nodes at this level
            level_mask = (node_depths == level)
            if level_mask.sum() > 0:
                level_nodes = node_embeddings[level_mask]
                
                # Apply mean and max pooling
                mean_pool = torch.mean(level_nodes, dim=0)
                max_pool = torch.max(level_nodes, dim=0)[0]
                
                # Concatenate pooling results
                level_emb = torch.cat([mean_pool, max_pool], dim=0)
                level_embeddings.append(level_emb)
            else:
                # If no nodes at this level, add zero embedding
                level_embeddings.append(torch.zeros(2 * self.hidden_dim, 
                                                  device=node_embeddings.device))
        
        # Stack to create sequence of level embeddings
        return torch.stack(level_embeddings)

