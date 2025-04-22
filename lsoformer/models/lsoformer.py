# models/lsoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_encoder import GraphEncoder, LevelWisePooling
from models.recipe_encoder import RecipeEncoder, PositionalEncoding
from models.transformer_decoder import TransformerDecoder, generate_causal_mask

class LSOformer(nn.Module):
    def __init__(self, input_node_dim, hidden_dim, num_heuristics, 
                 nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=1):
        """
        LSOformer model for Logic Synthesis Optimization
        
        Args:
            input_node_dim: Input dimension of node features
            hidden_dim: Hidden dimension for graph encoder
            num_heuristics: Number of unique heuristics
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_layers: Number of transformer decoder layers
        """
        super(LSOformer, self).__init__()
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(input_node_dim, hidden_dim)
        self.level_pooling = LevelWisePooling(hidden_dim)
        
        # Recipe encoder
        self.recipe_encoder = RecipeEncoder(num_heuristics, 2*hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(2*hidden_dim, dropout)
        
        # Transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoder(2*hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # QoR prediction head
        self.qor_head = nn.Linear(2*hidden_dim, 1)
    
    def forward(self, batch):
        """
        Forward pass
        
        Args:
            batch: Dictionary containing:
                - graph: PyG graph batch
                - recipe_indices: Recipe indices [batch_size, seq_len]
                - node_depths: Node depths [num_nodes]
        
        Returns:
            QoR trajectory predictions [batch_size, seq_len]
        """
        # Extract batch components
        graph_batch = batch['graph']
        recipe_indices = batch['recipe_indices']
        node_depths = batch['node_depths']
        
        # Get device
        device = recipe_indices.device
        
        # Encode graphs
        node_features, edge_index = graph_batch.x, graph_batch.edge_index
        node_embeddings = self.graph_encoder(node_features, edge_index)
        
        # Level-wise pooling
        graph_seq = self.level_pooling(node_embeddings, node_depths)  # [max_depth, 2*hidden_dim]
        
        # Encode recipes
        recipe_embeddings = self.recipe_encoder(recipe_indices)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Add positional encoding
        recipe_embeddings = self.pos_encoder(recipe_embeddings)
        
        # Create causal mask
        seq_len = recipe_indices.size(1)
        causal_mask = generate_causal_mask(seq_len, device)
        
        # Transpose for transformer input: [seq_len, batch_size, dim]
        recipe_embeddings = recipe_embeddings.transpose(0, 1)
        
        # Expand graph sequence for batch
        batch_size = recipe_indices.size(0)
        graph_seq = graph_seq.unsqueeze(1).expand(-1, batch_size, -1)  # [max_depth, batch_size, 2*hidden_dim]
        
        # Apply transformer decoder layers
        output = recipe_embeddings
        for layer in self.transformer_layers:
            output = layer(output, graph_seq, causal_mask)
        
        # Predict QoR trajectory
        qor_trajectory = self.qor_head(output).squeeze(-1).transpose(0, 1)  # [batch_size, seq_len]
        
        return qor_trajectory
        
    def to(self, device):
        """
        Move model to device (overrides nn.Module.to)
        
        Args:
            device: Device to move model to
            
        Returns:
            Model on specified device
        """
        model = super().to(device)
        return model
