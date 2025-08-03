# models/recipe_encoder.py
import torch
import torch.nn as nn
import math

class RecipeEncoder(nn.Module):
    def __init__(self, num_heuristics, embedding_dim):
        """
        Recipe encoder
        
        Args:
            num_heuristics: Number of unique heuristics
            embedding_dim: Embedding dimension
        """
        super(RecipeEncoder, self).__init__()
        self.embedding = nn.Embedding(num_heuristics, embedding_dim)
    
    def forward(self, recipe_indices):
        """
        Forward pass
        
        Args:
            recipe_indices: Indices of heuristics [batch_size, seq_len]
        
        Returns:
            Recipe embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(recipe_indices)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding for transformer
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output with positional encoding [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
