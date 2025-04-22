# models/recipe/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    
    def __init__(self, d_model, max_seq_len=20):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle the case where d_model is odd
        if d_model % 2 != 0:
            div_term = div_term[:d_model//2 + 1]
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] or [seq_len, d_model]
        # print(f"In positional encoding - x shape: {x.shape}, pe shape: {self.pe.shape}")
        
        # Add batch dimension if it's missing
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension [1, seq_len, d_model]
            # print(f"Added batch dimension, new shape: {x.shape}")
            
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder for optimization sequences with hyperparameters from Table 2."""
    
    def __init__(self, vocab_size, seq_len=20, embedding_dim=4):
        super(TransformerEncoder, self).__init__()
        
        # Parameters from Table 2
        # Input = (20, 4)
        # Num_Head = 2
        # Dim_feedforward = 32
        # Num_layers = 3
        # Linear = 50
        # Output = 50
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len=seq_len)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=2,  # Table 2: Num_Head = 2
            dim_feedforward=32,  # Table 2: Dim_feedforward = 32
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=3  # Table 2: Num_layers = 3
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, 50)  # Table 2: Linear = 50, Output = 50
        
    def forward(self, src):
        # src shape: [batch_size, seq_len] or [seq_len]
        # print(f"Input shape to transformer: {src.shape}")
        
        # Add batch dimension if it's missing
        if src.dim() == 1:
            src = src.unsqueeze(0)  # Add batch dimension [1, seq_len]
            # print(f"Added batch dimension to input, new shape: {src.shape}")
        
        # Embed tokens
        src = self.embedding(src)  # [batch_size, seq_len, embedding_dim]
        # print(f"After embedding shape: {src.shape}")
        
        # Add positional encoding
        src = self.pos_encoder(src)
        # print(f"After positional encoding shape: {src.shape}")
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)  # [batch_size, seq_len, embedding_dim]
        # print(f"After transformer encoder shape: {output.shape}")
        
        # Global average pooling
        output = torch.mean(output, dim=1)  # [batch_size, embedding_dim]
        # print(f"After global pooling shape: {output.shape}")
        
        # Apply output projection
        output = self.output_projection(output)  # [batch_size, 50]
        # print(f"Final output shape: {output.shape}")
        
        # Remove batch dimension if it was added
        # if output.size(0) == 1:
        #     output = output.squeeze(0)
        #     print(f"Removed batch dimension, final shape: {output.shape}")
        
        return output
