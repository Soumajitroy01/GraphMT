# models/recipe/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    """LSTM encoder for optimization sequences with hyperparameters from Table 2."""
    
    def __init__(self, vocab_size, seq_len=20, embedding_dim=3):
        super(LSTMEncoder, self).__init__()
        
        # Parameters from Table 2
        # Input = (20, 3)
        # Hidden_Size = 64
        # Num_layers = 2
        # Output = 64
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=64,  # Table 2: Hidden_Size = 64
            num_layers=2,    # Table 2: Num_layers = 2
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, src):
        # src shape: [batch_size, seq_len] or [seq_len]
        # print(f"Input shape to LSTM: {src.shape}")
        
        # Add batch dimension if it's missing
        if src.dim() == 1:
            src = src.unsqueeze(0)  # Add batch dimension [1, seq_len]
            # print(f"Added batch dimension to input, new shape: {src.shape}")
            was_batched = False
        else:
            was_batched = True
            
        # Embed tokens
        src = self.embedding(src)  # [batch_size, seq_len, embedding_dim]
        # print(f"After embedding shape: {src.shape}")
        
        # Apply LSTM
        output, (hidden, cell) = self.lstm(src)
        # print(f"LSTM output shape: {output.shape}, hidden shape: {hidden.shape}")
        
        # Use the last hidden state as the sequence representation
        # For multi-layer LSTM, hidden has shape [num_layers, batch_size, hidden_size]
        # We take the last layer's hidden state
        sequence_embedding = hidden[-1]  # [batch_size, hidden_size]
        # print(f"Final sequence embedding shape: {sequence_embedding.shape}")
        
        # Remove batch dimension if it was added
        # if not was_batched:
        #     sequence_embedding = sequence_embedding.squeeze(0)
        #     print(f"Removed batch dimension, final shape: {sequence_embedding.shape}")
        
        return sequence_embedding
