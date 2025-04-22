# models/recipe/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """CNN encoder for optimization sequences with hyperparameters from Table 2."""
    
    def __init__(self, vocab_size, seq_len=20, embedding_dim=3):
        super(CNNEncoder, self).__init__()
        
        # Parameters from Table 2
        # Input = 60 (this would be seq_len * embedding_dim = 20 * 3 = 60)
        # Filters = 4
        # Kernels = 21, 24, 27, 30
        # Stride = 3
        # Output = 50
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN layers with different kernel sizes
        self.conv1 = nn.Conv1d(1, 1, kernel_size=21, stride=3, padding=21//2)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=24, stride=3, padding=24//2)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=27, stride=3, padding=27//2)
        self.conv4 = nn.Conv1d(1, 1, kernel_size=30, stride=3, padding=30//2)
        
        # Output projection
        self.output_projection = nn.Linear(50, 50)  # Table 2: Output = 50
        
    def forward(self, src):
        # src shape: [batch_size, seq_len] or [seq_len]
        # print(f"Input shape to CNN: {src.shape}")
        
        # Add batch dimension if it's missing
        if src.dim() == 1:
            src = src.unsqueeze(0)  # Add batch dimension [1, seq_len]
            # print(f"Added batch dimension to input, new shape: {src.shape}")
            
        batch_size = src.size(0)
        
        # Embed tokens
        embedded = self.embedding(src)  # [batch_size, seq_len, embedding_dim]
        # print(f"After embedding shape: {embedded.shape}")
        
        # Reshape to [batch_size, 1, seq_len * embedding_dim]
        # This creates a 1D sequence of length 60 (20 * 3) for each batch item
        embedded = embedded.view(batch_size, 1, -1)
        # print(f"After reshaping: {embedded.shape}")
        
        # Apply convolutions
        x1 = F.relu(self.conv1(embedded))
        x2 = F.relu(self.conv2(embedded))
        x3 = F.relu(self.conv3(embedded))
        x4 = F.relu(self.conv4(embedded))
        
        # print(f"After conv1: {x1.shape}")
        # print(f"After conv2: {x2.shape}")
        # print(f"After conv3: {x3.shape}")
        # print(f"After conv4: {x4.shape}")
        
        # Flatten and concatenate
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x3 = x3.view(batch_size, -1)
        x4 = x4.view(batch_size, -1)
        
        # print(f"After flattening - x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
        
        # Ensure each tensor has the right size for concatenation to get total 50 features
        # We'll resize them to have sizes that sum to 50 (e.g., 14, 13, 12, 11)
        x1 = x1[:, :14] if x1.size(1) >= 14 else F.pad(x1, (0, 14 - x1.size(1)))
        x2 = x2[:, :13] if x2.size(1) >= 13 else F.pad(x2, (0, 13 - x2.size(1)))
        x3 = x3[:, :12] if x3.size(1) >= 12 else F.pad(x3, (0, 12 - x3.size(1)))
        x4 = x4[:, :11] if x4.size(1) >= 11 else F.pad(x4, (0, 11 - x4.size(1)))
        
        # Concatenate to get [batch_size, 50]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # print(f"After concatenation: {x.shape}")
        
        # Apply output projection
        x = self.output_projection(x)  # [batch_size, 50]
        # print(f"Final output shape: {x.shape}")
        
        # Remove batch dimension if it was added
        if batch_size == 1 and src.dim() == 1:
            x = x.squeeze(0)
            # print(f"Removed batch dimension, final shape: {x.shape}")
        
        return x
