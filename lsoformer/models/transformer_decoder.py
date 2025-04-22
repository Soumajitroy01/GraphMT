# models/transformer_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        Transformer decoder with cross-attention
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None):
        """
        Forward pass
        
        Args:
            tgt: Target sequence [seq_len, batch_size, d_model]
            memory: Memory from encoder [seq_len, batch_size, d_model]
            tgt_mask: Target mask for causal attention
        
        Returns:
            Output tensor [seq_len, batch_size, d_model]
        """
        # Self-attention with causal mask
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention with graph embeddings
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

def generate_causal_mask(seq_len, device=None):
    """
    Generate a causal mask for the transformer decoder
    
    Args:
        seq_len: Sequence length
        device: Device to put the mask on
    
    Returns:
        Causal mask [seq_len, seq_len]
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    if device is not None:
        mask = mask.to(device)
        
    return mask
