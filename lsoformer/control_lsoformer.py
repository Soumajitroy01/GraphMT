# control_lsoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.lsoformer import LSOformer

class ZeroLinear(nn.Module):
    """Linear layer with weights and bias initialized to zero"""
    def __init__(self, in_features, out_features):
        super(ZeroLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights and bias to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

class ControlLSOformer(nn.Module):
    def __init__(self, pretrained_model, alpha=0.5):
        """
        ControlNet-inspired fine-tuning for LSOformer
        
        Args:
            pretrained_model: Pre-trained LSOformer model
            alpha: Weight for combining outputs (0-1), higher means more influence from trainable copy
        """
        super(ControlLSOformer, self).__init__()
        
        # Create locked copy (frozen)
        self.locked_model = copy.deepcopy(pretrained_model)
        for param in self.locked_model.parameters():
            param.requires_grad = False
        self.locked_model.eval()
        
        # Create trainable copy
        self.trainable_model = copy.deepcopy(pretrained_model)
        
        # Get embedding dimension
        self.hidden_dim = pretrained_model.transformer_layers[0].self_attn.in_proj_weight.size(1)
        
        # Zero linear layers for input and output
        self.input_zero_linear = ZeroLinear(self.hidden_dim, self.hidden_dim)
        self.output_zero_linear = ZeroLinear(self.hidden_dim, self.hidden_dim)
        
        # Weight for combining outputs
        self.alpha = alpha
    
    def forward(self, batch):
        """
        Forward pass through both models and combine results
        
        Args:
            batch: Input batch containing graph, recipe_indices, and node_depths
            
        Returns:
            Combined QoR trajectory predictions
        """
        # Forward pass through locked model
        with torch.no_grad():
            locked_output = self.locked_model(batch)
        
        # Forward pass through trainable model with zero linear layers
        trainable_output = self.trainable_model(batch)
        trainable_output = self.output_zero_linear(trainable_output)
        
        # Combine outputs
        combined_output = (1 - self.alpha) * locked_output + self.alpha * trainable_output
        
        return combined_output