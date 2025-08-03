# utils/config.py
import os
import torch

class Config:
    """Configuration for QoR prediction model."""
    
    # Data paths
    csv_file = "data/results.csv"
    graph_dir = "data/graphs"
    output_dir = "output"
    
    # Model parameters
    gnn_type = "graphsage"  # 'gcn', 'gat', or 'graphsage'
    recipe_type = "transformer"  # 'transformer', 'lstm', or 'cnn'
    target_metric = "levels"  # 'nodes', 'levels', or 'iterations'
    
    # GNN hyperparameters (Table 1)
    gnn_input_dim = 4
    gnn_hidden_dim = 64
    
    # Recipe encoder hyperparameters (Table 2)
    recipe_seq_len = 20
    transformer_embedding_dim = 4
    lstm_embedding_dim = 3
    cnn_embedding_dim = 3
    
    # FC hyperparameters (Table 3)
    fc_dropout = 0.2
    
    # Training parameters
    batch_size = 1
    num_epochs = 1000
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # Loss function
    loss_type = "mse"  # 'mse' or 'mape'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    model_save_path = os.path.join(output_dir, "model.pt")
    results_save_path = os.path.join(output_dir, "results.csv")
    
    def __init__(self, **kwargs):
        # Update config with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
