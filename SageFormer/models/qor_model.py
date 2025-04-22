# models/qor_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gnn.gcn import GCNEncoder
from models.gnn.gat import GATEncoder
from models.gnn.graphsage import GraphSAGEEncoder
from models.recipe.transformer import TransformerEncoder
from models.recipe.lstm import LSTMEncoder
from models.recipe.cnn import CNNEncoder

class QoRPredictionModel(nn.Module):
    """Combined model for QoR prediction with hyperparameters from Table 3."""
    
    def __init__(self, node_feature_dim, vocab_size, gnn_type='graphsage', recipe_type='transformer', target='nodes'):
        super(QoRPredictionModel, self).__init__()
        
        # Circuit feature extractor (GNN)
        if gnn_type == 'gcn':
            self.circuit_encoder = GCNEncoder(input_dim=node_feature_dim)
        elif gnn_type == 'gat':
            self.circuit_encoder = GATEncoder(input_dim=node_feature_dim)
        elif gnn_type == 'graphsage':
            self.circuit_encoder = GraphSAGEEncoder(input_dim=node_feature_dim)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Recipe feature extractor
        if recipe_type == 'transformer':
            self.recipe_encoder = TransformerEncoder(vocab_size=vocab_size)
            # Input to FC will be 178 (128 from GNN + 50 from Transformer)
            self.fc_input_dim = 178
        elif recipe_type == 'lstm':
            self.recipe_encoder = LSTMEncoder(vocab_size=vocab_size)
            # Input to FC will be 192 (128 from GNN + 64 from LSTM)
            self.fc_input_dim = 192
        elif recipe_type == 'cnn':
            self.recipe_encoder = CNNEncoder(vocab_size=vocab_size)
            # Input to FC will be 178 (128 from GNN + 50 from CNN)
            self.fc_input_dim = 178
        else:
            raise ValueError(f"Unknown recipe encoder type: {recipe_type}")
        
        # Fully connected layers for regression (Table 3)
        # FC Stack: Input=178/192, Linear1=512, Linear2=256, Linear3=256, Linear4=1, Dropout=0.2
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Target metric
        self.target = target
        
        # Store model types for reference
        self.gnn_type = gnn_type
        self.recipe_type = recipe_type
        
    def forward(self, data):
        # Extract features from the circuit graph
        # print("Processing circuit graph...")
        circuit_features = self.circuit_encoder(data.x, data.edge_index, data.batch)
        # print(f"Circuit features shape: {circuit_features.shape}")
        
        # Extract features from the optimization sequence
        # print("Processing optimization sequence...")
        sequence_features = self.recipe_encoder(data.recipe)
        # print(f"Sequence features shape: {sequence_features.shape}")
        
        # Concatenate features
        combined_features = torch.cat([sequence_features, circuit_features], dim=1)
        # print(f"Combined features shape: {combined_features.shape}")
        
        # Apply fully connected layers
        x = F.relu(self.fc1(combined_features))
        # print(f"After FC1: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        # print(f"After FC2: {x.shape}")
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        # print(f"After FC3: {x.shape}")
        x = self.dropout(x)
        
        x = self.fc4(x)
        # print(f"Final output shape: {x.shape}")
        
        return x
    
    def predict(self, data):
        """Make a prediction and return the target value."""
        self.eval()
        with torch.no_grad():
            prediction = self.forward(data)
            
            # Return the appropriate target
            if self.target == 'nodes':
                return prediction, data.nodes
            elif self.target == 'levels':
                return prediction, data.levels
            elif self.target == 'iterations':
                return prediction, data.iterations
            else:
                return prediction, None
