# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset import CircuitGraphDataset, collate_fn
from models.qor_model import QoRPredictionModel
from utils.config import Config
from utils.metrics import mape_loss, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train QoR prediction model')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with results')
    parser.add_argument('--graph_dir', type=str, help='Directory with graph files')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--gnn_type', type=str, choices=['gcn', 'gat', 'graphsage'], help='GNN type')
    parser.add_argument('--recipe_type', type=str, choices=['transformer', 'lstm', 'cnn'], help='Recipe encoder type')
    parser.add_argument('--target', type=str, choices=['nodes', 'levels', 'iterations'], help='Target metric')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mape'], help='Loss function type')
    return parser.parse_args()

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in tqdm(loader, desc="Training"):
        # Move data to device
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Get target based on model's target metric
        if model.target == 'nodes':
            target = data.nodes
        elif model.target == 'levels':
            target = data.levels
        elif model.target == 'iterations':
            target = data.iterations
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get target based on model's target metric
            if model.target == 'nodes':
                target = data.nodes
            elif model.target == 'levels':
                target = data.levels
            elif model.target == 'iterations':
                target = data.iterations
            
            # Compute loss
            loss = criterion(output, target)
            
            # Track loss and predictions
            total_loss += loss.item() * data.num_graphs
            all_preds.append(output)
            all_targets.append(target)
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets)
    
    return total_loss / len(loader.dataset), metrics, all_preds, all_targets

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration
    config = Config()
    if args.csv_file:
        config.csv_file = args.csv_file
    if args.graph_dir:
        config.graph_dir = args.graph_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.gnn_type:
        config.gnn_type = args.gnn_type
    if args.recipe_type:
        config.recipe_type = args.recipe_type
    if args.target:
        config.target_metric = args.target
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.loss_type:
        config.loss_type = args.loss_type
    
    # Create model-specific output directory
    model_dir = f"{config.gnn_type}_{config.recipe_type}_{config.target_metric}"
    config.output_dir = os.path.join(config.output_dir, model_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set device
    device = config.device
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = CircuitGraphDataset(csv_file=config.csv_file, graph_dir=config.graph_dir)
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = QoRPredictionModel(
        node_feature_dim=dataset[0].x.size(1),
        vocab_size=len(dataset.vocab),
        gnn_type=config.gnn_type,
        recipe_type=config.recipe_type,
        target=config.target_metric
    ).to(device)
    
    # Print model architecture
    print(f"Model: {config.gnn_type} + {config.recipe_type} for {config.target_metric} prediction")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Define loss function
    if config.loss_type == 'mse':
        criterion = nn.MSELoss()
    else:  # mape
        criterion = mape_loss
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Train model
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_metrics['mae']:.6f}")
        print(f"  Val MAPE: {val_metrics['mape']:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': {
                    'gnn_type': config.gnn_type,
                    'recipe_type': config.recipe_type,
                    'target_metric': config.target_metric
                }
            }, os.path.join(config.output_dir, "best_model.pt"))
            print(f"  New best model saved with val_loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()
