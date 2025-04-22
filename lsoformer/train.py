# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
# With this
from tqdm.auto import tqdm
import pickle

from config import Config
from data.dataset import LSODataset
from data.dataloader import create_dataloaders, create_inductive_dataloaders
from models.lsoformer import LSOformer
from utils.metrics import mse_loss, mean_absolute_percentage_error
from utils.visualization import plot_training_curves, save_visualization_metadata
from utils.helpers import set_seed, get_timestamp, save_model, save_losses, move_batch_to_device, empty_cuda_cache, reset_tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    
    with tqdm(dataloader, desc="Training", position=0, leave=True) as pbar:
        for batch in pbar:
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_trajectory = model(batch)
            
            # Calculate loss
            loss = criterion(pred_trajectory, batch['qor_trajectory'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    
    with tqdm(dataloader, desc="Validation", position=0, leave=True) as pbar:
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                batch = move_batch_to_device(batch, device)
                
                # Forward pass
                pred_trajectory = model(batch)
                
                # Calculate loss
                loss = criterion(pred_trajectory, batch['qor_trajectory'])
                val_loss += loss.item()
                
                # Update progress bar with current loss
                pbar.set_postfix(loss=f"{loss.item():.6f}")
    
    return val_loss / len(dataloader)

def train(args):
    """Main training function"""
    # Create directories
    Config.create_dirs()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Create timestamp for this run
    timestamp = get_timestamp()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LSODataset(Config.CSV_PATH, Config.GRAPH_DIR)
    
    # Create dataloaders
    if args.inductive:
        # For inductive setup, we need to specify test designs
        with open(args.test_designs, 'r') as f:
            test_designs = [line.strip() for line in f.readlines()]
        
        train_loader, val_loader = create_inductive_dataloaders(
            dataset, Config.BATCH_SIZE, test_designs)
    else:
        train_loader, val_loader, _ = create_dataloaders(
            dataset, Config.BATCH_SIZE)
    
    # Get input dimension from first graph
    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch['graph'].x.size(1)
    
    # Create model
    print("Creating model...")
    model = LSOformer(
        input_node_dim=input_node_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_heuristics=len(dataset.heuristic_to_idx),
        nhead=Config.NUM_HEADS,
        dim_feedforward=Config.TRANSFORMER_DIM * 4,
        dropout=Config.DROPOUT,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Create criterion
    criterion = mse_loss
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        reset_tqdm()  # Reset tqdm instances
        # Train
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Empty CUDA cache if needed
        empty_cuda_cache()
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Empty CUDA cache if needed
        empty_cuda_cache()
        
        # Use tqdm.write instead of print to avoid disrupting progress bars
        tqdm.write(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(Config.SAVE_DIR, f"best_model_{epoch+1}.pt")
            save_model(model, optimizer, epoch, val_loss, save_path)
            tqdm.write(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
        
        save_path = os.path.join(Config.SAVE_DIR, f"best_model_{epoch+1}.pt")
        save_model(model, optimizer, epoch, val_loss, save_path)
            
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            tqdm.write(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save losses
    losses_path = os.path.join(Config.LOGS_DIR, f"losses_{timestamp}.pkl")
    save_losses(train_losses, val_losses, losses_path)
    
    # Plot training curves
    plot_path = os.path.join(Config.VISUALIZATION_DIR, f"training_curves_{timestamp}.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # Save visualization metadata
    save_visualization_metadata(Config, timestamp, Config.VISUALIZATION_DIR)
    
    # Save heuristic vocabulary
    vocab_path = os.path.join(Config.SAVE_DIR, f"heuristic_vocab_{timestamp}.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(dataset.heuristic_to_idx, f)
    
    print(f"Training completed. Results saved with timestamp: {timestamp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSOformer model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inductive", action="store_true", help="Use inductive setup")
    parser.add_argument("--test_designs", type=str, default="test_designs.txt",
                        help="File with test design names for inductive setup")
    parser.add_argument("--cuda_device", type=int, default=0, 
                        help="CUDA device index to use")
    
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        Config.CUDA_DEVICE = args.cuda_device
        
    train(args)
