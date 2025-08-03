# finetune.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm.auto import tqdm
import pickle

from config import Config
from data.dataset import LSODataset
from data.dataloader import create_dataloaders
from models.lsoformer import LSOformer
from control_lsoformer import ControlLSOformer
from utils.metrics import mse_loss
from utils.visualization import plot_training_curves, save_visualization_metadata
from utils.helpers import set_seed, get_timestamp, save_model, save_losses, move_batch_to_device, empty_cuda_cache

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

def finetune(args):
    """Main fine-tuning function"""
    # Create directories
    Config.create_dirs()
    finetune_dir = os.path.join(Config.SAVE_DIR, "finetuned_models")
    os.makedirs(finetune_dir, exist_ok=True)
    
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
    train_loader, val_loader, _ = create_dataloaders(dataset, Config.BATCH_SIZE)
    
    # Get input dimension from first graph
    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch['graph'].x.size(1)
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    pretrained_model = LSOformer(
        input_node_dim=input_node_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_heuristics=len(dataset.heuristic_to_idx),
        nhead=Config.NUM_HEADS,
        dim_feedforward=Config.TRANSFORMER_DIM * 4,
        dropout=Config.DROPOUT,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # Load pre-trained weights
    checkpoint = torch.load(args.pretrained_model_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create ControlLSOformer model
    print("Creating ControlLSOformer model...")
    model = ControlLSOformer(pretrained_model, alpha=args.alpha).to(device)
    
    # Create optimizer - only optimize trainable model parameters
    trainable_params = list(model.trainable_model.parameters()) + \
                      list(model.input_zero_linear.parameters()) + \
                      list(model.output_zero_linear.parameters())
    
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    
    # Create criterion
    criterion = mse_loss
    
    # Training loop
    print("Starting fine-tuning...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Empty CUDA cache if needed
        empty_cuda_cache()
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Empty CUDA cache if needed
        empty_cuda_cache()
        
        # Use tqdm.write to avoid disrupting progress bars
        tqdm.write(f"Epoch {epoch+1}/{args.num_epochs}, "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(finetune_dir, f"best_model_{timestamp}.pt")
            save_model(model, optimizer, epoch, val_loss, save_path)
            tqdm.write(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            tqdm.write(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save losses
    losses_path = os.path.join(Config.LOGS_DIR, f"finetune_losses_{timestamp}.pkl")
    save_losses(train_losses, val_losses, losses_path)
    
    # Plot training curves
    plot_path = os.path.join(Config.VISUALIZATION_DIR, f"finetune_curves_{timestamp}.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # Save metadata
    save_visualization_metadata(Config, timestamp, Config.VISUALIZATION_DIR)
    
    print(f"Fine-tuning completed. Results saved with timestamp: {timestamp}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LSOformer with ControlNet approach")
    parser.add_argument("--pretrained_model_path", type=str, required=True, 
                        help="Path to pre-trained LSOformer model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=50, 
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Patience for early stopping")
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Weight for combining outputs (0-1)")
    parser.add_argument("--cuda_device", type=int, default=0, 
                        help="CUDA device index to use")
    
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        Config.CUDA_DEVICE = args.cuda_device
        
    finetune(args)
