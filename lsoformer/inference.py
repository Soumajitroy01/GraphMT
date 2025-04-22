# inference.py
import os
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm

from config import Config
from data.dataset import LSODataset
from data.dataloader import create_dataloaders, create_inductive_dataloaders
from models.lsoformer import LSOformer
from utils.metrics import mean_absolute_percentage_error
from utils.visualization import (
    plot_qor_trajectory, 
    plot_mape_by_circuit,
    plot_comparison_heatmap
)
from utils.helpers import set_seed, get_timestamp, load_model, move_batch_to_device, empty_cuda_cache

def denormalize_qor(qor_normalized, mean, std):
    """Denormalize QoR values"""
    return qor_normalized * std + mean

def evaluate_model(model, dataloader, device, dataset):
    """Evaluate model on dataloader"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_design_names = []
    all_trajectories_pred = []
    all_trajectories_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch_device = move_batch_to_device(batch, device)
            
            # Forward pass
            pred_trajectory = model(batch_device)
            
            # Store predictions and targets
            for i in range(len(batch['design_name'])):
                design_name = batch['design_name'][i]
                pred_traj = pred_trajectory[i].cpu().numpy()
                true_traj = batch['qor_trajectory'][i].cpu().numpy()
                
                # Denormalize
                pred_traj = denormalize_qor(pred_traj, dataset.qor_mean, dataset.qor_std)
                true_traj = denormalize_qor(true_traj, dataset.qor_mean, dataset.qor_std)
                
                # Store final QoR
                all_preds.append(pred_traj[-1])
                all_targets.append(true_traj[-1])
                all_design_names.append(design_name)
                
                # Store trajectories
                all_trajectories_pred.append(pred_traj)
                all_trajectories_true.append(true_traj)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(all_targets, all_preds)
    
    # Calculate MAPE per circuit
    unique_designs = []
    design_mapes = []
    
    for design in set(all_design_names):
        indices = [i for i, d in enumerate(all_design_names) if d == design]
        design_preds = [all_preds[i] for i in indices]
        design_targets = [all_targets[i] for i in indices]
        
        design_mape = mean_absolute_percentage_error(design_targets, design_preds)
        unique_designs.append(design)
        design_mapes.append(design_mape)
    
    return {
        'mape': mape,
        'design_names': unique_designs,
        'design_mapes': design_mapes,
        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_design_names': all_design_names,
        'all_trajectories_pred': all_trajectories_pred,
        'all_trajectories_true': all_trajectories_true
    }

def inference(args):
    """Main inference function"""
    # Create directories
    Config.create_dirs()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Create timestamp for this run
    timestamp = get_timestamp()
    
    # Load heuristic vocabulary
    with open(args.vocab_path, 'rb') as f:
        heuristic_to_idx = pickle.load(f)
    
    # Load dataset
    print("Loading dataset...")
    dataset = LSODataset(Config.CSV_PATH, Config.GRAPH_DIR, heuristic_to_idx, train=False)
    
    # Create dataloaders
    if args.inductive:
        # For inductive setup, we need to specify test designs
        with open(args.test_designs, 'r') as f:
            test_designs = [line.strip() for line in f.readlines()]
        
        _, test_loader = create_inductive_dataloaders(
            dataset, Config.BATCH_SIZE, test_designs)
    else:
        _, _, test_loader = create_dataloaders(
            dataset, Config.BATCH_SIZE)
    
    # Get input dimension from first graph
    sample_batch = next(iter(test_loader))
    input_node_dim = sample_batch['graph'].x.size(1)
    
    # Create model
    print("Creating model...")
    model = LSOformer(
        input_node_dim=input_node_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_heuristics=len(heuristic_to_idx),
        nhead=Config.NUM_HEADS,
        dim_feedforward=Config.TRANSFORMER_DIM * 4,
        dropout=Config.DROPOUT,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # Load model
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    model, _, _, _ = load_model(model, optimizer, args.model_path, device)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, dataset)
    
    # Empty CUDA cache if needed
    empty_cuda_cache()
    
    # Print overall MAPE
    print(f"Overall MAPE: {results['mape']:.2f}%")
    
    # Save results
    results_path = os.path.join(Config.RESULTS_DIR, f"results_{timestamp}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Visualize results
    
    # 1. Plot MAPE by circuit
    mape_plot_path = os.path.join(Config.VISUALIZATION_DIR, f"mape_by_circuit_{timestamp}.png")
    plot_mape_by_circuit(results['design_mapes'], results['design_names'], 
                         metric_name=args.metric, save_path=mape_plot_path)
    
    # 2. Plot QoR trajectories for a few designs
    if args.plot_trajectories:
        os.makedirs(os.path.join(Config.VISUALIZATION_DIR, "trajectories"), exist_ok=True)
        
        # Get unique designs
        unique_designs = list(set(results['all_design_names']))
        
        # Plot for a subset of designs
        for design in unique_designs[:min(5, len(unique_designs))]:
            # Find first occurrence of this design
            idx = results['all_design_names'].index(design)
            
            # Plot trajectory
            traj_path = os.path.join(Config.VISUALIZATION_DIR, "trajectories", 
                                    f"trajectory_{design}_{timestamp}.png")
            plot_qor_trajectory(
                results['all_trajectories_true'][idx],
                results['all_trajectories_pred'][idx],
                design,
                save_path=traj_path
            )
    
    print(f"Inference completed. Results saved with timestamp: {timestamp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with LSOformer model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to heuristic vocabulary")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--inductive", action="store_true", 
                        help="Use inductive setup")
    parser.add_argument("--test_designs", type=str, default="test_designs.txt",
                        help="File with test design names for inductive setup")
    parser.add_argument("--metric", type=str, default="Delay", 
                        choices=["Delay", "Area"],
                        help="QoR metric name")
    parser.add_argument("--plot_trajectories", action="store_true",
                        help="Plot QoR trajectories")
    parser.add_argument("--cuda_device", type=int, default=0, 
                        help="CUDA device index to use")
    
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        Config.CUDA_DEVICE = args.cuda_device
        
    inference(args)
