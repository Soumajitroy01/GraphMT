# inference.py
import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset import CircuitGraphDataset, collate_fn
from models.qor_model import QoRPredictionModel
from utils.config import Config
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with QoR prediction model')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with results')
    parser.add_argument('--graph_dir', type=str, help='Directory with graph files')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--gnn_type', type=str, choices=['gcn', 'gat', 'graphsage'], help='GNN type')
    parser.add_argument('--recipe_type', type=str, choices=['transformer', 'lstm', 'cnn'], help='Recipe encoder type')
    parser.add_argument('--target', type=str, choices=['nodes', 'levels', 'iterations'], help='Target metric')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    return parser.parse_args()

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
    
    # Set model path
    model_path = args.model_path if args.model_path else os.path.join(
        config.output_dir, 
        f"{config.gnn_type}_{config.recipe_type}_{config.target_metric}", 
        "best_model.pt"
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set device
    device = config.device
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = CircuitGraphDataset(csv_file=config.csv_file, graph_dir=config.graph_dir)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        gnn_type = model_config.get('gnn_type', config.gnn_type)
        recipe_type = model_config.get('recipe_type', config.recipe_type)
        target_metric = model_config.get('target_metric', config.target_metric)
    else:
        gnn_type = config.gnn_type
        recipe_type = config.recipe_type
        target_metric = config.target_metric
    
    # Initialize model
    model = QoRPredictionModel(
        node_feature_dim=dataset[0].x.size(1),
        vocab_size=len(dataset.vocab),
        gnn_type=gnn_type,
        recipe_type=recipe_type,
        target=target_metric
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model: {gnn_type} + {recipe_type} for {target_metric} prediction")
    
    # Run inference
    all_preds = []
    all_targets = []
    all_circuits = []
    all_recipes = []

    with torch.no_grad():
        for data in tqdm(loader, desc="inference"):
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
            
            # Track predictions and targets
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())
            
            # Track circuit names and recipes using the stored values
            for i in range(data.num_graphs):
                # Access the stored design name and recipe directly
                if hasattr(data, 'design_name'):
                    # If design_name is a list or tensor
                    if isinstance(data.design_name, list) or torch.is_tensor(data.design_name):
                        circuit_name = data.design_name[i]
                        recipe = data.recipe_str[i]
                    else:
                        # If we have a batch with only one graph
                        circuit_name = data.design_name
                        recipe = data.recipe_str
                else:
                    # Fallback to using batch index (not ideal)
                    print(f"Warning: design_name not found in graph data. Using batch index {i} instead.")
                    circuit_name = f"unknown_circuit_{i}"
                    recipe = "unknown_recipe"
                
                all_circuits.append(circuit_name)
                all_recipes.append(recipe)
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'circuit': all_circuits,
        'recipe': all_recipes,
        'pred': all_preds.numpy().flatten(),
        'true': all_targets.numpy().flatten()
    })
    
    # Save results
    results_df.to_csv(os.path.join(config.output_dir, 'inference_results.csv'), index=False)
    
    # Print metrics
    print("\nInference Metrics:")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'model': f"{gnn_type}+{recipe_type}",
        'target': target_metric,
        'mae': metrics['mae'],
        'mse': metrics['mse'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'mape': metrics['mape']
    }])
    
    metrics_df.to_csv(os.path.join(config.output_dir, 'inference_metrics.csv'), index=False)

if __name__ == "__main__":
    main()
