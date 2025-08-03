# visualize_results.py
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config
from utils.visualization import (
    plot_training_curves,
    plot_qor_trajectory,
    plot_mape_by_circuit,
    plot_comparison_heatmap
)
from utils.helpers import get_timestamp

def visualize_training_curves(args):
    """Visualize training curves from saved losses"""
    # Load losses
    with open(args.losses_path, 'rb') as f:
        losses = pickle.load(f)
    
    train_losses = losses['train_loss']
    val_losses = losses['val_loss']
    
    # Plot training curves
    timestamp = get_timestamp()
    plot_path = os.path.join(Config.VISUALIZATION_DIR, f"training_curves_{timestamp}.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    
    print(f"Training curves visualization saved to {plot_path}")

def visualize_results(args):
    """Visualize results from saved inference results"""
    # Load results
    with open(args.results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create timestamp
    timestamp = get_timestamp()
    
    # 1. Plot MAPE by circuit
    mape_plot_path = os.path.join(Config.VISUALIZATION_DIR, f"mape_by_circuit_{timestamp}.png")
    plot_mape_by_circuit(results['design_mapes'], results['design_names'], 
                         metric_name=args.metric, save_path=mape_plot_path)
    
    # 2. Plot QoR trajectories for selected designs
    if args.design_names:
        os.makedirs(os.path.join(Config.VISUALIZATION_DIR, "trajectories"), exist_ok=True)
        
        for design in args.design_names:
            # Find all occurrences of this design
            indices = [i for i, d in enumerate(results['all_design_names']) if d == design]
            
            if indices:
                # Use the first occurrence
                idx = indices[0]
                
                # Plot trajectory
                traj_path = os.path.join(Config.VISUALIZATION_DIR, "trajectories", 
                                        f"trajectory_{design}_{timestamp}.png")
                plot_qor_trajectory(
                    results['all_trajectories_true'][idx],
                    results['all_trajectories_pred'][idx],
                    design,
                    save_path=traj_path
                )
            else:
                print(f"Design {design} not found in results")
    
    # 3. Plot error distribution
    errors = np.abs(np.array(results['all_preds']) - np.array(results['all_targets']))
    rel_errors = errors / np.array(results['all_targets']) * 100
    
    plt.figure(figsize=(10, 6))
    sns.histplot(rel_errors, kde=True)
    plt.title(f'Distribution of Relative Errors ({args.metric})')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    
    error_dist_path = os.path.join(Config.VISUALIZATION_DIR, f"error_distribution_{timestamp}.png")
    plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(results['all_targets'], results['all_preds'], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(results['all_targets']), min(results['all_preds']))
    max_val = max(max(results['all_targets']), max(results['all_preds']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Predicted vs Actual {args.metric}')
    plt.xlabel(f'Actual {args.metric}')
    plt.ylabel(f'Predicted {args.metric}')
    
    scatter_path = os.path.join(Config.VISUALIZATION_DIR, f"scatter_plot_{timestamp}.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results visualization saved to {Config.VISUALIZATION_DIR}")

def compare_models(args):
    """Compare multiple models using heatmap"""
    # Load results for each model
    model_results = {}
    
    for model_path in args.model_results:
        with open(model_path, 'rb') as f:
            results = pickle.load(f)
        
        model_name = os.path.basename(model_path).replace('results_', '').replace('.pkl', '')
        model_results[model_name] = results['design_mapes']
    
    # Create DataFrame
    df = pd.DataFrame(model_results, index=results['design_names'])
    
    # Create timestamp
    timestamp = get_timestamp()
    
    # Plot heatmap
    heatmap_path = os.path.join(Config.VISUALIZATION_DIR, f"model_comparison_{timestamp}.png")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
    
    plt.title(f'Model Comparison - {args.metric} MAPE (%)')
    plt.tight_layout()
    
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison visualization saved to {heatmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LSOformer results")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training curves visualization
    train_parser = subparsers.add_parser("training", help="Visualize training curves")
    train_parser.add_argument("--losses_path", type=str, required=True, 
                             help="Path to saved losses file")
    
    # Results visualization
    results_parser = subparsers.add_parser("results", help="Visualize inference results")
    results_parser.add_argument("--results_path", type=str, required=True,
                               help="Path to saved results file")
    results_parser.add_argument("--metric", type=str, default="Delay",
                               choices=["Delay", "Area"],
                               help="QoR metric name")
    results_parser.add_argument("--design_names", type=str, nargs="+",
                               help="Names of designs to plot trajectories for")
    
    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--model_results", type=str, nargs="+", required=True,
                               help="Paths to saved results files for multiple models")
    compare_parser.add_argument("--metric", type=str, default="Delay",
                               choices=["Delay", "Area"],
                               help="QoR metric name")
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    
    if args.command == "training":
        visualize_training_curves(args)
    elif args.command == "results":
        visualize_results(args)
    elif args.command == "compare":
        compare_models(args)
    else:
        parser.print_help()
