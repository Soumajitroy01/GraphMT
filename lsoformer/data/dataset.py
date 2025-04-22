# data/dataset.py
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class LSODataset(Dataset):
    def __init__(self, csv_path, graph_dir, heuristic_to_idx=None, train=True):
        """
        Dataset for Logic Synthesis Optimization
        
        Args:
            csv_path: Path to CSV file with columns Design, Recipe, Level_1, ..., Level_18
            graph_dir: Directory containing graph files (.pt)
            heuristic_to_idx: Dictionary mapping heuristics to indices
            train: Whether this is training set (to build heuristic_to_idx)
        """
        self.df = pd.read_csv(csv_path)
        self.graph_dir = graph_dir
        self.train = train
        
        # Create heuristic vocabulary if not provided
        if heuristic_to_idx is None and train:
            self.build_heuristic_vocab()
        else:
            self.heuristic_to_idx = heuristic_to_idx
            
        # Normalize QoR values (levels)
        self.normalize_qor()
        
        # Preload all graphs
        self.graph_cache = {}
    
    def build_heuristic_vocab(self):
        """Build vocabulary of heuristics from recipes"""
        all_heuristics = set()
        for recipe in self.df['Recipe']:
            heuristics = recipe.split(';')
            all_heuristics.update(heuristics)
        
        self.heuristic_to_idx = {h: i for i, h in enumerate(sorted(all_heuristics))}
        return self.heuristic_to_idx
    
    def normalize_qor(self):
        """Normalize QoR values (levels)"""
        qor_columns = [f'Level_{i+1}' for i in range(18)]
        qor_values = self.df[qor_columns].values.astype(np.float32)
        
        self.qor_mean = np.mean(qor_values)
        self.qor_std = np.std(qor_values)
        
        # Store normalized values
        for col in qor_columns:
            self.df[f'{col}_norm'] = (self.df[col].astype(np.float32) - self.qor_mean) / self.qor_std
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get design name and load graph
        design_name = row['Design']
        graph = self.load_graph(design_name)
        
        # Process recipe
        recipe = row['Recipe'].split(';')
        recipe_indices = torch.tensor([self.heuristic_to_idx[h] for h in recipe], dtype=torch.long)
        
        # Get QoR trajectory - handle potential object arrays
        qor_columns = [f'Level_{i+1}' for i in range(18)]
        try:
            qor_trajectory = torch.tensor(row[qor_columns].values.astype(np.float32), dtype=torch.float)
        except TypeError:
            # Handle case where values might be mixed types
            qor_values = [float(val) for val in row[qor_columns].values]
            qor_trajectory = torch.tensor(qor_values, dtype=torch.float)
        
        # Get node depths from graph
        node_depths = graph.node_depth
        
        return {
            'design_name': design_name,
            'graph': graph,
            'recipe_indices': recipe_indices,
            'node_depths': node_depths,
            'qor_trajectory': qor_trajectory
        }
    
    def load_graph(self, design_name):
        """Load graph from file with caching"""
        if design_name in self.graph_cache:
            return self.graph_cache[design_name]
            
        graph_path = os.path.join(self.graph_dir, design_name.replace('.bench', '.pt'))
        
        if os.path.exists(graph_path):
            graph = torch.load(graph_path, map_location=torch.device('cpu'))
            self.graph_cache[design_name] = graph  # Cache the graph
            return graph
        else:
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
