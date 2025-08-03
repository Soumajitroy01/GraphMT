# data/dataset.py
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

class CircuitGraphDataset(Dataset):
    """Dataset for circuit graphs and optimization sequences."""
    
    def __init__(self, csv_file, graph_dir, transform=None, max_seq_len=20):
        """
        Args:
            csv_file (string): Path to CSV file with recipes
            graph_dir (string): Directory with graph .pt files
            transform (callable, optional): Optional transform to be applied
            max_seq_len (int): Maximum sequence length for recipes
        """
        self.data_df = pd.read_csv(csv_file)
        self.graph_dir = graph_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Build vocabulary dynamically from the CSV
        self.vocab = self._build_vocabulary_from_csv()
        
        # Cache for graph data
        self.graph_cache = {}
        
        print(f"Dataset initialized with {len(self.data_df)} samples and vocabulary size {len(self.vocab)}")
        
    def _build_vocabulary_from_csv(self):
        """Build static vocabulary for synthesis recipes with all parameter combinations."""
        vocab = {"<PAD>": 0, "<UNK>": 1}
        token_idx = 2
        
        # Basic commands
        basic_commands = ["b", "rw", "rf", "rs", "rwz", "rfz", "rsz"]
        
        # Add basic commands without options
        for cmd in basic_commands:
            vocab[cmd] = token_idx
            token_idx += 1
            # With -l option
            vocab[f"{cmd} -l"] = token_idx
            token_idx += 1
        
        # For rs and rsz, add combinations with -K and -N
        for cmd in ["rs", "rsz"]:
            # Just -K options (even numbers from 4 to 16)
            for k in range(4, 17, 2):  # 4, 6, 8, 10, 12, 14, 16
                # Without -l
                vocab[f"{cmd} -K {k}"] = token_idx
                token_idx += 1
                # With -l
                vocab[f"{cmd} -K {k} -l"] = token_idx
                token_idx += 1
            
            # Just -N options (1 to 3)
            for n in range(1, 4):  # 1, 2, 3
                # Without -l
                vocab[f"{cmd} -N {n}"] = token_idx
                token_idx += 1
                # With -l
                vocab[f"{cmd} -N {n} -l"] = token_idx
                token_idx += 1
            
            # Both -K and -N options together
            for k in range(4, 17, 2):
                for n in range(1, 4):
                    # Without -l
                    vocab[f"{cmd} -K {k} -N {n}"] = token_idx
                    token_idx += 1
                    # With -l
                    vocab[f"{cmd} -K {k} -N {n} -l"] = token_idx
                    token_idx += 1
        
        print(f"Built static vocabulary with {len(vocab)} tokens")
        return vocab

    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get design file name and recipe
        design_name = self.data_df.iloc[idx]['design']
        recipe = self.data_df.iloc[idx]['recipe']
        
        # Replace .bench with .pt to get graph filename
        graph_file = design_name.replace('.bench', '.pt')
        graph_path = os.path.join(self.graph_dir, graph_file)
        
        # Load graph data from cache if available
        if graph_path in self.graph_cache:
            graph_data = self.graph_cache[graph_path]
        else:
            # Load from disk if not cached
            try:
                graph_data = torch.load(graph_path)
                # Cache it for future use
                self.graph_cache[graph_path] = graph_data
            except FileNotFoundError:
                print(f"Warning: Graph file not found: {graph_path}")
                # Create a dummy graph
                graph_data = Data(
                    x=torch.zeros((1, 4)),  # 4 features as per Table 1
                    edge_index=torch.zeros((2, 0), dtype=torch.long)
                )
        
        # Store the original index and design name in the graph data
        graph_data.original_idx = idx
        graph_data.design_name = design_name
        graph_data.recipe_str = recipe
        
        # Tokenize recipe - ensure it has shape [seq_len]
        recipe_tokens = self._tokenize_recipe(recipe)
        
        # Add recipe to graph data
        graph_data.recipe = recipe_tokens
        
        # Get target values
        if 'nodes' in self.data_df.columns:
            graph_data.nodes = torch.tensor([self.data_df.iloc[idx]['nodes']], dtype=torch.float)
        if 'levels' in self.data_df.columns:
            graph_data.levels = torch.tensor([self.data_df.iloc[idx]['levels']], dtype=torch.float)
        if 'iterations' in self.data_df.columns:
            graph_data.iterations = torch.tensor([self.data_df.iloc[idx]['iterations']], dtype=torch.float)
        
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data
    
    def _tokenize_recipe(self, recipe):
        """Convert recipe string to token indices with fixed length."""
        # Split the recipe into commands
        commands = recipe.split(';')
        
        # Convert commands to indices, padding to max_seq_len
        tokens = []
        for cmd in commands[:self.max_seq_len]:  # Limit to max_seq_len commands
            cmd = cmd.strip()
            if cmd in self.vocab:
                tokens.append(self.vocab[cmd])
            else:
                tokens.append(self.vocab["<UNK>"])  # Use UNK for unknown commands
        
        # Pad to max_seq_len
        tokens = tokens + [self.vocab["<PAD>"]] * (self.max_seq_len - len(tokens))
        
        # Return tensor of shape [seq_len]
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    from torch_geometric.data import Batch
    
    # Ensure all recipes have the correct shape before batching
    for item in batch:
        # Check recipe shape
        if not hasattr(item, 'recipe') or item.recipe is None:
            # Create a default recipe tensor if missing
            item.recipe = torch.zeros(20, dtype=torch.long)
        elif item.recipe.dim() == 1:
            # Ensure it has the correct sequence length (20)
            if item.recipe.size(0) != 20:
                if item.recipe.size(0) < 20:
                    # Pad with zeros (PAD token)
                    padding = torch.zeros(20 - item.recipe.size(0), dtype=torch.long)
                    item.recipe = torch.cat([item.recipe, padding], dim=0)
                else:
                    # Truncate to 20
                    item.recipe = item.recipe[:20]
    
    # Create batched data
    batched_data = Batch.from_data_list(batch)
    
    # Debug: Print shapes after batching
    print(f"Batched recipe shape: {batched_data.recipe.shape}")
    print(f"Batched x shape: {batched_data.x.shape}")
    print(f"Batched edge_index shape: {batched_data.edge_index.shape}")
    
    return batched_data
