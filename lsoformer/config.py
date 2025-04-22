# config.py
import os
import torch

class Config:
    # Data paths
    DATA_DIR = "data_files"
    GRAPH_DIR = os.path.join(DATA_DIR, "graph")
    CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
    
    # Model parameters
    HIDDEN_DIM = 32  # Hidden dimension for graph encoder
    TRANSFORMER_DIM = 64  # Dimension for transformer (2*HIDDEN_DIM)
    NUM_HEADS = 8  # Number of attention heads
    NUM_LAYERS = 6  # Number of transformer layers
    DROPOUT = 0.1  # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 10
    
    # CUDA parameters
    USE_CUDA = torch.cuda.is_available()
    CUDA_DEVICE = 0  # Default CUDA device index
    NUM_WORKERS = 4 if USE_CUDA else 0  # Number of data loading workers
    PIN_MEMORY = True if USE_CUDA else False  # Pin memory for faster GPU transfer
    
    # CUDA memory management
    CUDA_EMPTY_CACHE = True  # Whether to empty CUDA cache after each epoch
    
    # Recipe parameters
    MAX_RECIPE_LENGTH = 18
    
    # Paths for saving
    SAVE_DIR = "saved_models"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    VISUALIZATION_DIR = "visualizations"
    
    @staticmethod
    def get_device():
        """Get the appropriate device (CPU or CUDA)"""
        if Config.USE_CUDA:
            return torch.device(f'cuda:{Config.CUDA_DEVICE}')
        return torch.device('cpu')
    
    # Create directories if they don't exist
    @staticmethod
    def create_dirs():
        dirs = [Config.SAVE_DIR, Config.LOGS_DIR, Config.RESULTS_DIR, Config.VISUALIZATION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
