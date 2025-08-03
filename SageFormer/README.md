# Traditional Models for QoR Prediction

## Overview

This repository contains implementations of various neural network architectures for Quality of Results (QoR) prediction in logic synthesis. The models include Graph Neural Networks (GNN), Transformer, Long Short-Term Memory (LSTM), and Convolutional Neural Networks (CNN) for processing circuit graphs and optimization sequences.

## Repository Structure

traditional_models/
├── data/
│ ├── init.py
│ ├── dataset.py # Dataset classes for loading graph data
│ ├── preprocessing.py # Preprocessing utilities for circuits and recipes
│ └── graph_generator.py # Utilities for generating graph representations
├── models/
│ ├── init.py
│ ├── gnn/
│ │ ├── init.py
│ │ ├── gcn.py # GCN-based circuit encoder
│ │ ├── gat.py # GAT-based circuit encoder
│ │ └── graphsage.py # GraphSage-based circuit encoder
│ ├── recipe/
│ │ ├── init.py
│ │ ├── transformer.py # Transformer-based recipe encoder
│ │ ├── lstm.py # LSTM-based recipe encoder
│ │ └── cnn.py # CNN-based recipe encoder
│ └── qor_model.py # Combined QoR prediction model
├── utils/
│ ├── init.py
│ ├── config.py # Configuration parameters
│ ├── metrics.py # Evaluation metrics
│ └── visualization.py # Visualization utilities
├── train.py # Training script
└── inference.py # Inference script

text

## Model Architectures

### Graph Neural Networks (GNN)

We implement three types of GNNs for circuit graph encoding:

1. **GCN (Graph Convolutional Network)**
   - Input: Node features (4-dimensional)
   - Hidden layers: 64 units
   - Output: 128-dimensional circuit embedding
   - Pooling: Combination of max and mean pooling

2. **GAT (Graph Attention Network)**
   - Input: Node features (4-dimensional)
   - First layer: 32 units × 2 heads with concatenation
   - Second layer: 64 units with averaging
   - Output: 128-dimensional circuit embedding
   - Pooling: Combination of max and mean pooling

3. **GraphSAGE**
   - Input: Node features (4-dimensional)
   - Hidden layers: 64 units
   - Output: 128-dimensional circuit embedding
   - Pooling: Combination of max and mean pooling

### Recipe Encoders

1. **Transformer**
   - Input: Recipe tokens (sequence length 20)
   - Embedding dimension: 4
   - Number of heads: 2
   - Feedforward dimension: 32
   - Number of layers: 3
   - Output: 50-dimensional recipe embedding

2. **LSTM**
   - Input: Recipe tokens (sequence length 20)
   - Embedding dimension: 3
   - Hidden size: 64
   - Number of layers: 2
   - Output: 64-dimensional recipe embedding

3. **CNN**
   - Input: Embedded recipe (60-dimensional)
   - Filters: 4 parallel convolutions
   - Kernel sizes: 21, 24, 27, 30
   - Stride: 3
   - Output: 50-dimensional recipe embedding

### Combined QoR Model

The QoR prediction model combines circuit embeddings from GNNs with recipe embeddings from sequence models:

- Input: Circuit graph + optimization recipe
- Circuit encoder: GCN, GAT, or GraphSAGE
- Recipe encoder: Transformer, LSTM, or CNN
- Fully connected layers: [512] → [256] → [256]
- Dropout: 0.2
- Output: QoR prediction (nodes, levels, or iterations)

## Dataset

The dataset consists of:
- Circuit graphs in PyTorch Geometric format
- Optimization recipes (sequences of synthesis commands)
- QoR metrics (nodes, levels, iterations)

The vocabulary for tokenizing recipes includes:
- Basic commands (b, rw, rf, rs, rwz, rfz, rsz)
- Commands with -l option
- rs and rsz with -K options (even numbers 4-16)
- rs and rsz with -N options (1-3)
- All combinations of the above

## Usage

### Training

python train.py --csv_file data/results.csv --graph_dir data/graphs --output_dir output --gnn_type graphsage --recipe_type transformer --target nodes

text

### Inference

python inference.py --csv_file data/test_results.csv --graph_dir data/graphs --output_dir output/evaluation --model_path output/graphsage_transformer_nodes/best_model.pt

text

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm
