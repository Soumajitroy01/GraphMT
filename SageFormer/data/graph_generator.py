import os
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

def generate_graph_from_bench(bench_file, output_file=None):
    """
    Generate a PyG graph from a .bench circuit file.
    
    Args:
        bench_file: Path to .bench file
        output_file: Optional path to save the graph
        
    Returns:
        PyG Data object
    """
    # Parse the .bench file
    with open(bench_file, 'r') as f:
        lines = f.readlines()
    
    # Extract nodes and connections
    inputs = []
    outputs = []
    gates = {}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('INPUT'):
            # Extract input node
            node = line[line.find('(')+1:line.find(')')]
            inputs.append(node)
        
        elif line.startswith('OUTPUT'):
            # Extract output node
            node = line[line.find('(')+1:line.find(')')]
            outputs.append(node)
        
        elif '=' in line:
            # Extract gate
            parts = line.split('=')
            target = parts[0].strip()
            
            # Extract gate type and inputs
            gate_def = parts[1].strip()
            gate_type = gate_def[:gate_def.find('(')]
            gate_inputs = gate_def[gate_def.find('(')+1:gate_def.find(')')].split(',')
            gate_inputs = [inp.strip() for inp in gate_inputs]
            
            gates[target] = {
                'type': gate_type,
                'inputs': gate_inputs
            }
    
    # Create node mapping
    node_map = {}
    node_features = []
    
    # Add inputs
    for i, node in enumerate(inputs):
        node_map[node] = i
        # Feature: [1, 0, 0, 0] for input nodes (4 features as per Table 1)
        node_features.append([1, 0, 0, 0])
    
    # Add gates
    for i, (node, gate) in enumerate(gates.items()):
        node_map[node] = i + len(inputs)
        
        # Feature: [0, 0, 1, 0] for AND gates, [0, 0, 0, 1] for other gates
        if gate['type'].upper() == 'AND':
            node_features.append([0, 0, 1, 0])
        else:
            node_features.append([0, 0, 0, 1])
    
    # Add outputs (if not already added as gates)
    for node in outputs:
        if node not in node_map:
            node_map[node] = len(node_map)
            # Feature: [0, 1, 0, 0] for output nodes
            node_features.append([0, 1, 0, 0])
    
    # Create edges
    edges = []
    
    # Add edges from gate inputs to gates
    for target, gate in gates.items():
        target_idx = node_map[target]
        for input_node in gate['inputs']:
            # Handle inverted inputs (remove NOT)
            if input_node.startswith('!'):
                input_node = input_node[1:]
            
            if input_node in node_map:
                source_idx = node_map[input_node]
                edges.append((source_idx, target_idx))
    
    # Create PyG Data object
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    # Save to file if specified
    if output_file:
        torch.save(data, output_file)
    
    return data

def generate_all_graphs(bench_dir, output_dir):
    """
    Generate graphs for all .bench files in a directory.
    
    Args:
        bench_dir: Directory containing .bench files
        output_dir: Directory to save graph files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .bench files
    bench_files = [f for f in os.listdir(bench_dir) if f.endswith('.bench')]
    
    # Generate graphs for each file
    for bench_file in tqdm(bench_files, desc="Generating graphs"):
        input_path = os.path.join(bench_dir, bench_file)
        output_path = os.path.join(output_dir, bench_file.replace('.bench', '.pt'))
        
        try:
            generate_graph_from_bench(input_path, output_path)
        except Exception as e:
            print(f"Error processing {bench_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph representations from .bench files')
    parser.add_argument('--bench_dir', type=str, required=True, help='Directory containing .bench files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save graph files')
    parser.add_argument('--single_file', type=str, help='Process a single .bench file')
    args = parser.parse_args()
    
    if args.single_file:
        # Process a single file
        input_path = os.path.join(args.bench_dir, args.single_file)
        output_path = os.path.join(args.output_dir, args.single_file.replace('.bench', '.pt'))
        
        print(f"Processing {args.single_file}...")
        generate_graph_from_bench(input_path, output_path)
        print(f"Graph saved to {output_path}")
    else:
        # Process all .bench files in the directory
        print(f"Processing all .bench files in {args.bench_dir}...")
        generate_all_graphs(args.bench_dir, args.output_dir)
        print("All graphs generated successfully")
