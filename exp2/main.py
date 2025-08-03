import os
import argparse
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

def parse_bench_file(bench_file):
    """
    Parse a .bench file and extract nodes and connections.
    Modified to handle explicit NOT gates rather than ! prefix.
    
    Args:
        bench_file (str): Path to the .bench file
        
    Returns:
        dict: Dictionary of nodes and their connections
        list: List of input nodes
        list: List of output nodes
    """
    nodes = {}
    inputs = []
    outputs = []
    
    with open(bench_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Parse input nodes
            if line.startswith('INPUT'):
                node_name = line.split('(')[1].split(')')[0]
                inputs.append(node_name)
                nodes[node_name] = {'type': 0, 'inputs': []}
                
            # Parse output nodes
            elif line.startswith('OUTPUT'):
                node_name = line.split('(')[1].split(')')[0]
                outputs.append(node_name)
                
            # Parse AND gates
            elif ' = AND(' in line:
                parts = line.split(' = AND(')
                node_name = parts[0].strip()
                input_nodes = parts[1].split(')')[0].split(',')
                
                # Clean up input nodes (remove whitespace)
                clean_inputs = [inp.strip() for inp in input_nodes]
                
                nodes[node_name] = {
                    'type': 1,  # AND gate
                    'inputs': clean_inputs,
                    'is_and': True
                }
                
            # Parse NOT gates
            elif ' = NOT(' in line:
                parts = line.split(' = NOT(')
                node_name = parts[0].strip()
                input_node = parts[1].split(')')[0].strip()
                
                nodes[node_name] = {
                    'type': 1,  # Intermediate node
                    'inputs': [input_node],
                    'is_not': True
                }
    
    return nodes, inputs, outputs

def create_graph(nodes, inputs, outputs):
    """
    Create a NetworkX graph from parsed .bench data.
    Modified to handle explicit NOT gates.
    
    Args:
        nodes (dict): Dictionary of nodes and their connections
        inputs (list): List of input nodes
        outputs (list): List of output nodes
        
    Returns:
        nx.DiGraph: Directed graph representation
    """
    G = nx.DiGraph()
    
    # Add nodes with features
    for node_name, node_data in nodes.items():
        # Set node type: 0 for input, 1 for intermediate, 2 for output
        node_type = node_data['type']
        if node_name in outputs:
            node_type = 2
            
        # Add node to graph
        G.add_node(node_name, type=node_type)
        
        # Set additional attributes for AND and NOT gates
        if node_data.get('is_and', False):
            G.nodes[node_name]['gate_type'] = 'AND'
            G.nodes[node_name]['inverted_count'] = 0
        elif node_data.get('is_not', False):
            G.nodes[node_name]['gate_type'] = 'NOT'
    
    # Add edges
    for node_name, node_data in nodes.items():
        for input_node in node_data['inputs']:
            # Add edge from input to current node
            G.add_edge(input_node, node_name)
            
            # Mark edge as inverted if the source is a NOT gate
            if input_node in nodes and nodes.get(input_node, {}).get('is_not', False):
                G.edges[input_node, node_name]['inverted'] = 1
            else:
                G.edges[input_node, node_name]['inverted'] = 0
    
    return G

def compute_node_depths(G):
    """
    Compute depth for each node in the graph.
    
    Args:
        G (nx.DiGraph): Directed graph
        
    Returns:
        dict: Dictionary mapping node names to depths
    """
    # Find input nodes (nodes with in-degree 0)
    input_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    
    # Initialize depths
    depths = {node: 0 for node in input_nodes}
    
    # Perform topological sort
    for node in nx.topological_sort(G):
        if node not in depths:
            # Compute depth as max depth of predecessors + 1
            pred_depths = [depths[pred] for pred in G.predecessors(node) if pred in depths]
            depths[node] = max(pred_depths) + 1 if pred_depths else 0
    
    return depths

def count_inverted_inputs(G, node):
    """
    Count the number of inverted inputs for a node.
    An input is considered inverted if it comes from a NOT gate.
    
    Args:
        G (nx.DiGraph): Directed graph
        node (str): Node name
        
    Returns:
        int: Number of inverted inputs
    """
    inverted_count = 0
    for pred in G.predecessors(node):
        if G.nodes[pred].get('gate_type') == 'NOT':
            inverted_count += 1
    return inverted_count

def convert_to_pytorch_geometric(G, depths):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    Modified to handle explicit NOT gates.
    
    Args:
        G (nx.DiGraph): Directed graph
        depths (dict): Dictionary mapping node names to depths
        
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Create node mapping
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Extract node features
    num_nodes = len(G.nodes())
    node_features = torch.zeros((num_nodes, 2), dtype=torch.float)
    node_depth_list = torch.zeros(num_nodes, dtype=torch.long)
    
    for node, idx in node_mapping.items():
        node_data = G.nodes[node]
        
        # Set node type (0: input, 1: intermediate, 2: output)
        node_features[idx, 0] = node_data.get('type', 1)
        
        # Count inverted inputs for AND gates
        if node_data.get('gate_type') == 'AND':
            inverted_count = count_inverted_inputs(G, node)
            node_features[idx, 1] = inverted_count
        
        # Set node depth
        node_depth_list[idx] = depths[node]
    
    # Extract edge indices and attributes
    edge_index = []
    edge_attr = []
    
    for src, dst, data in G.edges(data=True):
        src_idx = node_mapping[src]
        dst_idx = node_mapping[dst]
        edge_index.append([src_idx, dst_idx])
        
        # Set edge attribute (inverted or not)
        is_inverted = data.get('inverted', 0)
        edge_attr.append([is_inverted])
    
    if edge_index:  # Check if there are any edges
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Create empty tensors if no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_depth=node_depth_list
    )
    
    return data

def bench_to_graph(bench_file, output_dir, output_format='pt'):
    """
    Convert a .bench file to a graph format.
    
    Args:
        bench_file (str): Path to the .bench file
        output_dir (str): Directory to save the output
        output_format (str): Output format ('pt' for PyTorch, 'graphml' for GraphML)
        
    Returns:
        str: Path to the output file
    """
    # Parse the bench file
    nodes, inputs, outputs = parse_bench_file(bench_file)
    
    # Create a graph
    G = create_graph(nodes, inputs, outputs)
    
    # Compute node depths
    depths = compute_node_depths(G)
    
    # Create output filename
    base_name = os.path.basename(bench_file).replace('.bench', '')
    
    if output_format == 'pt':
        # Convert to PyTorch Geometric format
        data = convert_to_pytorch_geometric(G, depths)
        
        # Save as PyTorch file
        output_file = os.path.join(output_dir, f"{base_name}.pt")
        torch.save(data, output_file)
    
    elif output_format == 'graphml':
        # Add depth as node attribute
        for node, depth in depths.items():
            G.nodes[node]['depth'] = depth
            
        # Save as GraphML
        output_file = os.path.join(output_dir, f"{base_name}.graphml")
        nx.write_graphml(G, output_file)
    
    return output_file

def process_directory(input_dir, output_dir, output_format='pt', num_workers=1):
    """
    Process all .bench files in a directory.
    
    Args:
        input_dir (str): Directory containing .bench files
        output_dir (str): Directory to save the output files
        output_format (str): Output format ('pt' for PyTorch, 'graphml' for GraphML)
        num_workers (int): Number of parallel workers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .bench files
    bench_files = [f for f in os.listdir(input_dir) if f.endswith('.bench')]
    
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for bench_file in bench_files:
                input_path = os.path.join(input_dir, bench_file)
                futures.append(executor.submit(bench_to_graph, input_path, output_dir, output_format))
            
            # Show progress
            for _ in tqdm(futures, total=len(futures), desc="Converting files"):
                pass
    else:
        # Process files sequentially
        for bench_file in tqdm(bench_files, desc="Converting files"):
            input_path = os.path.join(input_dir, bench_file)
            bench_to_graph(input_path, output_dir, output_format)

def main():
    parser = argparse.ArgumentParser(description='Convert AIG .bench files to graph format')
    parser.add_argument('--input', type=str, required=True, help='Input .bench file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--format', type=str, choices=['pt', 'graphml'], default='pt',
                        help='Output format (pt for PyTorch, graphml for GraphML)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.format, args.workers)
    else:
        bench_to_graph(args.input, args.output, args.format)
        print(f"Converted {args.input} to {args.format} format")

if __name__ == "__main__":
    main()
