#process.py
import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
import argparse
import dgl
import torch
import numpy as np

# Force unbuffered output
def preprocess_dataset(cfg):
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    print("Script starting - testing output...")

    # Load graph data
    print(f"Loading graph from {cfg.dataset.graph_path}...")
    graph_list, _ = dgl.load_graphs(cfg.dataset.graph_path)
    g = graph_list[0]

    # Print basic graph info
    print("Graph loaded successfully")
    print(f"Graph info: {g}")
    print(f"Number of nodes: {g.num_nodes()}")
    print(f"Number of edges: {g.num_edges()}")

    # Load CSV for text info
    print(f"Loading node text info from {cfg.dataset.text_path}...")
    df = pd.read_csv(cfg.dataset.text_path)

    print(f"CSV loaded successfully, total rows: {len(df)}")
    
    # Create mapping from node ID to text
    node_to_text = {}
    # Create mapping from node_id to asin
    node_to_asin = {}
    
    for i, row in df.iterrows():
        if cfg.dataset.name in {"Toys", "Movies", "Grocery", "RedditS"}:
            node_id = row["id"]
            text = str(row["text"])
        elif cfg.dataset.name in {"books-nc", "cloth", "ele-fashion", "sports"}:
            node_id = i
            text = str(row["text"])
        elif cfg.dataset.name in {"Bili_Cartoon", "Bili_Dance", "Bili_Food", "Bili_Movie", "Bili_Music", "DY", "QB", "KU", "TN"}:
            node_id = row["id"]
            text = str(row["text_en"])
        elif cfg.dataset.name == "Flickr30k":
            node_id = i
            text = str(row["description1"])
        elif cfg.dataset.name == "SemArt":
            node_id = i
            
            title = str(row.get('TITLE', '')).strip()
            desc = str(row.get('DESCRIPTION', '')).strip()
            
            parts = []
            if title: parts.append(f"Title: {title}")
            if desc: parts.append(f"Description: {desc}")
            text = ". ".join(parts)
        
        node_to_text[node_id] = text
        node_to_asin[node_id] = str(row["id"])
    
    print(f"Created text mapping for {len(node_to_text)} nodes")
    print(f"Created asin mapping for {len(node_to_asin)} nodes")
    
    # Save node_2_asin.pt
    processed_dir = Path(cfg.task.save_data_path) / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    asin_path = processed_dir / "node_2_asin.pt"
    import torch
    torch.save(node_to_asin, asin_path)
    print(f"Saved node_2_asin.pt to {asin_path}")

    # Get all node IDs
    all_nodes = list(range(g.num_nodes()))

    # Randomly shuffle node IDs
    random.seed(42)  # For reproducibility
    random.shuffle(all_nodes)

    # Split train and test sets by ratio
    split_idx = int(len(all_nodes) * cfg.task.train_ratio)
    train_nodes = all_nodes[:split_idx]
    test_nodes = all_nodes[split_idx:]

    print(f"Train set size: {len(train_nodes)}")
    print(f"Test set size: {len(test_nodes)}")

    # Create output directories (if not exist)
    train_dir = Path(cfg.task.save_data_path + "/train")
    test_dir = Path(cfg.task.save_data_path + "/test")
    
    print(f"Creating directories: {train_dir} and {test_dir}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Function to create node metadata and write to JSONL
    def write_nodes_to_jsonl(nodes, output_path):
        print(f"Writing {len(nodes)} nodes to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                for node_id in nodes:
                    # Get node features
                    node_data = {}
                    
                    # Set node ID as center field
                    node_data["center"] = str(node_id)
                    
                    # Get text info from mapping
                    if node_id in node_to_text:
                        node_data["text"] = node_to_text[node_id]
                    else:
                        # Use default if text not found in CSV
                        node_data["text"] = f"Node {node_id}"
                    
                    # Get neighbors (using outgoing edges of directed graph)
                    neighbors = g.predecessors(node_id).tolist()
                    node_data["neighbors"] = [str(n) for n in neighbors]
                    
                    # Write to JSONL file
                    f.write(json.dumps(node_data) + "\n")
            print(f"Finished writing {output_path}")
            return True
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            return False

    # Write train and test nodes to JSONL files
    train_file = os.path.join(train_dir, "metadata.jsonl")
    test_file = os.path.join(test_dir, "metadata.jsonl")
    
    train_success = write_nodes_to_jsonl(train_nodes, train_file)
    test_success = write_nodes_to_jsonl(test_nodes, test_file)

    print(f"Train file write success: {train_success}")
    print(f"Test file write success: {test_success}")
    
    # Verify file creation
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Both files created successfully")
    else:
        print(f"Train file exists: {os.path.exists(train_file)}")
        print(f"Test file exists: {os.path.exists(test_file)}")

    print("Script execution completed")