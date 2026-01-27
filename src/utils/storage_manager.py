import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union, Dict
import logging
import json
import tarfile
import dgl
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class StorageManager:
    """Storage Manager, responsible for managing storage paths and naming conventions for dataset feature files."""
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.dataset.data_root
        self.dataset_name = config.dataset.name
        self.dataset_path = os.path.join(self.data_root, self.dataset_name)
        self.graph_path = config.dataset.graph_path
        
        self._node_id_cache: Dict[str, List[str]] = {}
        self._raw_text_cache: Dict[str, Dict[str, str]] = {}
        self._image_path_cache: Dict[str, Dict[str, Path]] = {}
        
    def load_graph(self) -> Data:
        path = self.graph_path
        # Directly load unified format graph files
        g_data = dgl.load_graphs(path)[0][0]
        
        # Convert to PyG format
        src, dst = g_data.edges()
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = g_data.num_nodes()
    
        return Data(x=torch.zeros((num_nodes, 1)), edge_index=edge_index)
    
    def get_embedding(
        self,
        feature_path: str
    ) -> Optional[np.ndarray]:
        

        if not Path(feature_path).exists():
            print(f"Warning: Embedding file not found: {feature_path}")
            return None
        
        embedding = np.load(feature_path)

        # ======= Embedding Value Analysis =======
        nan_count = np.isnan(embedding).sum()
        pos_inf_count = np.isposinf(embedding).sum()
        neg_inf_count = np.isneginf(embedding).sum()

        # Calculate min/max values after removing inf and nan
        finite_mask = np.isfinite(embedding)
        if np.any(finite_mask):
            min_val = embedding[finite_mask].min()
            max_val = embedding[finite_mask].max()
        else:
            min_val = max_val = None

        return embedding
        #return self._storage.load_features(feature_path)
    
    def load_node_ids(self, dataset_name: str) -> List[str]:
        """Directly load node_ids.json file."""
        # 1. Check cache
        if dataset_name in self._node_id_cache:
            return self._node_id_cache[dataset_name]

        # 2. If not in cache, load it
        node_ids_path = os.path.join(self.dataset_path, "node_ids.json")
        if not os.path.exists(node_ids_path):
            raise FileNotFoundError(f"Node ID file not found: {node_ids_path}")
            
        with open(node_ids_path, 'r') as f:
            data = json.load(f)
            
        # 3. Store in cache
        self._node_id_cache[dataset_name] = data
        return data
            
    def load_raw_text_map(self, dataset_name: str) -> Dict[str, str]:
        """Load or retrieve raw text from cache."""
        # 1. Check cache
        if dataset_name in self._raw_text_cache:
            return self._raw_text_cache[dataset_name]

        # 2. Load file (this file is usually large, slow without cache)
        jsonl_path = os.path.join(self.dataset_path, f"{dataset_name}-raw-text.jsonl")
        if not os.path.exists(jsonl_path): 
            raise FileNotFoundError(f"Raw text file not found: {jsonl_path}")
            
        text_map = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                node_id = str(record.get("id") or record.get("asin"))
                
                raw_text = record.get("raw_text", "")
                text_content = " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
                text_map[node_id] = text_content.strip()
        
        # 3. Store in cache
        self._raw_text_cache[dataset_name] = text_map
        return text_map
    
    def load_image_path_map(self, dataset_name: str) -> Dict[str, Path]:
        """Load or retrieve image path map from cache."""
        # 1. Check cache
        if dataset_name in self._image_path_cache:
            return self._image_path_cache[dataset_name]

        image_dir = Path(self.config.dataset.image_path)
        if not image_dir.exists():
            print(f"Warning: Configured image path does not exist: {image_dir}")
            return {}

        print(f"Building image path cache (Scanning: {image_dir})...") 
        image_path_map = {}
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                image_path_map[image_path.stem] = image_path
        
        # 3. Store in cache
        self._image_path_cache[dataset_name] = image_path_map
        return image_path_map
        
    def get_raw_data_by_index(self, dataset_name: str, node_index: int) -> Optional[Dict[str, str]]:
        try:
            # These calls will now return directly from memory (self._cache), very fast
            node_ids = self.load_node_ids(dataset_name)
            
            if node_index >= len(node_ids):
                return None
            
            target_id = node_ids[node_index]

            text_map = self.load_raw_text_map(dataset_name)
            text = text_map.get(target_id, "")

            image_map = self.load_image_path_map(dataset_name)
            image_path = image_map.get(target_id, "")
            
            if not text and not image_path:
                return None

            return {"text": text, "image_path": str(image_path)}

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None