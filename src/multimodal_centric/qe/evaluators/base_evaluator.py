# -*- coding: utf-8 -*-
"""
Evaluator Base Class

This module defines BaseEvaluator, an abstract base class that provides
shared functionality for all concrete evaluators, including:
-   Unified initialization process handling configuration and GNN models.
-   A common `_get_enhanced_embeddings` method for retrieving graph-enhanced embeddings.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from src.utils.storage_manager import StorageManager


# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class BaseEvaluator(ABC):
    """
    Abstract base class for all QE evaluators.
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        Initialize the evaluator.

        Args:
            config (Dict[str, Any]): Dictionary containing task and dataset configuration.
            gnn_model (torch.nn.Module): A trained GNN model available for evaluation.
        """
        self.config = config
        self.gnn_model = gnn_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(self.device)
        
        self._storage = StorageManager(config)

    def _get_enhanced_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute neighborhood-enhanced embeddings for the entire graph using the GNN model.
        This is a generic method usable by all subclasses.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]:
                    A tuple containing (enhanced image embeddings for all nodes, enhanced text embeddings for all nodes).
                    Returns None if any required data cannot be loaded.
            """
        dataset_name = self.config.dataset.name

        image_embeddings = self._storage.get_embedding(self.config.dataset.v_emb_path)
        text_embeddings = self._storage.get_embedding(self.config.dataset.t_emb_path)

        if image_embeddings is None or text_embeddings is None:
            print("Error: Failed to load base embeddings.")
            return None

        try:
            graph_data = self._storage.load_graph()
            edge_index = graph_data.edge_index.to(self.device)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: Failed to load graph structure: {e}")
            return None

        multimodal_features = np.concatenate((text_embeddings, image_embeddings), axis=1)
        features_tensor = torch.from_numpy(multimodal_features).float().to(self.device)

        self.gnn_model.eval()
        with torch.no_grad():
            _, enhanced_image_embeddings, enhanced_text_embeddings = self.gnn_model(features_tensor, edge_index)

        return enhanced_image_embeddings.cpu().numpy(), enhanced_text_embeddings.cpu().numpy()

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Abstract method to execute evaluation.
        Each subclass must implement its own evaluation logic.
        """
        pass