# -*- coding: utf-8 -*-
"""
GNN Model Trainer

This module is responsible for:
1.  Initializing a GNN model based on the configuration.
2.  Checking if pre-trained weights exist for a specific dataset and GNN model.
3.  Loading weights if they exist; otherwise, executing the training loop.
4.  Training using the InfoNCE contrastive learning objective.
5.  Implementing Early Stopping on the validation set to prevent overfitting.
6.  Saving the trained model weights.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import Optional
import dgl
from torch_geometric.data import Data

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.models import GCN, GraphSAGE, GAT, MLP, GIN, ChebNet, LGMRec, GCNII, GATv2, MHGAT
from src.model.MMGCN import Net as MMGCN
from src.model.MGAT import MGAT as MGAT_model
from src.model.REVGAT import RevGAT
from src.model.GSMN import GSMN
from src.model.DMGC import DMGC
from src.model.DGF import DGF
from src.utils.storage_manager import StorageManager

def calculate_clip_score(image_embedding: np.ndarray, text_embedding: np.ndarray) -> Optional[float]:
    """
    Calculates and returns the CLIP-score between a pair of image and text embeddings.
    Returns None if either embedding is a zero vector.
    """
    img_norm = np.linalg.norm(image_embedding)
    txt_norm = np.linalg.norm(text_embedding)

    if img_norm == 0 or txt_norm == 0:
        return None

    image_embedding = image_embedding / img_norm
    text_embedding = text_embedding / txt_norm
    return 100.0 * np.dot(image_embedding, text_embedding)

class GNNTrainer:
    """
    Responsible for training, saving, and loading GNN models for QE tasks.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = self.config.dataset.name
        self.gnn_model_name = self.config.model.name
        
        self.text_encoder = self.config.text_encoder.replace('.', '_').replace('-', '_')
        self.visual_encoder = self.config.visual_encoder.replace('.', '_').replace('-', '_')
        
        train_params = self.config.training
        self.epochs = train_params.epochs
        self.lr = train_params.lr
        self.patience = train_params.patience
        self.val_ratio = train_params.val_ratio
        self.tau = train_params.tau

        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_save_dir = self.base_dir / "trained_models" / self.dataset_name / f"{self.gnn_model_name}_{self.text_encoder}_{self.visual_encoder}"
        self.model_save_path = self.model_save_dir / "model.pt"
        os.makedirs(self.model_save_dir, exist_ok=True)

        self._storage = StorageManager(config)
        self.model = self._init_model().to(self.device) 

    def _init_model(self) -> torch.nn.Module:
        model_params = self.config.model.params
        sample_text_embed = self._storage.get_embedding(self.config.dataset.t_emb_path)
        if sample_text_embed is None: raise ValueError("Cannot load embedding to determine model input dimension.")
        in_dim = sample_text_embed.shape[1] * 2
        
        model_params_with_in_dim = {'in_dim': in_dim, **model_params}

        if self.gnn_model_name == 'GCN': model = GCN(**model_params_with_in_dim)
        elif self.gnn_model_name == 'GAT': model = GAT(**model_params_with_in_dim)
        elif self.gnn_model_name == 'GraphSAGE': model = GraphSAGE(**model_params_with_in_dim)
        elif self.gnn_model_name == 'MLP': model = MLP(**model_params_with_in_dim)
        elif self.gnn_model_name == "GIN": model = GIN(**model_params_with_in_dim)
        elif self.gnn_model_name == "LGMRec": model = LGMRec(**model_params_with_in_dim)
        elif self.gnn_model_name == "GCNII": model = GCNII(**model_params_with_in_dim)
        elif self.gnn_model_name == "GATv2": model = GATv2(**model_params_with_in_dim)
        elif self.gnn_model_name == "ChebNet":
            n_layers = model_params['num_layers']  
            n_hidden = model_params['hidden_dim']
            K = model_params['K']
            dropout = model_params['dropout']
            model = ChebNet(in_dim=in_dim, hidden_dim=n_hidden, num_layers=n_layers, K=K, dropout=dropout)
        elif self.gnn_model_name == "GravNet":
            n_layers = model_params['num_layers']  
            n_hidden = model_params['hidden_dim']
            k = model_params['k']
            space_dimensions = model_params['space_dimensions']
            propagate_dimensions = model_params['propagate_dimensions']
            dropout = model_params['dropout']
            model = GravNet(in_dim=in_dim, hidden_dim=n_hidden, num_layers=n_layers, k=k, space_dimensions=space_dimensions, propagate_dimensions=propagate_dimensions, dropout=dropout)
        elif self.gnn_model_name == "RevGAT":
            n_layers = model_params['num_layers']  
            n_hidden = model_params['hidden_dim']
            n_heads = model_params['heads']
            dropout = model_params['dropout']
            model = RevGAT(in_feats=in_dim, n_hidden=n_hidden, n_layers=n_layers, n_heads=n_heads, activation=F.relu, dropout = dropout)
      
        elif self.gnn_model_name == "MMGCN":
            v_dim = sample_text_embed.shape[1]
            num_nodes = self._storage.load_graph().num_nodes
            num_layers = model_params['num_layers']  
            dim_x = model_params['hidden_dim']     
            model = MMGCN(v_feat_dim=768, t_feat_dim=in_dim-768, num_nodes=num_nodes, v_dim=v_dim, aggr_mode='mean', concate=False, num_layers=num_layers, has_id=True, dim_x=dim_x)
        elif self.gnn_model_name == "MGAT":
            v_dim = sample_text_embed.shape[1]
            num_nodes = self._storage.load_graph().num_nodes
            num_layers = model_params['num_layers']  
            dim_x = model_params['hidden_dim']     
            model = MGAT_model(v_feat_dim=768, t_feat_dim=in_dim-768, num_nodes=num_nodes, num_layers=num_layers, dim_x=dim_x, v_dim=v_dim)
        elif self.gnn_model_name == "GSMN":
            v_dim = sample_text_embed.shape[1]
            num_nodes = self._storage.load_graph().num_nodes
            num_layers = model_params['num_layers']
            dim_x = model_params['hidden_dim']
            out_dim = model_params['out_dim']
            dropout = model_params['dropout']
            model = GSMN(
                v_feat_dim=768,
                t_feat_dim=in_dim - 768,
                num_nodes=num_nodes,
                num_layers=num_layers,
                hidden_dim=dim_x,
                out_dim=out_dim,
                dropout=dropout,
            )
        elif self.gnn_model_name == "MHGAT":
            v_dim = sample_text_embed.shape[1]
            t_dim = in_dim - v_dim
            model = MHGAT(
                v_dim=v_dim,
                t_dim=t_dim,
                hidden_dim=model_params['hidden_dim'],
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout'],
                heads=model_params['heads']
            )
        elif self.gnn_model_name == "DMGC":
            v_dim = sample_text_embed.shape[1]
            t_dim = in_dim - v_dim
            model = DMGC(
                v_feat_dim=v_dim,     # Assume image feature dimension
                t_feat_dim=t_dim,     # Assume text feature dimension
                hidden_dim=model_params['hidden_dim'],
                num_layers=model_params.get('num_layers', 1),
                tau=self.tau,            # Use tau from Trainer config
                lambda_cr=model_params.get('lambda_cr', 0.001),
                lambda_cm=model_params.get('lambda_cm', 1.0),
                dropout=model_params.get('dropout', 0.5),
                sparse=model_params.get('sparse', False),
                graph_update_freq=model_params.get('graph_update_freq', -1)
            )
        elif self.gnn_model_name == "DGF":
            v_dim = sample_text_embed.shape[1]
            t_dim = in_dim - v_dim
            model = DGF(
                v_feat_dim=v_dim,
                t_feat_dim=t_dim,
                hidden_dim=model_params['hidden_dim'],
                alpha=model_params['alpha'],
                beta=model_params['beta'],
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout'],
            )
        else:
            raise ValueError(f"Unknown GNN model: {self.gnn_model_name}")
        return model

    def _calculate_infonce_loss(self, query, positive_key, all_keys) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        all_keys = F.normalize(all_keys, p=2, dim=1)
        l_pos = (query * positive_key).sum(dim=-1)
        logits = torch.matmul(query, all_keys.T)
        logits /= self.tau
        labels = torch.arange(len(query), device=self.device)
        return F.cross_entropy(logits, labels)

    def _get_data_splits(self, edge_index):
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)
        val_size = int(num_edges * self.val_ratio)
        val_edges = edge_index[:, perm[:val_size]]
        train_edges = edge_index[:, perm[val_size:]]
        return train_edges, val_edges

    def train(self):
        print(f"Start training GNN model '{self.gnn_model_name}' for dataset '{self.dataset_name}'...")
        
        graph = self._storage.load_graph()
        train_edge_index, val_edge_index = self._get_data_splits(graph.edge_index)
        train_edge_index = train_edge_index.to(self.device)
        val_edge_index = val_edge_index.to(self.device)

        image_embeds = self._storage.get_embedding(self.config.dataset.v_emb_path)
        text_embeds = self._storage.get_embedding(self.config.dataset.t_emb_path)
        scores = [calculate_clip_score(img, txt) for img, txt in zip(image_embeds, text_embeds)]
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        print(f"Baseline CLIP-Score (no GNN enhancement): {np.mean(valid_scores):.4f}")
        if num_invalid > 0:
            print(f"  (Warning: Ignored {num_invalid} zero-vector samples when calculating baseline)")

        features = torch.from_numpy(np.concatenate((text_embeds, image_embeds), axis=1)).float().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="GNN Training"):
            self.model.train()
            optimizer.zero_grad()
            if self.gnn_model_name in ["DMGC", "DGF"]:
                _, enhanced_img, enhanced_txt, internal_loss = self.model(features, train_edge_index)
                task_loss = self._calculate_infonce_loss(enhanced_img, enhanced_txt, enhanced_txt)
                loss = task_loss + internal_loss
            else:
                _, enhanced_img, enhanced_txt = self.model(features, train_edge_index)
                loss = self._calculate_infonce_loss(enhanced_img, enhanced_txt, enhanced_txt)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                _, val_img, val_txt = self.model(features, val_edge_index)
                val_loss = self._calculate_infonce_loss(val_img, val_txt, val_txt)

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                tqdm.write(f"Validation loss has not improved for {self.patience} consecutive epochs. Early stopping.")
                break
        
        print("Training completed. Loading best performing model.")
        self.model.load_state_dict(torch.load(self.model_save_path))

    def train_or_load_model(self) -> torch.nn.Module:
        if self.model_save_path.exists():
            print(f"Found pre-trained model at '{self.model_save_path}'. Loading...")
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        else:
            print(f"No pre-trained model found at '{self.model_save_path}'.")
            self.train()
        
        self.model.eval()
        return self.model