# -*- coding: utf-8 -*-
"""
Two-Tower Model Trainer for Modality Retrieval Task (Stage 2 Training).
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from omegaconf import OmegaConf

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.trainers.gnn_trainer import GNNTrainer
from src.multimodal_centric.qe.models.retrieval_model import TwoTowerModel

class RetrievalTrainer:
    """
    Responsible for Stage 2 training of the retrieval task: training the Two-Tower model.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = config.dataset.name
        self.gnn_model_name = config.model.name
        retrieval_params = config.task.retrieval_training
        self.epochs = retrieval_params.epochs
        self.lr = retrieval_params.lr
        self.batch_size = retrieval_params.batch_size
        self.patience = retrieval_params.patience
        self.tau = retrieval_params.tau
        
        self.text_encoder = self.config.text_encoder.replace('.', '_').replace('-', '_')
        self.visual_encoder = self.config.visual_encoder.replace('.', '_').replace('-', '_')

        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_save_dir = self.base_dir / "trained_models" / self.dataset_name / f"{self.gnn_model_name}_{self.text_encoder}_{self.visual_encoder}"
        self.model_save_path = self.model_save_dir / "model_retrieval.pt"
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.gnn_trainer = GNNTrainer(config)
        self.enhanced_text_embeds = None
        self.enhanced_image_embeds = None

    def _get_enhanced_embeddings(self) -> (torch.Tensor, torch.Tensor):
        """
        Get GNN-enhanced embeddings from Stage 1.
        If already acquired, return directly from memory.
        """
        if self.enhanced_text_embeds is not None and self.enhanced_image_embeds is not None:
            return self.enhanced_text_embeds, self.enhanced_image_embeds

        gnn_model = self.gnn_trainer.train_or_load_model()
        
        evaluator = self.gnn_trainer
        image_embeds = evaluator._storage.get_embedding(self.config.dataset.v_emb_path)
        text_embeds = evaluator._storage.get_embedding(self.config.dataset.t_emb_path)

        features = np.concatenate((text_embeds, image_embeds), axis=1)
        features = torch.from_numpy(features).float().to(self.device)
        
        graph = evaluator._storage.load_graph()
        edge_index = graph.edge_index.to(self.device)
        
        gnn_model.eval()
        with torch.no_grad():
            _, enhanced_img, enhanced_txt = gnn_model(features, edge_index)
        
        print("Successfully got GNN enhanced embeddings.")
        self.enhanced_text_embeds = enhanced_txt
        self.enhanced_image_embeds = enhanced_img
        return self.enhanced_text_embeds, self.enhanced_image_embeds

    def _calculate_infonce_loss(self, text_embeds, image_embeds):
        """
        Calculate symmetric InfoNCE loss.
        """
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / self.tau
        logits_per_image = logits_per_text.t()
        labels = torch.arange(len(text_embeds), device=self.device)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss_image = F.cross_entropy(logits_per_image, labels)
        return (loss_text + loss_image) / 2

    def train(self, retrieval_model):
        enhanced_text_embeds, enhanced_image_embeds = self._get_enhanced_embeddings()
        
        optimizer = optim.Adam(retrieval_model.parameters(), lr=self.lr)
        dataset = TensorDataset(enhanced_text_embeds, enhanced_image_embeds)
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="Retrieval Training"):
            retrieval_model.train()
            total_train_loss = 0
            for batch_text, batch_image in train_loader:
                optimizer.zero_grad()
                proj_text = retrieval_model.encode_text(batch_text)
                proj_image = retrieval_model.encode_image(batch_image)
                loss = self._calculate_infonce_loss(proj_text, proj_image)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            retrieval_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_text, batch_image in val_loader:
                    proj_text = retrieval_model.encode_text(batch_text)
                    proj_image = retrieval_model.encode_image(batch_image)
                    val_loss = self._calculate_infonce_loss(proj_text, proj_image)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            if epoch % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(retrieval_model.state_dict(), self.model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                tqdm.write(f"Validation loss has not improved for {self.patience} consecutive epochs. Early stopping.")
                break
        
        print("Training completed. Loading best performing model.")
        retrieval_model.load_state_dict(torch.load(self.model_save_path))
        return retrieval_model

    def train_or_load_model(self) -> TwoTowerModel:
        """
        Check if the model exists. If it exists, load it; otherwise, perform training.
        """
        enhanced_text_embeds, _ = self._get_enhanced_embeddings()
        
        # Get preset parameters from configuration and convert to a regular dictionary
        retrieval_model_params = OmegaConf.to_container(self.config.task.retrieval_model, resolve=True)
        # Dynamically add input_dim
        retrieval_model_params['input_dim'] = enhanced_text_embeds.shape[1]
        # Instantiate the model using the updated parameter dictionary
        retrieval_model = TwoTowerModel(**retrieval_model_params).to(self.device)

        if self.model_save_path.exists():
            print(f"Found pre-trained retrieval model at '{self.model_save_path}'. Loading...")
            retrieval_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        else:
            print(f"No pre-trained retrieval model found at '{self.model_save_path}'.")
            retrieval_model = self.train(retrieval_model)
        
        retrieval_model.eval()
        return retrieval_model