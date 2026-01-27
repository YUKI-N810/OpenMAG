# -*- coding: utf-8 -*-
"""
Two-Tower Model for Modality Retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    A simple Two-Tower model for modality retrieval.
    Each tower is a Multi-Layer Perceptron (MLP) that projects input embeddings into a shared semantic space.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        Initialize the Two-Tower model.

        Args:
            input_dim (int): Dimension of input embeddings (enhanced embeddings from GNN).
            hidden_dim (int): Dimension of MLP hidden layers.
            output_dim (int): Dimension of output embeddings (final retrieval space dimension).
            num_layers (int): Number of MLP layers for each tower.
            dropout (float): Dropout ratio.
        """
        super().__init__()
        self.text_tower = self._create_tower(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.image_tower = self._create_tower(input_dim, hidden_dim, output_dim, num_layers, dropout)

    def _create_tower(self, input_dim, hidden_dim, output_dim, num_layers, dropout) -> nn.Sequential:
        """
        Create an MLP tower.
        """
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass.

        Args:
            text_embeds (torch.Tensor): Input embeddings for the text tower.
            image_embeds (torch.Tensor): Input embeddings for the image tower.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Projected text and image embeddings.
        """
        # 1. Pass through MLP
        h_text = self.text_tower(text_embeds)
        h_image = self.image_tower(image_embeds)
        
        # 2. [Critical Change] L2 normalization to ensure outputs are on the unit hypersphere
        h_text = F.normalize(h_text, p=2, dim=1)
        h_image = F.normalize(h_image, p=2, dim=1)
        
        return h_text, h_image

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Encode text embeddings only."""
        return self.text_tower(text_embeds)

    def encode_image(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Encode image embeddings only."""
        return self.image_tower(image_embeds)