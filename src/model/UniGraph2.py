import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import dgl
import numpy as np

# def compute_spd_matrix(graph: dgl.DGLGraph) -> torch.Tensor:
#    """Compute shortest path distance matrix for a graph"""
#    num_nodes = graph.num_nodes()
#    spd_matrix = torch.full((num_nodes, num_nodes), float('inf'))
#   
#    # Convert graph to adjacency matrix
#    adj_matrix = torch.zeros((num_nodes, num_nodes))
#    src, dst = graph.edges()
#    adj_matrix[src, dst] = 1
#    adj_matrix[dst, src] = 1  # Undirected graph
#    
#    # Initialize SPD matrix with direct connections
#    spd_matrix[adj_matrix == 1] = 1
#    # np.fill_diagonal(spd_matrix, 0)
#    spd_matrix.fill_diagonal_(0)
#    
#    # Floyd-Warshall algorithm
#    for k in range(num_nodes):
#        for i in range(num_nodes):
#            for j in range(num_nodes):
#                if spd_matrix[i, k] + spd_matrix[k, j] < spd_matrix[i, j]:
#                    spd_matrix[i, j] = spd_matrix[i, k] + spd_matrix[k, j]
#                    
#    return spd_matrix

def compute_spd_matrix_optimized(graph: dgl.DGLGraph, k: int) -> torch.Tensor:
    """
    Args:
        graph: DGL graph object
        k: Maximum hop distance to compute (default: 10)
        
    Returns:
        dist_matrix: (num_nodes, num_nodes) tensor with shortest path distances
                    Distance > k will be set to inf
    """
    num_nodes = graph.num_nodes()
    device = graph.device
    src, dst = graph.edges()
    
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    dist_matrix = torch.full((num_nodes, num_nodes), float('inf'), device=device, dtype=torch.float32)
    dist_matrix.fill_diagonal_(0)
    filled_mask = torch.eye(num_nodes, device=device, dtype=torch.bool)
    
    curr_adj = adj
    
    for hop in range(1, k + 1):
        # Only update positions not yet filled (ensures shortest distance)
        reachable = (curr_adj > 0) & (~filled_mask)
        src_indices, dst_indices = reachable.nonzero(as_tuple=True)
        
        if src_indices.numel() > 0:
            dist_matrix[src_indices, dst_indices] = float(hop)
            filled_mask[src_indices, dst_indices] = True
        
        if hop < k:
            next_adj = torch.mm(curr_adj, adj)
            # Truncate values greater than 1 to 1 (since we only care about reachability, not path count)
            next_adj = (next_adj > 0).float()
            if torch.equal(curr_adj, next_adj):
                break # At this point A^(k+1) = A^k, calculation is done
            curr_adj = next_adj
    
    return dist_matrix

class MoE(nn.Module):
    """Mixture of Experts module for cross-domain and cross-modality alignment"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get expert weights
        weights = self.gate(x)
        
        # Select top-k experts
        top_weights, top_indices = torch.topk(weights, self.num_selected_experts, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine expert outputs
        selected_outputs = torch.gather(
            expert_outputs,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        )
        output = torch.sum(selected_outputs * top_weights.unsqueeze(-1), dim=1)
        
        return output


class DomainSpecificDecoder(nn.Module):
    """Decoder for specific graph domains"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class SPDDecoder(nn.Module):
    """Shortest Path Distance decoder"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor = None) -> torch.Tensor:
        if x_j is None:
            x = x_i
        else:
            x = torch.cat([x_i, x_j], dim=-1)
        return self.decoder(x)


class UniGraph2Core(nn.Module):
    """UniGraph2 model for multimodal graph representation learning"""
    
    def __init__(
        self,
        input_dims: Dict[str, int],  # Dictionary of input dimensions for each modality
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int,
        num_layers: int,
        feat_drop_rate: float,
        edge_mask_rate: float,
        gamma: float,
        lambda_spd: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_drop_rate = feat_drop_rate
        self.edge_mask_rate = edge_mask_rate
        self.gamma = gamma
        self.lambda_spd = lambda_spd
        
        # Fix UniGraph2 bug: input feature dimension might not equal hidden_dim
        # Will raise error when entering MoE layer
        self.projector = nn.ModuleDict({
          domain: nn.Linear(dim, hidden_dim)
          for domain, dim in input_dims.items()
        })
        
        # Mixture of Experts
        self.moe = MoE(hidden_dim, hidden_dim, num_experts, num_selected_experts)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            dgl.nn.GATConv(hidden_dim, hidden_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        
        # Domain-specific decoders
        self.domain_decoders = nn.ModuleDict({
            domain: DomainSpecificDecoder(hidden_dim, input_dims[domain])
            for domain in input_dims.keys()
        })
        
        # SPD decoder
        self.spd_decoder = SPDDecoder(hidden_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        
    def _mask_features(
        self,
        features: torch.Tensor,
        mask_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask node features"""
        num_nodes = features.size(0)
        num_masked = int(num_nodes * mask_rate)
        
        # Create mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[:num_masked] = True
        mask = mask[torch.randperm(num_nodes)]
        
        # Apply mask
        masked_features = features.clone()
        # masked_features[mask] = self.mask_token
        masked_features[mask] = self.mask_token.to(masked_features.dtype)
        
        return masked_features, mask
        
    def _compute_spd_loss(
        self,
        embeddings: torch.Tensor,
        spd_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute shortest path distance loss"""
        num_nodes = embeddings.size(0)
        spd_pred = torch.zeros_like(spd_matrix)
        
        # Compute predicted SPD for all node pairs
        # ! Severe time and memory usage issues here
        # ! Testing num_nodes approx 10000, not only slow, but computational graph saves 10^9 concatenated vectors, OOM halfway.
        for i in range(num_nodes):
            for j in range(num_nodes):
                spd_pred[i, j] = self.spd_decoder(embeddings[i], embeddings[j])
                
        return F.mse_loss(spd_pred, spd_matrix)
        
    def _compute_spd_loss_optimized(
        self,
        embeddings: torch.Tensor,
        spd_matrix: torch.Tensor,
        sample_size: int = 500
    ) -> torch.Tensor:
        """
        Compute shortest path distance loss
        
        Time optimization: O(n^2) -> O(1)
        Memory optimization: Depends on sample_size setting, space complexity O(n^2)
        """
        num_nodes = embeddings.size(0)
        sample_indices = torch.randperm(num_nodes)[:min(sample_size, num_nodes)].sort().values
        sample_embeddings = embeddings[sample_indices]
        sample_spd_matrix = spd_matrix[sample_indices][:, sample_indices]
        
        valid_mask = torch.isfinite(sample_spd_matrix)
        rows, cols = valid_mask.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        emb_rows = sample_embeddings[rows]
        emb_cols = sample_embeddings[cols]
        concat_emb = torch.cat([emb_rows, emb_cols], dim=-1)
        spd_pred = self.spd_decoder(concat_emb).squeeze(-1)
        spd_true = sample_spd_matrix[rows, cols]
        return F.mse_loss(spd_pred, spd_true)
        
        ##########################################
        
        # num_nodes = embeddings.size(0)
        # embed_dim = embeddings.size(1)
        # spd_pred = torch.zeros_like(spd_matrix)
        
        # # Batch processing rows
        # for start_idx in range(0, num_nodes, batch_size):
        #     end_idx = min(start_idx + batch_size, num_nodes)
        #     batch_size_actual = end_idx - start_idx
            
        #     # Only expand current batch rows
        #     batch_emb = embeddings[start_idx:end_idx]  # (batch_size, embed_dim)
        #     expanded_batch = batch_emb.unsqueeze(1).expand(-1, num_nodes, -1)  # (batch_size, num_nodes, embed_dim)
        #     repeated_all = embeddings.unsqueeze(0).expand(batch_size_actual, -1, -1)  # (batch_size, num_nodes, embed_dim)
            
        #     # Concatenate
        #     concat_batch = torch.cat([expanded_batch, repeated_all], dim=-1)  # (batch_size, num_nodes, 2*embed_dim)
            
        #     # Flatten and batch decode
        #     concat_batch_flat = concat_batch.view(-1, 2 * embed_dim)
        #     pred_batch_flat = self.spd_decoder(concat_batch_flat).squeeze(-1)
            
        #     # Reshape and assign
        #     pred_batch = pred_batch_flat.view(batch_size_actual, num_nodes)
        #     spd_pred[start_idx:end_idx] = pred_batch
        
        # return F.mse_loss(spd_pred, spd_matrix)
        
        ##########################################
        
        # valid_mask = torch.isfinite(spd_matrix)
        # if valid_mask.sum() == 0:
        #     return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # rows, cols = valid_mask.nonzero(as_tuple=True)
        # num_pairs = rows.numel()
        # # Need batched input to decoder to avoid OOM
        # num_batches = (num_pairs + batch_size - 1) // batch_size
        
        # total_loss_sum = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        # total_samples = 0
        
        # for batch_idx in range(num_batches):
        #     start_idx = batch_idx * batch_size
        #     end_idx = min(start_idx + batch_size, num_pairs)
            
        #     batch_rows = rows[start_idx:end_idx]
        #     batch_cols = cols[start_idx:end_idx]
            
        #     emb_rows = embeddings[batch_rows]
        #     emb_cols = embeddings[batch_cols]
        #     concat_emb = torch.cat([emb_rows, emb_cols], dim=-1)
            
        #     spd_pred_batch = self.spd_decoder(concat_emb).squeeze(-1)
        #     spd_true_batch = spd_matrix[batch_rows, batch_cols]
            
        #     batch_loss_sum = F.mse_loss(spd_pred_batch, spd_true_batch, reduction='sum')
        #     total_loss_sum = total_loss_sum + batch_loss_sum
        #     total_samples += batch_rows.numel()
        
        # return total_loss_sum / total_samples
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        features: Dict[str, torch.Tensor],
        spd_matrix: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Fix UniGraph2 bug: input feature dimension might not equal hidden_dim
        # Will raise error when entering MoE layer
        projected_features = {
          domain: self.projector[domain](feat)
          for domain, feat in features.items()
        }
        
        # Average features across modalities
        x = torch.stack(list(projected_features.values())).mean(dim=0)
        
        # Mask features
        masked_x, mask = self._mask_features(x, self.feat_drop_rate)
        
        # Apply MoE
        aligned_x = self.moe(masked_x)
        
        # Apply GNN layers
        h = aligned_x
        for layer in self.gnn_layers:
            h = layer(graph, h).mean(dim=1)
            
        if return_embeddings:
            return h
            
        # Reconstruct features for each domain
        reconstruction_loss = 0
        for domain, decoder in self.domain_decoders.items():
            reconstructed = decoder(h[mask])
            original = features[domain][mask]
            similarity = F.cosine_similarity(reconstructed, original, dim=-1)
            reconstruction_loss += (1 - similarity).pow(self.gamma).mean()
            
        # Compute SPD loss if provided
        spd_loss = 0
        if spd_matrix is not None:
            # # Original algorithm had severe time/memory issues, so optimized.
            # spd_loss = self._compute_spd_loss(h, spd_matrix)
            spd_loss = self._compute_spd_loss_optimized(h, spd_matrix)
            
        # Combine losses
        total_loss = reconstruction_loss + self.lambda_spd * spd_loss
        
        return total_loss, h 

############################################################
# Above mainly from UniGraph2's models/unigraph2.py and train.py
# And optimized for time complexity and memory usage.
# The following wrapper class is for OpenMAG adaptation.
############################################################

class UniGraph2(nn.Module):
    """UniGraph2 Wrapper for OpenMAG"""
    def __init__(
        self,
        v_feat_dim: int,
        t_feat_dim: int,
        hidden_dim: int = 768,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        num_layers: int = 3,
        feat_drop_rate: float = 0.1,
        edge_mask_rate: float = 0.1,
        gamma: float = 2.0,
        lambda_spd: float = 0.5,
        dropout: float = 0.2
    ):
        """
        Args:
            v_feat_dim (int): Visual feature dimension, corresponds to image feature dimension.
            t_feat_dim (int): Text feature dimension, corresponds to text feature dimension.
            hidden_dim (int, optional): Hidden layer dimension, default 768. Model uses this dimension internally.
            num_experts (int, optional): Number of expert networks in MoE, default 8.
            num_selected_experts (int, optional): Number of experts activated per sample, default 2.
            num_layers (int, optional): Number of GNN layers, default 3.
            feat_drop_rate (float, optional): Feature mask rate, for self-supervised learning, default 0.1.
            edge_mask_rate (float, optional): Edge mask rate, currently unused, default 0.1.
            gamma (float, optional): Focal Loss parameter for reconstruction loss, default 2.0.
                                    Larger gamma increases weight for hard samples.
            lambda_spd (float, optional): Weight for SPD loss, default 0.5.
                                        Used to balance reconstruction loss and SPD loss.
            dropout (float, optional): Dropout rate for vision_head and text_head, default 0.2.
        
        Note:
            - v_feat_dim and t_feat_dim can differ, aligned to hidden_dim via projection layer internally.
            - feat_drop_rate controls mask ratio in self-supervised learning, affecting learning difficulty.
            - lambda_spd controls weight of structural info learning, adjustable experimentally.
        """
        super(UniGraph2, self).__init__()
        self.v_feat_dim = v_feat_dim
        self.t_feat_dim = t_feat_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.num_layers = num_layers
        self.feat_drop_rate = feat_drop_rate
        self.edge_mask_rate = edge_mask_rate
        self.gamma = gamma
        self.lambda_spd = lambda_spd
        self.dropout = dropout
        self.can_return_loss: bool = True
        self.input_dims = {
            "text": t_feat_dim,
            "image": v_feat_dim
        }
        
        self.core = UniGraph2Core(self.input_dims, self.hidden_dim, self.num_experts, self.num_selected_experts, self.num_layers, self.feat_drop_rate, self.edge_mask_rate, self.gamma, self.lambda_spd)
        
        # # UniGraph2 only returns node-level features. Added modality heads following models.py approach.
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
        
    def reset_parameters(self):
        # Recursively reset all submodules using PyTorch default initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                if module is self.core.mask_token:
                    nn.init.normal_(module, mean=0.0, std=0.02)
        
        # DGL's GATConv has reset_parameters method.
        for layer in self.core.gnn_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        k: int = 10,
        use_subgraph: bool = False
    ):
        """
        Forward propagation
        
        Args:
            x (torch.Tensor): Node features, shape (num_nodes, v_feat_dim + t_feat_dim)
                           Features concatenated in order: [visual features, text features]
            edge_index (torch.Tensor): Edge index, PyTorch Geometric format, shape (2, num_edges)
            k (int, optional): Max hops for k-hop distance calculation, default 10.
                             Controls range of SPD loss calculation; larger values include more structural info but compute slower.
        
        Returns:
            **Training mode (training=True):**
            - h (torch.Tensor): Fused node embedding, shape (num_nodes, hidden_dim)
            - x_vision (torch.Tensor): Visual specific representation, shape (num_nodes, hidden_dim)
            - x_text (torch.Tensor): Text specific representation, shape (num_nodes, hidden_dim)
            - total_loss (torch.Tensor): Model internal loss, scalar
              Contains two parts:
              - reconstruction_loss: Feature reconstruction loss (Focal Loss variant)
              - spd_loss: Shortest Path Distance loss (weighted by lambda_spd)
            
            **Evaluation mode (training=False):**
            - h (torch.Tensor): Fused node embedding
            - x_vision (torch.Tensor): Visual specific representation
            - x_text (torch.Tensor): Text specific representation
        
        Note:
            - Returns 4 values during training, 3 during evaluation.
            - total_loss needs to be added to total loss via config.task.lambda_model weight.
        """
        src, dst = edge_index
        graph = dgl.graph((src, dst)).to(x.device)
        graph = dgl.add_self_loop(graph)
        
        v_feat, t_feat = x[:, :self.v_feat_dim], x[:, self.v_feat_dim:]
        features = {
            "text": t_feat.to(x.device),
            "image": v_feat.to(x.device)
        }
        
        if self.training:
            # # Use optimized algorithm to compute shortest path distance matrix
            # spd_matrix = compute_spd_matrix(graph).to(x.device)
            spd_matrix = compute_spd_matrix_optimized(graph, k).to(x.device)
            total_loss, h = self.core(graph, features, spd_matrix)
        else:
            h = self.core(graph, features, return_embeddings=True)
        
        x_vision = F.dropout(F.relu(self.vision_head(h)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(h)), p=self.dropout, training=self.training)
        
        # ! Note UniGraph2 generates two losses itself.
        # ! One is modality reconstruction loss, one is SPD loss.
        # ! Needs to be attached to external loss.
        if self.training:
            return h, x_vision, x_text, total_loss
        else:
            return h, x_vision, x_text