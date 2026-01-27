"""
DGF (Dual Graph Filtering) Model for OpenMAG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_cluster
random_walk = torch.ops.torch_cluster.random_walk


############################################################
# Loss Functions
############################################################

class MMS_loss(nn.Module):
    """Multimodal similarity loss (with margin)"""
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0), device=S.device)
        S = S - deltas
        target = torch.arange(S.size(0), device=S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        return I2C_loss + C2I_loss


def compute_cross_modal_loss(z1, z2, h, margin=0.001):
    """Compute cross-modal contrastive loss between three modalities"""
    loss_fn = MMS_loss()
    sim_12 = torch.matmul(z1, z2.t())
    sim_1h = torch.matmul(z1, h.t())
    sim_2h = torch.matmul(z2, h.t())
    return loss_fn(sim_12, margin) + loss_fn(sim_1h, margin) + loss_fn(sim_2h, margin)


def cluster_loss(H, centroids, labels, theta=0.3):
    """
    Community contrastive loss (hard positive sampling)
    
    Args:
        H: Normalized features [N, d]
        centroids: K-Means cluster centroids [K, d]
        labels: Cluster assignment labels [N]
        theta: Percentage of top samples in each cluster
    """
    n_samples = H.shape[0]
    K = centroids.shape[0]
    device = H.device

    if n_samples == 0 or K == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = 0.0
    for k in range(K):
        mask = (labels == k)
        num_in_cluster = mask.sum().item()
        if num_in_cluster == 0:
            continue

        H_k = H[mask]
        centroid_k = centroids[k]
        
        # Top-k positive samples
        sims = torch.matmul(H_k, centroid_k)
        k_topk = max(1, int(math.ceil(num_in_cluster * theta)))
        k_topk = min(k_topk, num_in_cluster)
        
        _, top_idx = torch.topk(sims, k=k_topk)
        sum_exp_pos = torch.sum(torch.exp(sims[top_idx]))
        
        if sum_exp_pos <= 0:
            continue

        # Denominator: similarity between all samples and all cluster centroids
        sim_all = torch.matmul(H_k, centroids.T)
        denom = torch.sum(torch.exp(sim_all))
        
        if denom <= 0:
            continue
            
        total_loss += -torch.log(sum_exp_pos / denom)

    return total_loss / n_samples if n_samples > 0 else torch.tensor(0.0, device=device)


############################################################
# Graph Contrastive Learning Utility Functions
############################################################

def filter_adj_by_similarity(adj, z1, z2, threshold):
    """Filter edges based on cross-modal similarity (Attribute-Aware Sampling AAS)"""
    edge_indices = adj.coalesce().indices()
    rows, cols = edge_indices[0], edge_indices[1]
    
    z1_norm = F.normalize(z1, dim=1, p=2)
    z2_norm = F.normalize(z2, dim=1, p=2)
    
    sims = (z1_norm[rows] * z2_norm[cols]).sum(dim=1)
    mask = sims >= threshold
    
    new_indices = edge_indices[:, mask]
    new_values = torch.ones(new_indices.size(1), device=adj.device)
    
    return torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_values,
        size=adj.size(),
        device=adj.device
    ).coalesce()


def determine_threshold(adj, z1, z2, num_std=1.0):
    """Determine similarity threshold based on random sampling"""
    n = adj.size(0)
    m = adj._nnz()
    
    # Sample random node pairs
    rows = torch.randint(0, n, (m,), device=z1.device)
    cols = torch.randint(0, n, (m,), device=z1.device)
    
    z1_norm = F.normalize(z1, dim=1, p=2)
    z2_norm = F.normalize(z2, dim=1, p=2)
    sims = (z1_norm[rows] * z2_norm[cols]).sum(dim=1)
    
    return (sims.mean() + num_std * sims.std()).item()


def pos_sample_rw(adj_csr, n_nodes, walks_per_node, walk_length, context_size, device):
    """Generate positive samples using random walks"""
    nodes = torch.arange(n_nodes, device=device).repeat(walks_per_node)
    adj_csr_cpu = adj_csr.to_sparse_csr().cpu()
    rowptr = adj_csr_cpu.crow_indices().to(device)
    col = adj_csr_cpu.col_indices().to(device)
    
    rw = random_walk(rowptr, col, nodes, walk_length, 1.0, 1.0)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    
    walks = []
    num_walks = 1 + walk_length + 1 - context_size
    for j in range(num_walks):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def neg_sample_rw(n_nodes, walks_per_node, walk_length, context_size, device):
    """Generate negative samples via random node sequences"""
    nodes = torch.arange(n_nodes, device=device).repeat(walks_per_node)
    rw = torch.randint(n_nodes, (nodes.size(0), walk_length), device=device)
    rw = torch.cat([nodes.view(-1, 1), rw], dim=-1)
    
    walks = []
    num_walks = 1 + walk_length + 1 - context_size
    for j in range(num_walks):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)


def graph_contrastive_loss(pos_rw, neg_rw, embedding, hidden_dim, device):
    """Compute graph contrastive loss using random walks"""
    n_nodes = embedding.size(0)
    
    # Filter valid indices
    pos_valid = (pos_rw < n_nodes).all(dim=1)
    neg_valid = (neg_rw < n_nodes).all(dim=1)
    pos_rw = pos_rw[pos_valid]
    neg_rw = neg_rw[neg_valid]
    
    if pos_rw.size(0) == 0 or neg_rw.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Limit sample size
    max_samples = 4096
    if pos_rw.size(0) > max_samples:
        pos_rw = pos_rw[torch.randperm(pos_rw.size(0), device=device)[:max_samples]]
    if neg_rw.size(0) > max_samples:
        neg_rw = neg_rw[torch.randperm(neg_rw.size(0), device=device)[:max_samples]]
    
    # Positive sample loss
    start_pos = pos_rw[:, 0]
    rest_pos = pos_rw[:, 1:].contiguous()
    h_start_pos = embedding[start_pos].unsqueeze(1)
    h_rest_pos = embedding[rest_pos.flatten()].view(pos_rw.size(0), -1, hidden_dim)
    out_pos = (h_start_pos * h_rest_pos).sum(dim=-1)
    pos_loss = torch.logsumexp(out_pos, dim=-1)
    
    # Negative sample loss
    start_neg = neg_rw[:, 0]
    rest_neg = neg_rw[:, 1:].contiguous()
    h_start_neg = embedding[start_neg].unsqueeze(1)
    h_rest_neg = embedding[rest_neg.flatten()].view(neg_rw.size(0), -1, hidden_dim)
    out_neg = (h_start_neg * h_rest_neg).sum(dim=-1)
    neg_loss = torch.logsumexp(out_neg, dim=-1)
    
    # Combined loss
    combined = torch.logsumexp(
        torch.cat([neg_loss.view(-1, 1), pos_loss.view(-1, 1)], dim=-1), 
        dim=-1
    )
    
    return -torch.mean(pos_loss - combined)


############################################################
# DGF Core
############################################################

class DGFCore(nn.Module):
    """
    DGF Core Model
    
    Performs graph filtering in both node domain and feature domain
    """
    def __init__(self, d1, d2, hidden_dim, alpha, beta, num_layers, device):
        super(DGFCore, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_layers = num_layers
        self.device = device

        # Linear projection layers for each modality
        self.linear1 = nn.Linear(d1, hidden_dim)
        self.linear2 = nn.Linear(d2, hidden_dim)

    def symmetric_softmax(self, ZM):
        """Compute symmetric softmax of feature similarity matrix"""
        n, d = ZM.shape
        scale = 1.0 / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=ZM.device))

        similarity = torch.mm(ZM.T, ZM) * scale
        exp_sim = torch.exp(similarity - similarity.max())  # Numerical stability
        
        row_sum = torch.sqrt(exp_sim.sum(dim=1, keepdim=True) + 1e-10)
        col_sum = torch.sqrt(exp_sim.sum(dim=0, keepdim=True) + 1e-10)

        S = exp_sim / (row_sum @ col_sum)
        return S

    def forward(self, X1, X2, NAM):
        """
        Args:
            X1: Text features [N, d1]
            X2: Image features [N, d2]
            NAM: Normalized adjacency matrix (sparse)
        
        Returns:
            ZM1, ZM2, HM: Projected and filtered representations
        """
        # Disable autocast to avoid Half precision issues (sparse matrix multiplication does not support Half)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Ensure inputs are float32
            X1 = X1.float()
            X2 = X2.float()
            
            # Project features
            ZM1 = self.linear1(X1)
            ZM2 = self.linear2(X2)

            # L2 normalization
            ZM1 = F.normalize(ZM1, p=2, dim=1)
            ZM2 = F.normalize(ZM2, p=2, dim=1)
            ZM = (ZM1 + ZM2) / 2

            # Feature domain shift operator
            with torch.no_grad():
                SM1 = self.symmetric_softmax(ZM1)
                SM2 = self.symmetric_softmax(ZM2)
                SM_combined = (SM1 + SM2) / 2

            alpha_term = self.alpha / (self.alpha + 1)
            beta_term = self.beta / (self.beta + 1)

            # Node domain filtering: sum((alpha/(alpha+1) * A)^t * Z)
            NAM = NAM.to(self.device).float()
            part_alpha = ZM.clone()
            current = ZM.clone()
            for _ in range(self.num_layers):
                current = alpha_term * torch.sparse.mm(NAM, current)
                part_alpha = part_alpha + current

            # Feature domain filtering: sum_{t=0}^{T} ((beta/(beta+1)) * S)^t
            d = ZM.size(1)
            sum_beta = torch.zeros(d, d, device=self.device)
            current_power = torch.eye(d, device=self.device)
            for _ in range(self.num_layers):
                sum_beta = sum_beta + current_power
                current_power = current_power @ (beta_term * SM_combined)

            # Combined filter
            HM = (1 / ((self.alpha + 1) * (self.beta + 1))) * (part_alpha @ sum_beta)
            HM = F.normalize(HM, p=2, dim=1)

            return ZM1, ZM2, HM


############################################################
# OpenMAG Wrapper
############################################################

class DGF(nn.Module):
    """DGF Wrapper for OpenMAG"""
    can_return_loss = True
    
    def __init__(
        self,
        v_feat_dim: int,
        t_feat_dim: int,
        hidden_dim: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        num_layers: int = 10,
        theta: float = 0.3,
        walks_per_node: int = 10,
        walk_length: int = 5,
        context_size: int = 3,
        aas: bool = True,
        std: float = 1.0,
        dropout: float = 0.0,
        num_clusters: int = 10,
        **kwargs
    ):
        super(DGF, self).__init__()
        self.v_feat_dim = v_feat_dim
        self.t_feat_dim = t_feat_dim
        self.hidden_dim = hidden_dim
        self.theta = theta
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.aas = aas
        self.std = std
        self.num_clusters = num_clusters
        self.dropout = dropout
        
        # Core model
        self.core = DGFCore(
            d1=t_feat_dim,
            d2=v_feat_dim,
            hidden_dim=hidden_dim,
            alpha=alpha,
            beta=beta,
            num_layers=num_layers,
            device=None  # Set later
        )
        
        # OpenMAG compatible output heads
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
        
        # Cache
        self._adj_cache = None
        self._norm_adj_cache = None
        self._kmeans_centers = None
        self._kmeans_labels = None
    
    def reset_parameters(self):
        """Reset all parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _build_adj(self, edge_index, n_nodes, device):
        """Build sparse adjacency matrix from edge_index"""
        src, dst = edge_index[0], edge_index[1]
        
        # Symmetrization
        all_src = torch.cat([src, dst])
        all_dst = torch.cat([dst, src])
        indices = torch.stack([all_src, all_dst], dim=0)
        indices = torch.unique(indices, dim=1)
        
        # Use float32 to avoid Half precision issues with AMP
        values = torch.ones(indices.size(1), device=device, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()
        
        # Normalization: D^(-1/2) A D^(-1/2)
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        inv_sqrt_deg = 1.0 / (torch.sqrt(degree) + 1e-10)
        
        row, col = adj.indices()
        norm_values = adj.values() * inv_sqrt_deg[row] * inv_sqrt_deg[col]
        norm_adj = torch.sparse_coo_tensor(adj.indices(), norm_values.float(), adj.size()).coalesce()
        
        return adj, norm_adj
    
    def forward(self, x, edge_index, use_subgraph=False):
        """
        Forward propagation
        
        Args:
            x: Concatenated features [N, v_dim + t_dim]
            edge_index: Edge index [2, E]
        
        Returns:
            Training mode: (h, x_v, x_t, loss)
            Evaluation mode: (h, x_v, x_t)
        """
        device = x.device
        n_nodes = x.size(0)
        self.core.device = device
        
        # Split features
        v_feat = x[:, :self.v_feat_dim]
        t_feat = x[:, self.v_feat_dim:]
        
        # Build adjacency matrix (if needed)
        if self._adj_cache is None or self._adj_cache.size(0) != n_nodes:
            self._adj_cache, self._norm_adj_cache = self._build_adj(edge_index, n_nodes, device)
        
        # Core forward propagation
        z1, z2, h = self.core(t_feat, v_feat, self._norm_adj_cache)
        
        # Output heads
        x_v = F.dropout(F.relu(self.vision_head(h)), p=self.dropout, training=self.training)
        x_t = F.dropout(F.relu(self.text_head(h)), p=self.dropout, training=self.training)
        
        if not self.training:
            return h, x_v, x_t
        
        # === Compute Loss ===
        # 1. Cross-modal contrastive loss
        loss_mod = compute_cross_modal_loss(z1, z2, h)
        
        # 2. Graph contrastive loss
        adj = self._adj_cache
        if self.aas:
            threshold = determine_threshold(adj, z1, z2, self.std)
            filtered_adj = filter_adj_by_similarity(adj, z1, z2, threshold)
        else:
            filtered_adj = adj
        
        pos_rw = pos_sample_rw(filtered_adj, n_nodes, self.walks_per_node, 
                               self.walk_length, self.context_size, device)
        neg_rw = neg_sample_rw(n_nodes, self.walks_per_node,
                               self.walk_length, self.context_size, device)
        loss_nbr = graph_contrastive_loss(pos_rw, neg_rw, h, self.hidden_dim, device)
        
        # 3. Community contrastive loss (using cached K-means)
        loss_comm = torch.tensor(0.0, device=device, requires_grad=True)
        if self._kmeans_centers is not None and self._kmeans_labels is not None:
            h_norm = F.normalize(h, dim=1, p=2)
            centers_norm = F.normalize(self._kmeans_centers, dim=1, p=2)
            loss_comm = cluster_loss(h_norm, centers_norm, self._kmeans_labels, self.theta)
        
        # === Combined Loss ===
        # Original DGF trainer uses loss normalization: loss = loss1/loss1.detach() + loss2/loss2.detach() + ...
        # Issue: Normalized loss is always around ~3.0, making it hard to observe training progress
        # Solution: Return raw_loss for display, but normalized_loss for gradient calculation
        # Trick: loss = raw_loss.detach() + (normalized_loss - normalized_loss.detach())
        
        eps = 1e-8
        
        # Raw loss (for display)
        raw_loss = loss_mod + loss_nbr + loss_comm
        
        # Normalized loss (for gradient computation, ensuring equal contribution)
        normalized_loss = (loss_mod / (loss_mod.detach() + eps) + 
                          loss_nbr / (loss_nbr.detach() + eps))
        
        # Add only if community loss is valid
        if loss_comm.requires_grad and loss_comm.item() > eps:
            normalized_loss = normalized_loss + loss_comm / (loss_comm.detach() + eps)
        
        # Combine: return raw_loss value, but gradients come from normalized_loss
        loss = raw_loss.detach() + (normalized_loss - normalized_loss.detach())
        
        return h, x_v, x_t, loss
    
    def update_kmeans(self, centers, labels):
        """Update cached K-means results (called by trainer)"""
        self._kmeans_centers = centers
        self._kmeans_labels = labels