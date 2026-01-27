import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

EPS = 1e-10

############################################################
# Graph Construction Utility Functions
############################################################

def build_sparse_adj_from_edges(src, dst, num_nodes, device, symmetric=True):
    """
    Build sparse adjacency matrix from edge index
    
    Args:
        src, dst: Source and destination node indices of edges
        num_nodes: Number of nodes
        device: Device
        symmetric: Whether to symmetrize
    
    Returns:
        Sparse adjacency matrix [N, N]
    """
    if symmetric:
        # Add reverse edges for symmetrization
        all_src = torch.cat([src, dst])
        all_dst = torch.cat([dst, src])
    else:
        all_src, all_dst = src, dst
    
    # Deduplication
    edge_index = torch.stack([all_src, all_dst], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    
    values = torch.ones(edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    return adj

def normalize_sparse_adj(adj, mode='sym'):
    """Normalize sparse adjacency matrix"""
    adj = adj.coalesce()
    N = adj.size(0)
    device = adj.device
    
    # Compute degree
    indices = adj.indices()
    values = adj.values()
    degree = torch.zeros(N, device=device)
    degree.index_add_(0, indices[0], values)
    
    if mode == 'sym':
        inv_sqrt_degree = 1.0 / (torch.sqrt(degree) + EPS)
        new_values = values * inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
    else:  # row
        inv_degree = 1.0 / (degree + EPS)
        new_values = values * inv_degree[indices[0]]
    
    return torch.sparse_coo_tensor(indices, new_values, adj.size()).coalesce()

def sparse_laplacian(adj_sparse):
    """Compute Laplacian matrix L = I - D^(-1/2) A D^(-1/2)"""
    adj_normalized = normalize_sparse_adj(adj_sparse, mode='sym')
    N = adj_sparse.size(0)
    device = adj_sparse.device
    
    adj_dense = adj_normalized.to_dense()
    L = torch.eye(N, device=device) - adj_dense
    return L

############################################################
# Encoder Components
############################################################

class GCNConv_dense(nn.Module):
    """GCN convolution layer for dense matrices"""
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output

class GCNConv_dgl(nn.Module):
    """GCN convolution layer for DGL graphs"""
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']

class GraphEncoder(nn.Module):
    """Graph encoder, supporting both sparse and dense modes"""
    def __init__(self, in_dim, hidden_dim, dropout, nlayers, sparse=False):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()
        
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 1):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 1):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.sparse = sparse
    
    def forward(self, x, Adj):
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gnn_encoder_layers[-1](x, Adj)
        return x

class Attention_shared(nn.Module):
    """Multimodal attention fusion"""
    def __init__(self, hidden_dim, attn_drop=0.1):
        super(Attention_shared, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        
        self.tanh = nn.Tanh()
        self.att_l = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)
        
        self.att_h = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)
        
        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
    
    def forward(self, embeds_l, embeds_h):
        beta = []
        attn_l = self.attn_drop(self.att_l)
        attn_h = self.attn_drop(self.att_h)
        for embed in embeds_l:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_l.matmul(sp.t()))
        for embed in embeds_h:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_h.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        
        z_fusion = 0
        embeds = embeds_l + embeds_h
        for i in range(len(embeds)):
            z_fusion += embeds[i] * beta[i]
        
        return F.normalize(z_fusion, dim=1, p=2)

class FusionRepresentation(nn.Module):
    """Adaptively fuse low-pass and high-pass representations"""
    def __init__(self):
        super(FusionRepresentation, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, z1, z2):
        a = torch.sigmoid(self.a)
        return (1 - a) * z1 + a * z2

############################################################
# Loss Functions
############################################################

class Contrast(nn.Module):
    """Contrastive Learning Loss (InfoNCE) - Memory Optimized Version"""
    def __init__(self, hidden_dim, project_dim, tau, max_samples=2048):
        super(Contrast, self).__init__()
        self.tau = tau
        self.max_samples = max_samples  # Max sample size
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
        for model in self.proj_2:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
    
    def forward(self, z_1, z_2, pos=None):
        """
        Contrastive learning loss - using sampling strategy
        
        If node count exceeds threshold, perform sampling to compute contrastive loss
        """
        N = z_1.size(0)
        device = z_1.device
        
        # If node count exceeds threshold, perform sampling
        if N > self.max_samples:
            indices = torch.randperm(N, device=device)[:self.max_samples]
            z_1_sampled = z_1[indices]
            z_2_sampled = z_2[indices]
        else:
            z_1_sampled = z_1
            z_2_sampled = z_2
        
        # Projection
        z_proj_1 = self.proj_1(z_1_sampled)
        z_proj_2 = self.proj_2(z_2_sampled)
        
        # Normalize for cosine similarity
        z_proj_1 = F.normalize(z_proj_1, dim=1, p=2)
        z_proj_2 = F.normalize(z_proj_2, dim=1, p=2)
        
        # Compute similarity matrix (Sampled size: max_samples × max_samples)
        sim_matrix = torch.mm(z_proj_1, z_proj_2.t()) / self.tau  # [S, S]
        
        # InfoNCE loss: positive samples on the diagonal
        labels = torch.arange(sim_matrix.size(0), device=device)
        
        # Bidirectional contrastive loss
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_1 + loss_2) / 2

############################################################
# DMGC Core Model
############################################################

class DMGCCore(nn.Module):
    """DMGC Core Model"""
    def __init__(
        self,
        modals: int,
        feats_dim: int,
        hidden_dim: int,
        tau: float,
        dropout: float,
        nlayer: int,
        sparse: bool = False
    ):
        super(DMGCCore, self).__init__()
        self.modals = modals
        self.feats_dim = feats_dim
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.tau = tau
        
        self.encoder = GraphEncoder(hidden_dim, hidden_dim, dropout, self.nlayer, sparse=sparse)
        self.att = Attention_shared(hidden_dim)
        
        self.contrast_l = Contrast(hidden_dim, hidden_dim, self.tau)
        self.contrast_h = Contrast(hidden_dim, hidden_dim, self.tau)
        self.contrast = Contrast(hidden_dim, hidden_dim, self.tau)
        
        self.fusion_1 = FusionRepresentation()
        self.fusion_2 = FusionRepresentation()
        
        # Multi-view projection layers (one per modality)
        self.multi_view_projections_t = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(feats_dim), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(modals)
        ])
        
        self.multi_view_projections_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(feats_dim), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(modals)
        ])
    
    def forward(self, f_list, adj_l, L_hs, lambda_cr, lambda_cm):
        """
        Args:
            f_list: [[text_features], [image_features]] - Each modality may have multiple relations
            adj_l: Homogeneous adjacency matrix
            L_hs: [L_h_text, L_h_image] - List of heterogeneous Laplacian matrices
            lambda_cr: Dual-frequency alignment loss weight
            lambda_cm: Cross-modal alignment loss weight
        """
        t_list = f_list[0]  # Text feature list
        m_list = f_list[1]  # Image feature list
        
        # Feature projection
        zt_list = []
        zm_list = []
        for i in range(len(t_list)):
            zt_list.append(self.multi_view_projections_t[i](t_list[i]))
            zm_list.append(self.multi_view_projections_m[i](m_list[i]))
        
        z_list = [zt_list, zm_list]
        
        # Dual-frequency filtering
        z_lps = []
        z_hps = []
        for i in range(self.modals):
            z_l_sublist = []
            z_h_sublist = []
            for j in range(len(z_list[0])):
                z_l = self.encoder(z_list[i][j], adj_l)
                z_h = self.encoder(z_list[i][j], L_hs[i])
                z_l_sublist.append(z_l)
                z_h_sublist.append(z_h)
            z_lps.append(z_l_sublist)
            z_hps.append(z_h_sublist)
        
        # Intra-relation averaging
        zt_mean_l = torch.stack(z_lps[0], dim=0).mean(dim=0)
        zm_mean_l = torch.stack(z_lps[1], dim=0).mean(dim=0)
        zt_mean_h = torch.stack(z_hps[0], dim=0).mean(dim=0)
        zm_mean_h = torch.stack(z_hps[1], dim=0).mean(dim=0)
        
        # Dual-frequency fusion
        z_t = self.fusion_1(zt_mean_l, zt_mean_h)
        z_m = self.fusion_2(zm_mean_l, zm_mean_h)
        
        # Cross-modal fusion
        z = self.att([z_t], [z_m])
        
        # Compute loss (keep contrastive loss only)
        loss_l = 0
        loss_h = 0
        loss_con = 0
        
        # Dual-frequency alignment loss
        z_lps_mean = [zt_mean_l, zm_mean_l]
        z_hps_mean = [zt_mean_h, zm_mean_h]
        for i in range(self.modals):
            z_lp = z_lps_mean[i]
            z_hp = z_hps_mean[i]
            loss_l += self.contrast_l(z_lp, z_t) + self.contrast_l(z_lp, z_m)
            loss_h += self.contrast_h(z_hp, z_t) + self.contrast_h(z_hp, z_m)
        
        # Cross-modal alignment loss
        loss_con += self.contrast(z_m, z_t)
        
        # Total loss (Dual-frequency + Cross-modal)
        loss = lambda_cr * (loss_l + loss_h) + lambda_cm * loss_con

        return loss, z
    
    def get_embeds(self, f_list, adj_l, L_hs):
        """Get embeddings (for evaluation)"""
        t_list = f_list[0]
        m_list = f_list[1]
        
        zt_list = []
        zm_list = []
        for i in range(len(t_list)):
            zt_list.append(self.multi_view_projections_t[i](t_list[i]))
            zm_list.append(self.multi_view_projections_m[i](m_list[i]))
        
        z_list = [zt_list, zm_list]
        
        z_lps = []
        z_hps = []
        for i in range(self.modals):
            z_l_sublist = []
            z_h_sublist = []
            for j in range(len(z_list[0])):
                z_l = self.encoder(z_list[i][j], adj_l)
                z_h = self.encoder(z_list[i][j], L_hs[i])
                z_l_sublist.append(z_l)
                z_h_sublist.append(z_h)
            z_lps.append(z_l_sublist)
            z_hps.append(z_h_sublist)
        
        zt_mean_l = torch.stack(z_lps[0], dim=0).mean(dim=0)
        zm_mean_l = torch.stack(z_lps[1], dim=0).mean(dim=0)
        zt_mean_h = torch.stack(z_hps[0], dim=0).mean(dim=0)
        zm_mean_h = torch.stack(z_hps[1], dim=0).mean(dim=0)
        
        z_t = self.fusion_1(zt_mean_l, zt_mean_h)
        z_m = self.fusion_2(zm_mean_l, zm_mean_h)
        z = self.att([z_t], [z_m])
        
        return z.detach()

############################################################
# OpenMAG Adapter
############################################################

class DMGC(nn.Module):
    """DMGC Wrapper for OpenMAG"""
    
    def __init__(
        self,
        v_feat_dim: int,
        t_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        tau: float = 1.0,
        lambda_cr: float = 0.001,
        lambda_cm: float = 1.0,
        dropout: float = 0.5,
        sparse: bool = False,
        graph_update_freq: int = -1,
        **kwargs
    ):
        """
        Args:
            v_feat_dim: Visual feature dimension
            t_feat_dim: Text feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            tau: Contrastive learning temperature parameter
            lambda_cr: Dual-frequency alignment loss weight
            lambda_cm: Cross-modal alignment loss weight
            dropout: Dropout rate
            sparse: Whether to use sparse matrices
            graph_update_freq: Graph update frequency (-1 means build only once)
        """
        super(DMGC, self).__init__()
        self.v_feat_dim = v_feat_dim
        self.t_feat_dim = t_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tau = tau
        self.lambda_cr = lambda_cr
        self.lambda_cm = lambda_cm
        self.dropout = dropout
        self.sparse = sparse
        self.can_return_loss = True
        self.graph_update_freq = graph_update_freq
        self._step_count = 0
        
        # Feature projection layers: project features of different dimensions to hidden_dim
        self.v_proj = nn.Linear(v_feat_dim, hidden_dim)
        self.t_proj = nn.Linear(t_feat_dim, hidden_dim)
        
        modals = 2  # Two modalities: text and image
        self.core = DMGCCore(
            modals=modals,
            feats_dim=hidden_dim,  # Projected feature dimension
            hidden_dim=hidden_dim,
            tau=tau,
            dropout=dropout,
            nlayer=num_layers,
            sparse=sparse
        )
        
        # OpenMAG modality output heads
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
        
        # Graph cache
        self._adj_l = None
        self._L_hs = None
        self._graph_built = False
    
    def reset_parameters(self):
        """Reset parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _build_graphs(self, edge_index, num_nodes, device):
        """
        Build adjacency matrix using original graph structure
        
        Args:
            edge_index: [2, E] edge indices
            num_nodes: Number of nodes
            device: Device
        """
        # Build sparse adjacency matrix from edge_index
        src, dst = edge_index[0], edge_index[1]
        adj_sparse = build_sparse_adj_from_edges(src, dst, num_nodes, device, symmetric=True)
        
        # Normalized adjacency matrix as homogeneous graph
        adj_l_normalized = normalize_sparse_adj(adj_sparse, mode='sym')
        adj_l = adj_l_normalized.to_dense()
        
        # Heterogeneous graph uses Laplacian matrix L = I - A_norm
        # Using shared graph structure for both modalities (simplified version)
        L_h = sparse_laplacian(adj_sparse)
        L_hs = [L_h, L_h]  # Two modalities share the same heterogeneous graph
        
        return adj_l, L_hs
    
    def forward(self, x, edge_index, use_subgraph=False):
        """
        Forward propagation
        
        Args:
            x: [N, v_feat_dim + t_feat_dim] - Concatenated features (Visual first, Text second)
            edge_index: [2, E] - PyTorch Geometric format edge indices
            use_subgraph: Whether to use subgraph (ignored)
        
        Returns:
            Training mode: (h, x_vision, x_text, loss)
            Evaluation mode: (h, x_vision, x_text)
        """
        device = x.device
        
        # Separate and project features
        v_feat_raw = x[:, :self.v_feat_dim]
        t_feat_raw = x[:, self.v_feat_dim:]
        v_feat = self.v_proj(v_feat_raw)
        t_feat = self.t_proj(t_feat_raw)
        
        # Build graph (control update frequency via graph_update_freq, or when node count changes)
        num_nodes = x.size(0)
        
        # Check if graph reconstruction is needed
        # 1. Not built yet
        # 2. Node count changed (varies per batch in mini-batch mode)
        # 3. graph_update_freq triggered
        size_changed = (self._adj_l is not None and self._adj_l.size(0) != num_nodes)
        should_update = (
            not self._graph_built or
            size_changed or
            (self.graph_update_freq > 0 and self._step_count % self.graph_update_freq == 0)
        )
        if should_update:
            self._adj_l, self._L_hs = self._build_graphs(edge_index, num_nodes, device)
            self._graph_built = True
        
        if self.training:
            self._step_count += 1
        
        # Convert to DMGC format: f_list = [[t_feat], [v_feat]]
        # Note: DMGC expects multiple relations per modality, here simplified to single relation due to dataset
        f_list = [[t_feat], [v_feat]]
        
        # Call core model
        if self.training:
            loss, h = self.core(f_list, self._adj_l, self._L_hs, self.lambda_cr, self.lambda_cm)
        else:
            h = self.core.get_embeds(f_list, self._adj_l, self._L_hs)
        
        # Output adaptation
        x_vision = F.dropout(F.relu(self.vision_head(h)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(h)), p=self.dropout, training=self.training)
        
        if self.training:
            return h, x_vision, x_text, loss
        else:
            return h, x_vision, x_text