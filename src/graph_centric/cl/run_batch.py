"""
Node Clustering
"""
import dgl
import torch
import os
import logging
from torch_geometric.data import Data
import numpy as np

# Get Hydra configuration logger
log = logging.getLogger(__name__)
from model.models import GCN, GraphSAGE, GAT, MLP, GIN, ChebNet, LGMRec, GCNII, GATv2, MHGAT
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
from model.UniGraph2 import UniGraph2
from model.DMGC import DMGC
from model.DGF import DGF
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from utils.pretrained_model import TrainableBackbone
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import random

def safe_kmeans(X, num_clusters, device, max_iter=100, tol=1e-4):
    """
    Safe K-means wrapper function
    
    Solves issues with kmeans_pytorch:
    1. NaN values causing center_shift=nan preventing termination
    2. No iteration limit
    
    Args:
        X: Input data [N, d]
        num_clusters: Number of clusters
        device: Device
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        labels: Cluster labels [N]
        centers: Cluster centers [K, d]
    """
    N = X.shape[0]
    
    # 1. Clean NaN/Inf
    X_clean = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Check if there are enough unique samples
    if N < num_clusters:
        # Fewer samples than clusters, return random labels
        labels = torch.randint(0, num_clusters, (N,), device=device)
        centers = X_clean[:num_clusters] if N >= num_clusters else X_clean
        return labels, centers
    
    # 3. Manually implement K-means with iteration limit
    # Randomly initialize centers
    indices = torch.randperm(N, device=device)[:num_clusters]
    centers = X_clean[indices].clone()
    
    labels = torch.zeros(N, dtype=torch.long, device=device)
    
    for iteration in range(max_iter):
        # Calculate distances and assign labels
        # Use batch calculation to avoid OOM
        batch_size = 4096
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_X = X_clean[start:end]
            distances = torch.cdist(batch_X, centers, p=2)  # [batch, K]
            labels[start:end] = distances.argmin(dim=1)
        
        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(num_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = X_clean[mask].mean(dim=0)
            else:
                # Empty cluster: randomly re-initialize
                new_centers[k] = X_clean[torch.randint(0, N, (1,), device=device)].squeeze()
        
        # Check convergence
        center_shift = torch.norm(new_centers - centers)
        centers = new_centers
        
        # Handle NaN (possibly due to empty clusters or numerical issues)
        if torch.isnan(center_shift) or torch.isinf(center_shift):
            # Numerical issue occurred, return current result
            break
        
        if center_shift < tol:
            break
    
    return labels, centers

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGE_CACHE = {}

def preload_images(image_paths):
    print(f">>> Pre-loading {len(image_paths)} images into RAM...")
    success = 0
    for p in tqdm(image_paths, desc="Caching Images"):
        if p and os.path.exists(p) and p not in IMAGE_CACHE:
            try:
                img = Image.open(p).convert('RGB')
                img_tensor = img_transforms(img)
                IMAGE_CACHE[p] = img_tensor
                success += 1
            except:
                pass
    print(f">>> Caching finished. {success}/{len(image_paths)} images loaded.")

class EndToEndModel(nn.Module):
    """End-to-end model wrapper"""
    def __init__(self, gnn_model, vision_backbone, modality):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.gnn = gnn_model
        self.modality = modality
        
    def forward(self, x_txt, raw_images, edge_index):
        v_emb = self.vision_backbone(raw_images)
        if self.modality == 'text':
            v_emb = torch.zeros_like(v_emb)
        x_fused = torch.cat([v_emb, x_txt], dim=1)
        gnn_output = self.gnn(x_fused, edge_index)
        if len(gnn_output) == 4:
            return gnn_output[0], gnn_output[1], gnn_output[2], gnn_output[3], v_emb
        else:
            return gnn_output[0], gnn_output[1], gnn_output[2], v_emb

def split_graph(nodes_num, train_ratio=0.6, val_ratio=0.2, fewshots=False, label=None):
    """Split dataset"""
    indices = np.random.permutation(nodes_num)
    if not fewshots:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)
        train_mask = torch.zeros(nodes_num, dtype=torch.bool)
        val_mask = torch.zeros(nodes_num, dtype=torch.bool)
        test_mask = torch.zeros(nodes_num, dtype=torch.bool)
        indices = torch.from_numpy(indices).to(torch.long)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
    return train_mask, val_mask, test_mask

def load_data(graph_path, v_emb_path, t_emb_path, image_path, train_ratio, val_ratio, fewshots=False, self_loop=True, undirected=True, modality="multimodal"):
    """Load data"""
    graph = dgl.load_graphs(graph_path)[0][0]
    if undirected:
        graph = dgl.add_reverse_edges(graph)
    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    is_trainable = (v_emb_path == "TRAINABLE_MODE")
    
    t_x = torch.from_numpy(np.load(t_emb_path)).to(torch.float32)
    image_paths = []
    
    if is_trainable:
        print(">>> [Data Loading] Mode: End-to-End Trainable (Raw Images)")
        v_x = None
        num_nodes = graph.num_nodes()
        for i in range(num_nodes):
            p = os.path.join(image_path, f"{i}.jpg")
            if not os.path.exists(p):
                p = os.path.join(image_path, f"{i}.png")
            if not os.path.exists(p):
                p = None
            image_paths.append(p)
        x = t_x
        v_dim = 768
    else:
        v_x = torch.from_numpy(np.load(v_emb_path)).to(torch.float32)
        if modality == 'text':
            v_x = torch.zeros_like(v_x)
        v_x = torch.nan_to_num(v_x, nan=0.0, posinf=0.0, neginf=0.0)
        t_x = torch.nan_to_num(t_x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([v_x, t_x], dim=1)
        v_dim = v_x.size(1)
    
    if modality == 'text':
        v_x = torch.zeros_like(v_x) if v_x is not None else None
    elif modality == 'visual':
        t_x = torch.zeros_like(t_x)
    elif modality == 'multimodal':
        pass
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    
    y = graph.ndata["label"]
    if "train_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"].to(torch.bool)
        val_mask = graph.ndata["val_mask"].to(torch.bool)
        test_mask = graph.ndata["test_mask"].to(torch.bool)
    else:
        train_mask, val_mask, test_mask = split_graph(graph.num_nodes(), train_ratio, val_ratio, fewshots, y)
    
    data = Data(
        x=x,
        v_dim=v_dim,
        t_dim=t_x.size(1),
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data.x = x
    if is_trainable:
        data.aux_info = {'image_paths': image_paths}
        data.is_trainable = True
    else:
        data.is_trainable = False
    return data

class GNNModel(nn.Module):
    def __init__(self, encoder, dim_hidden, dim_v, dim_t):
        super().__init__()
        self.encoder = encoder
        self.decoder_v = nn.Linear(dim_hidden, dim_v)
        self.decoder_t = nn.Linear(dim_hidden, dim_t)
        self.dim_hidden = dim_hidden
        
        # # Learnable cluster centers
        # self.num_clusters = num_clusters
        # if num_clusters is not None:
        #     self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, dim_hidden))
        #     nn.init.xavier_uniform_(self.cluster_centers)
        # else:
        #     self.cluster_centers = None
    
    # def init_cluster_centers(self, embeddings, num_clusters):
    #     """Initialize cluster centers using K-means"""
    #     device = embeddings.device
    #     if self.cluster_centers is None:
    #         self.num_clusters = num_clusters
    #         self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, self.dim_hidden).to(device))
        
    #     with torch.no_grad():
    #         embeddings_norm = F.normalize(embeddings, dim=1, p=2)
    #         # Initialize using K-means
    #         pred_labels, centers = kmeans(
    #             X=embeddings_norm,
    #             num_clusters=num_clusters,
    #             distance='euclidean',
    #             device=device
    #         )
    #         self.cluster_centers.data = centers
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder_v.reset_parameters()
        self.decoder_t.reset_parameters()
        # if self.cluster_centers is not None:
        #     nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(self, x, edge_index):
        loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        kwargs = {}
        if isinstance(self.encoder, (Net, MGAT)):
            kwargs['use_subgraph'] = False
            
        if self.training and hasattr(self.encoder, "can_return_loss") and self.encoder.can_return_loss:
            x, x_v, x_t, loss = self.encoder(x, edge_index, **kwargs)
        else:
            x, x_v, x_t = self.encoder(x, edge_index, **kwargs)
        out_v = self.decoder_v(x_v)
        out_t = self.decoder_t(x_t)
        if self.training:
            return x, out_v, out_t, loss
        return x, out_v, out_t

def set_seed(seed: int):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy (via optimal label mapping)
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        accuracy: Accuracy score
    """
    # Build confusion matrix
    n_clusters = max(y_pred.max() + 1, y_true.max() + 1)
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    
    for i in range(len(y_true)):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    
    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)
    
    # Calculate accuracy
    correct = confusion_matrix[row_ind, col_ind].sum()
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0

def evaluate_clustering(embeddings, true_labels, num_clusters=None, cluster_centers=None):
    """
    Evaluate clustering results
    
    Args:
        embeddings: Node embeddings [N, d]
        true_labels: True labels [N]
        num_clusters: Number of clusters (inferred from true_labels if None)
        cluster_centers: Optional cluster centers (if provided, use NN assignment instead of K-means)
    
    Returns:
        acc: Clustering accuracy
        nmi: Normalized Mutual Information
        ari: Adjusted Rand Index
        pred_labels: Predicted labels [N]
    """
    device = embeddings.device
    
    if num_clusters is None:
        num_clusters = int(true_labels.max().item() + 1)
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=1, p=2)
    
    if cluster_centers is not None:
        # Use provided cluster centers for nearest neighbor assignment (deterministic, no randomness)
        cluster_centers = cluster_centers.to(device)
        centers_norm = F.normalize(cluster_centers, dim=1, p=2)
        distances = torch.cdist(embeddings_norm, centers_norm, p=2)  # [N, K]
        pred_labels = distances.argmin(dim=1)  # [N]
    else:
        # Fallback: Use safe K-means for clustering
        pred_labels, _ = safe_kmeans(
            X=embeddings_norm,
            num_clusters=num_clusters,
            device=device,
            max_iter=100
        )
    
    # Convert to numpy for evaluation
    pred_labels_np = pred_labels.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()
    
    # Calculate clustering accuracy (requires label mapping)
    acc = cluster_accuracy(true_labels_np, pred_labels_np)
    
    # Calculate NMI
    nmi = normalized_mutual_info_score(true_labels_np, pred_labels_np)
    
    # Calculate ARI
    ari = adjusted_rand_score(true_labels_np, pred_labels_np)
    
    return acc, nmi, ari, pred_labels

def infoNCE_loss(out, orig_features, tau=0.07):
    """
    InfoNCE loss implementation
    - out: Feature embeddings output by model (batch_size, emb_dim)
    - orig_features: Original features (batch_size, feat_dim)
    - tau: Temperature coefficient
    """
    # 1. Feature normalization
    out_norm = F.normalize(out, p=2, dim=1)
    orig_norm = F.normalize(orig_features, p=2, dim=1)
    
    # 2. Calculate similarity matrix
    sim_matrix = torch.mm(out_norm, orig_norm.t()) / tau  # [batch_size, batch_size]
    
    # 3. Create labels (diagonal elements are positive samples)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # 4. Use cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

def node_contrastive_loss(embeddings, edge_index, temperature=0.5, num_neg_samples=5):
    """
    Node contrastive learning loss (fixed version)
    Use InfoNCE to make adjacent nodes more similar and random nodes more dissimilar
    
    Args:
        embeddings: Node embeddings [N, d]
        edge_index: Edge indices [2, E] (local index, suitable for mini-batch)
        temperature: Temperature parameter (higher temp makes distribution smoother)
        num_neg_samples: Number of negative samples per positive pair
    
    Returns:
        loss: Contrastive learning loss
    """
    device = embeddings.device
    N = embeddings.shape[0]
    
    # Boundary check
    if edge_index.shape[1] == 0 or N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=1, p=2)
    
    # Filter valid edges (ensure indices are within range)
    src, dst = edge_index
    valid_mask = (src < N) & (dst < N) & (src != dst)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    # Limit number of edges to avoid VRAM issues
    max_edges = min(src.shape[0], 2048)
    if src.shape[0] > max_edges:
        perm = torch.randperm(src.shape[0], device=device)[:max_edges]
        src = src[perm]
        dst = dst[perm]
    
    num_edges = src.shape[0]
    
    # Positive sample similarity: neighbor node pairs
    pos_sim = (embeddings_norm[src] * embeddings_norm[dst]).sum(dim=1)  # [E]
    
    # Negative sample similarity: randomly sampled nodes
    # Use vectorized batch sampling to avoid Python loops
    neg_indices = torch.randint(0, N, (num_edges, num_neg_samples), device=device)  # [E, K]
    
    # Calculate negative sample similarity
    src_emb = embeddings_norm[src].unsqueeze(1)  # [E, 1, d]
    neg_emb = embeddings_norm[neg_indices.flatten()].view(num_edges, num_neg_samples, -1)  # [E, K, d]
    neg_sim = (src_emb * neg_emb).sum(dim=2)  # [E, K]
    
    # Build logits: [pos, neg1, neg2, ..., negK]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature  # [E, 1+K]
    
    # Labels: positive sample at position 0
    labels = torch.zeros(num_edges, dtype=torch.long, device=device)
    
    # InfoNCE loss
    loss = F.cross_entropy(logits, labels)
    
    return loss

def graph_reconstruction_loss(embeddings, edge_index, neg_ratio=1.0):
    """
    Graph reconstruction loss (fixed version)
    Predict edge existence based on node embeddings, using cosine similarity loss
    
    Args:
        embeddings: Node embeddings [N, d]
        edge_index: Edge indices [2, E] (local index, suitable for mini-batch)
        neg_ratio: Ratio of negative samples to positive samples
    
    Returns:
        loss: Reconstruction loss
    """
    device = embeddings.device
    N = embeddings.shape[0]
    
    # Boundary check
    if edge_index.shape[1] == 0 or N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=1, p=2)
    
    # Filter valid edges
    src, dst = edge_index
    valid_mask = (src < N) & (dst < N) & (src != dst)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    # Limit number of edges
    max_edges = min(src.shape[0], 2048)
    if src.shape[0] > max_edges:
        perm = torch.randperm(src.shape[0], device=device)[:max_edges]
        src = src[perm]
        dst = dst[perm]
    
    num_pos = src.shape[0]
    num_neg = int(num_pos * neg_ratio)
    
    # Positive samples: actually existing edges
    pos_src_emb = embeddings_norm[src]  # [E, d]
    pos_dst_emb = embeddings_norm[dst]  # [E, d]
    pos_sim = (pos_src_emb * pos_dst_emb).sum(dim=1)  # [E]
    
    # Negative samples: random sampling (vectorized, no Python loops)
    neg_src = torch.randint(0, N, (num_neg,), device=device)
    neg_dst = torch.randint(0, N, (num_neg,), device=device)
    
    # Avoid self-loops
    same_mask = neg_src == neg_dst
    neg_dst[same_mask] = (neg_dst[same_mask] + 1) % N
    
    neg_src_emb = embeddings_norm[neg_src]  # [num_neg, d]
    neg_dst_emb = embeddings_norm[neg_dst]  # [num_neg, d]
    neg_sim = (neg_src_emb * neg_dst_emb).sum(dim=1)  # [num_neg]
    
    # Use SCE-style loss (more stable)
    # Positive samples: expect similarity close to 1
    # Negative samples: expect similarity close to -1 (or 0)
    pos_loss = (1 - pos_sim).mean()  # Positive samples should be similar
    neg_loss = F.relu(neg_sim + 0.5).mean()  # Negative sample similarity should be below -0.5 (margin)
    
    loss = pos_loss + neg_loss
    
    return loss

@torch.no_grad()
def evaluate(model, data, mask, config, num_clusters, cluster_centers=None):
    """
    Evaluate clustering results (full graph mode)
    
    Args:
        model: Model
        data: PyG Data object
        mask: Mask for evaluation nodes
        config: Configuration
        num_clusters: Number of clusters
        cluster_centers: Optional cluster centers
    
    Returns:
        acc, nmi, ari: Clustering metrics
    """
    model.eval()
    device = config.device
    
    # Full graph forward pass
    x = data.x.to(device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    edge_index = data.edge_index.to(device)
    
    kwargs = {}
    if isinstance(model, (Net, MGAT)):
        kwargs['use_subgraph'] = False

    res = model(x, edge_index, **kwargs)
    embeddings = res[0]
    embeddings = res[0]  # [N, hidden_dim]
    
    # Only take embeddings and labels of evaluation nodes
    eval_embeddings = embeddings[mask]
    eval_labels = data.y[mask].to(device)
    
    # Evaluate clustering results
    acc, nmi, ari, pred_labels = evaluate_clustering(
        eval_embeddings,
        eval_labels,
        num_clusters=num_clusters,
        cluster_centers=cluster_centers
    )
    
    return acc, nmi, ari

def train_and_eval(config, model, data, run_id=0):
    """
    Training and evaluation (full graph training mode)
    
    Use full graph training instead of mini-batch, more suitable for clustering tasks
    """
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    model = model.to(config.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.task.lr, 
        weight_decay=config.task.weight_decay
    )
    
    # Prepare full graph data
    device = config.device
    x = data.x.to(device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    
    num_clusters = int(y.max().item() + 1)
    print(f"=== Num Clusters: {num_clusters} ===" )
    
    # === Baseline: Evaluating raw features directly ===
    print(">>> Baseline: Evaluating raw features without GNN...")
    raw_embeddings_norm = F.normalize(x, dim=1, p=2)
    raw_pred_labels, _ = safe_kmeans(
        X=raw_embeddings_norm,
        num_clusters=num_clusters,
        device=device,
        max_iter=100
    )
    
    raw_nmi = normalized_mutual_info_score(y.cpu().numpy(), raw_pred_labels.cpu().numpy())
    raw_ari = adjusted_rand_score(y.cpu().numpy(), raw_pred_labels.cpu().numpy())
    raw_acc = cluster_accuracy(y.cpu().numpy(), raw_pred_labels.cpu().numpy())
    
    print(f">>> Raw Features Baseline: ACC: {raw_acc:.4f} | NMI: {raw_nmi:.4f} | ARI: {raw_ari:.4f}")
    print("=" * 60)
    
    best_val_acc = 0
    best_val_nmi = 0
    best_val_ari = 0
    final_test_acc = 0
    final_test_nmi = 0
    final_test_ari = 0
    
    for epoch in tqdm(range(config.task.n_epochs), desc="Training"):
        # === Full graph training steps ===
        model.train()
        optimizer.zero_grad()

        # DGF Model: Update K-means cache for community contrastive loss every epoch
        if config.model.name == "DGF" and epoch > 0:
            with torch.no_grad():
                model.eval()
                res_eval = model(x, edge_index)
                embeddings_for_kmeans = F.normalize(res_eval[0], dim=1, p=2)
                labels_km, centers_km = safe_kmeans(
                    embeddings_for_kmeans, num_clusters, device, max_iter=50
                )
                model.update_kmeans(centers_km, labels_km)
                model.train()
        
        # Forward pass (full graph)
        kwargs = {}
        if isinstance(model, (Net, MGAT)):
            kwargs['use_subgraph'] = False
            
        res = model(x, edge_index, **kwargs)
        embeddings = res[0]  # [N, hidden_dim]
        
        # Handle model output
        if len(res) >= 4:
            out_v, out_t = res[1], res[2]
            loss_model = res[3]
        else:
            out_v, out_t = res[1], res[2]
            loss_model = torch.tensor(0.0, device=device)

        # === Calculate Loss ===        
        # 1. Contrastive learning loss (full graph edges)
        if edge_index.shape[1] > 0:
            loss_contrastive = node_contrastive_loss(embeddings, edge_index)
        else:
            loss_contrastive = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 2. Graph reconstruction loss
        if edge_index.shape[1] > 0:
            loss_reconstruction = graph_reconstruction_loss(embeddings, edge_index)
        else:
            loss_reconstruction = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Total loss = contrastive loss + reconstruction loss + internal model loss
        loss = (config.task.lambda_contrastive * loss_contrastive + 
                config.task.lambda_reconstruction * loss_reconstruction +
                config.task.lambda_model * loss_model)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # === Evaluation ===
        if (epoch != 0 and epoch % 10 == 0) or epoch == config.task.n_epochs - 1:
            val_acc, val_nmi, val_ari = evaluate(model, data, data.val_mask, config, num_clusters)
            test_acc, test_nmi, test_ari = evaluate(model, data, data.test_mask, config, num_clusters)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                final_test_nmi = test_nmi
                final_test_ari = test_ari
            
            tqdm.write(f"[Run {run_id+1}] Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                      f"Val ACC: {val_acc:.4f} NMI: {val_nmi:.4f} ARI: {val_ari:.4f} | "
                      f"Test ACC: {test_acc:.4f} NMI: {test_nmi:.4f} ARI: {test_ari:.4f}")
    
    return best_val_acc, best_val_nmi, best_val_ari, final_test_acc, final_test_nmi, final_test_ari

def run_clustering(config):
    """Run clustering task"""
    print(config)
    np.random.seed(config.seed)
    
    # 1. Load data
    data = load_data(
        config.dataset.graph_path,
        config.dataset.v_emb_path,
        config.dataset.t_emb_path,
        config.dataset.image_path,
        config.dataset.train_ratio,
        config.dataset.val_ratio,
        config.task.fewshots,
        config.task.self_loop,
        config.task.undirected,
        config.modality
    ).to(config.device)
    
    if getattr(data, 'is_trainable', False):
        preload_images(data.aux_info['image_paths'])
    
    # 2. Build model
    num_clusters = int(data.y.max().item() + 1)
    print("=== Num Clusters ===")
    print("num_clusters:", num_clusters)
    in_dim = data.v_dim + data.t_dim
    
    if config.model.name == "MLP":
        encoder = MLP(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name == "GAT":
        encoder = GAT(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, heads=config.model.heads, dropout=config.model.dropout, att_dropout=config.model.att_dropout)
    elif config.model.name == "GCN":
        encoder = GCN(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name == "GIN":
        encoder = GIN(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name == "GraphSAGE":
        encoder = GraphSAGE(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name == "ChebNet":
        encoder = ChebNet(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, K=config.model.K, dropout=config.model.dropout)
    elif config.model.name == "RevGAT":
        encoder = RevGAT(in_feats=in_dim, n_hidden=config.model.hidden_dim, n_layers=config.model.num_layers, n_heads=config.model.heads, activation=F.relu, dropout=config.model.dropout)
    elif config.model.name == "GCNII":
        encoder = GCNII(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            alpha=config.model.alpha,
            theta=config.model.theta
        )
    elif config.model.name == "GATv2":
        encoder = GATv2(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            heads=config.model.heads,
            dropout=config.model.dropout,
            att_dropout=config.model.att_dropout
        )
    elif config.model.name == "LGMRec":
        encoder = LGMRec(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            hyper_num=config.model.hyper_num,
            alpha=config.model.alpha
        )
    elif config.model.name == "MMGCN":
        encoder = Net(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            num_nodes=data.x.size(0),
            aggr_mode='mean',
            concate=False,
            num_layers=config.model.num_layers,
            has_id=True,
            dim_x=config.model.hidden_dim,
            v_dim=data.v_dim
        )
    elif config.model.name == "MGAT":
        encoder = MGAT(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            num_nodes=data.x.size(0),
            num_layers=config.model.num_layers,
            dim_x=config.model.hidden_dim,
            v_dim=data.v_dim
        )
        config.model.hidden_dim = config.model.hidden_dim * config.model.num_layers
    elif config.model.name == "MHGAT":
        encoder = MHGAT(
            v_dim=data.v_dim,
            t_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            heads=config.model.get('heads', 4)
        )
    elif config.model.name == "UniGraph2":
        encoder = UniGraph2(
            v_feat_dim=768,
            t_feat_dim=data.x.size(1) - 768,
            hidden_dim=config.model.hidden_dim,
            num_experts=config.model.num_experts,
            num_selected_experts=config.model.num_selected_experts,
            num_layers=config.model.num_layers,
            feat_drop_rate=config.model.feat_drop_rate,
            edge_mask_rate=config.model.edge_mask_rate,
            gamma=config.model.gamma,
            lambda_spd=config.model.lambda_spd,
            dropout=config.model.dropout
        )
    elif config.model.name == "DMGC":
        encoder = DMGC(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            embed_dim=config.model.get('embed_dim', 128),
            num_layers=config.model.num_layers,
            k_l=config.model.get('k_l', 10),
            k_h=config.model.get('k_h', 6),
            tau=config.model.get('tau', 1.0),
            lambda_cr=config.model.get('lambda_cr', 0.001),
            lambda_cm=config.model.get('lambda_cm', 1.0),
            dropout=config.model.dropout,
            num_clusters=num_clusters,
            sparse=config.model.get('sparse', False),
            graph_update_freq=config.model.get('graph_update_freq', -1),
        )
    elif config.model.name == "DGF":
        encoder = DGF(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            alpha=config.model.get('alpha', 1.0),
            beta=config.model.get('beta', 1.0),
            num_layers=config.model.get('num_layers', 10),
            theta=config.model.get('theta', 0.3),
            walks_per_node=config.model.get('walks_per_node', 10),
            walk_length=config.model.get('walk_length', 5),
            context_size=config.model.get('context_size', 3),
            aas=config.model.get('aas', True),
            std=config.model.get('std', 1.0),
            dropout=config.model.get('dropout', 0.0),
            num_clusters=num_clusters,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model.name}")
    
    # 3. Assemble model
    if config.model.name in ["MMGCN", "MGAT", "MHGAT", "UniGraph2", "DMGC", "DGF"]:
        # These models already return (h, x_vision, x_text, [loss])
        gnn_model = encoder
    else:
        gnn_model = GNNModel(encoder, config.model.hidden_dim, data.v_dim, data.t_dim)
    
    # 4. Wrapper (if End-to-End is needed)
    if getattr(data, 'is_trainable', False):
        print(f"=== Wrapper: End-to-End Trainable ({config.visual_encoder}) ===")
        backbone = TrainableBackbone(config.visual_encoder, target_dim=data.v_dim)
        model = EndToEndModel(gnn_model, backbone, modality=config.modality)
    else:
        print("=== Wrapper: Frozen Features ===")
        model = gnn_model
    
    # 5. Training and testing
    accs = []
    nmis = []
    aris = []
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        best_val_acc, best_val_nmi, best_val_ari, final_test_acc, final_test_nmi, final_test_ari = train_and_eval(config, model, data, run)
        accs.append(final_test_acc)
        nmis.append(final_test_nmi)
        aris.append(final_test_ari)
    
    accs_mean = np.mean(accs) * 100
    accs_std = np.std(accs) * 100
    nmis_mean = np.mean(nmis) * 100
    nmis_std = np.std(nmis) * 100
    aris_mean = np.mean(aris) * 100
    aris_std = np.std(aris) * 100
    
    # Use logging to record final results (Hydra automatically saves to log file)
    log.info("="*60)
    log.info("[CL Task] Final Results")
    log.info(f"Model: {config.model.name} | Dataset: {config.dataset.name}")
    log.info(f"Average Test Accuracy over {config.task.n_runs} runs: {accs_mean:.2f} ± {accs_std:.2f}")
    log.info(f"Average Test NMI over {config.task.n_runs} runs: {nmis_mean:.2f} ± {nmis_std:.2f}")
    log.info(f"Average Test ARI over {config.task.n_runs} runs: {aris_mean:.2f} ± {aris_std:.2f}")
    log.info(f"Individual ACCs: {[round(a * 100, 2) for a in accs]}")
    log.info(f"Individual NMIs: {[round(n * 100, 2) for n in nmis]}")
    log.info(f"Individual ARIs: {[round(a * 100, 2) for a in aris]}")
    log.info("="*60)
    
    # Also keep print for terminal viewing
    print(f"Test ACCs: {[round(a * 100, 2) for a in accs]}")
    print(f"Average Test Accuracy over {config.task.n_runs} runs: {accs_mean:.2f} ± {accs_std:.2f}")
    print(f"Test NMIs: {[round(n * 100, 2) for n in nmis]}")
    print(f"Average Test NMI over {config.task.n_runs} runs: {nmis_mean:.2f} ± {nmis_std:.2f}")
    print(f"Test ARIs: {[round(a * 100, 2) for a in aris]}")
    print(f"Average Test ARI over {config.task.n_runs} runs: {aris_mean:.2f} ± {aris_std:.2f}")