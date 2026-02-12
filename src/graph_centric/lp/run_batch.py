import dgl
import torch
import logging
from torch_geometric.data import Data
import numpy as np

log = logging.getLogger(__name__)
from model.models import GCN, GraphSAGE, GAT, MLP, GIN, ChebNet, LGMRec, GCNII, GATv2, MHGAT
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
from model.GSMN import GSMN
from model.UniGraph2 import UniGraph2
from model.DMGC import DMGC
from model.DGF import DGF
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import LinkNeighborLoader

import os

def split_edge(graph, val_ratio=0.2, test_ratio=0.2, num_neg=150, path=None):
    if os.path.exists(os.path.join(path, f'edge_split_{num_neg}.pt')):
        edge_split = torch.load(os.path.join(path, f'edge_split_{num_neg}.pt'))
    else:
        # edges = np.arange(graph.num_edges())
        # edges = np.random.permutation(edges)
        # edges = torch.arange(graph.num_edges()) 
        edges = torch.randperm(graph.num_edges()) 

        source, target = graph.edges()

        val_size = int(len(edges) * val_ratio)
        test_size = int(len(edges) * test_ratio)
        test_source, test_target = source[edges[:test_size]], target[edges[:test_size]]
        val_source, val_target = source[edges[test_size:test_size + val_size]], target[edges[test_size:test_size + val_size]]
        train_source, train_target = source[edges[test_size + val_size:]], target[edges[test_size + val_size:]]

        val_target_neg = torch.randint(low=0, high=graph.num_nodes(), size=(len(val_source), int(num_neg)))
        test_target_neg = torch.randint(low=0, high=graph.num_nodes(), size=(len(test_source), int(num_neg)))

        edge_split = {'train': {'source_node': train_source, 'target_node': train_target},
            'valid': {'source_node': val_source, 'target_node': val_target,
                    'target_node_neg': val_target_neg},
            'test': {'source_node': test_source, 'target_node': test_target,
                    'target_node_neg': test_target_neg}}
        if os.path.exists(path):  
            torch.save(edge_split, os.path.join(path, f'edge_split_{num_neg}.pt'))

    return edge_split

        
def load_data(graph_path, v_emb_path, t_emb_path, val_ratio=0.1, test_ratio=0.2, num_neg=150, path=None, fewshots=False, self_loop=False, undirected=False, modality="multimodal"):
    graph = dgl.load_graphs(graph_path)[0][0]
    # Self-loops, undirected graph
    if undirected:
        graph = dgl.add_reverse_edges(graph)
    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    
    # Embeddings, labels
    # v_x = torch.load(v_emb_path)
    # t_x = torch.load(t_emb_path)
    v_x = torch.from_numpy(np.load(v_emb_path)).to(torch.float32)
    t_x = torch.from_numpy(np.load(t_emb_path)).to(torch.float32)
    max_val_v = torch.finfo(v_x.dtype).max  # Get the maximum finite value for this data type
    min_val_v = torch.finfo(v_x.dtype).min  # Get the minimum finite value for this data type
    max_val_t = torch.finfo(t_x.dtype).max  # Get the maximum finite value for this data type
    min_val_t = torch.finfo(t_x.dtype).min  # Get the minimum finite value for this data type
    v_x = torch.nan_to_num(v_x, nan=0.0, posinf=max_val_v, neginf=min_val_v)
    t_x = torch.nan_to_num(t_x, nan=0.0, posinf=max_val_t, neginf=min_val_t)
    if modality == 'text':
        v_x = torch.zeros_like(v_x) 
    elif modality == 'visual':
        t_x = torch.zeros_like(t_x)
    elif modality == 'multimodal':
        pass
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    x = torch.cat([v_x, t_x], dim=1)
    print("Input data statistics:")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")
    print(f"NaN in x: {torch.isnan(x).any()}, Inf in x: {torch.isinf(x).any()}")
    print(x.shape)
    # Split dataset
    edge_split = split_edge(graph, val_ratio=val_ratio, test_ratio=test_ratio, num_neg=num_neg, path=path)

    train_edges = torch.stack(
        (edge_split['train']['source_node'], edge_split['train']['target_node']), 
        dim=1
    ).t()
    adj_t = SparseTensor.from_edge_index(train_edges).t()
    adj_t = adj_t.to_symmetric()
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, v_dim=v_x.size(1), t_dim=t_x.size(1), edge_split=edge_split, edge_index=edge_index, adj_t=adj_t)

class Linear_v_t(nn.Module):
    def __init__(self, in_channels, out_channels_v, out_channels_t):
        super(Linear_v_t, self).__init__()

        self.v = nn.Linear(in_channels, out_channels_v)
        self.t = nn.Linear(in_channels, out_channels_t)

    def reset_parameters(self):
        self.v.reset_parameters()
        self.t.reset_parameters()

    def forward(self, x_v, x_t):
        out_v = self.v(x_v)
        out_t = self.t(x_t)
        return out_v, out_t

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # Calculate embedding similarity between two nodes
        x = x_i * x_j  # Dot product to calculate similarity

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return torch.sigmoid(x)


def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, predictor, x, adj_t, edge_split, num_neg=1000, k_list=[1,3,10], batch_size=2048):
    model.eval()
    predictor.eval()

    if hasattr(adj_t, 'coo'):
        edge_index = torch.stack(adj_t.coo()[:2], dim=0).to(x.device)
    else:
        edge_index = adj_t
    # Get validation set edges
    source_edge = edge_split['valid']['source_node'].to(x.device)  # [num_pos]
    target_edge = edge_split['valid']['target_node'].to(x.device)  # [num_pos]
    neg_target_edge = edge_split['valid']['target_node_neg'].to(x.device)  # [num_pos, num_neg]

    # Get node embeddings
    emb, out_v, out_t = model(x, edge_index, use_subgraph=False)

    # Batch calculation of positive sample scores
    pos_preds = []
    for batch in DataLoader(range(source_edge.size(0)), batch_size):

        src = source_edge[batch]
        dst = target_edge[batch]

        pos_preds.append(predictor(emb[src], emb[dst]).squeeze().cpu()) 
    pos_out = torch.cat(pos_preds, dim=0)  
    # Batch calculation of negative sample scores
    neg_preds = []
    num_pos = source_edge.size(0)

    flat_source = source_edge.view(-1, 1).repeat(1, num_neg).view(-1) 
    flat_neg_target = neg_target_edge.view(-1) 
    for batch in DataLoader(range(flat_source.size(0)), batch_size):
        src = flat_source[batch]
        neg = flat_neg_target[batch]
        neg_preds.append(predictor(emb[src], emb[neg]).squeeze().cpu())  
    neg_out = torch.cat(neg_preds, dim=0).view(-1, num_neg)
    # Calculate evaluation metrics
    pos_out = pos_out.unsqueeze(1) 
    all_scores = torch.cat([pos_out, neg_out], dim=1)  
    
    ranks = (all_scores >= pos_out).sum(dim=1) 
    # Calculate Hits@K for each K value
    hits_results = {}
    for k in sorted(k_list):
        hits_at_k = (ranks <= k).float().mean().item()
        hits_results[f'Hits@{k}'] = hits_at_k
    
    # Calculate MRR
    mrr = (1.0 / ranks.float()).mean().item()
    
    return {**hits_results, 'MRR': mrr}

@torch.no_grad()
def test(model, predictor, x, adj_t, edge_split, num_neg=1000, k_list=[1,3,10], batch_size=2048):
    model.eval()
    predictor.eval()

    if hasattr(adj_t, 'coo'):
        edge_index = torch.stack(adj_t.coo()[:2], dim=0).to(x.device)
    else:
        edge_index = adj_t
    
    # Get test set edges
    source_edge = edge_split['test']['source_node'].to(x.device)  # [num_pos]
    target_edge = edge_split['test']['target_node'].to(x.device)  # [num_pos]
    neg_target_edge = edge_split['test']['target_node_neg'].to(x.device)  # [num_pos, num_neg]

    # Get node embeddings
    emb, out_v, out_t = model(x, edge_index, use_subgraph=False)

    # Batch calculation of positive sample scores
    pos_preds = []
    for batch in DataLoader(range(source_edge.size(0)), batch_size):

        src = source_edge[batch]
        dst = target_edge[batch]

        pos_preds.append(predictor(emb[src], emb[dst]).squeeze().cpu())  
    pos_out = torch.cat(pos_preds, dim=0)  # [num_pos]
    # Batch calculation of negative sample scores
    neg_preds = []
    num_pos = source_edge.size(0)

    flat_source = source_edge.view(-1, 1).repeat(1, num_neg).view(-1) 
    flat_neg_target = neg_target_edge.view(-1)  
    for batch in DataLoader(range(flat_source.size(0)), batch_size):
        src = flat_source[batch]
        neg = flat_neg_target[batch]
        neg_preds.append(predictor(emb[src], emb[neg]).squeeze().cpu()) 
    neg_out = torch.cat(neg_preds, dim=0).view(-1, num_neg)
    # Calculate evaluation metrics
    pos_out = pos_out.unsqueeze(1)  
    all_scores = torch.cat([pos_out, neg_out], dim=1) 

    ranks = (all_scores >= pos_out).sum(dim=1)  
    # Calculate Hits@K for each K value
    hits_results = {}
    for k in sorted(k_list):
        hits_at_k = (ranks <= k).float().mean().item()
        hits_results[f'Hits@{k}'] = hits_at_k
    
    # Calculate MRR
    mrr = (1.0 / ranks.float()).mean().item()
    
    return {**hits_results, 'MRR': mrr}

def infoNCE_loss(out, orig_features, tau=0.07):
    """
    InfoNCE loss implementation
    - out: Feature embeddings output by the model (batch_size, emb_dim)
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

def train(model, predictor, linear_v_t,
          data, config, x, adj_t, edge_split,
          optimizer, batch_size):

    model.train()
    predictor.train()
    linear_v_t.train()

    train_edge_index = torch.stack([
        edge_split['train']['source_node'],
        edge_split['train']['target_node']
    ], dim=0)

    loader = LinkNeighborLoader(
        data=data,
        edge_label_index=train_edge_index,
        edge_label=torch.ones(train_edge_index.size(1)),
        num_neighbors=config.task.num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=0.0,
    )

    total_loss = total_examples = 0
    MAX_NCE = 1024

    for subgraph in loader:
        optimizer.zero_grad()
        subgraph = subgraph.to(config.device)
        n_id = subgraph.n_id

        # Handle models returning 4 values (with loss_model)
        loss_model = 0
        if hasattr(model, 'can_return_loss') and model.can_return_loss and model.training:
            emb, out_v, out_t, loss_model = model(subgraph.x, subgraph.edge_index, use_subgraph=True, n_id=n_id)
        else:
            emb, out_v, out_t = model(subgraph.x, subgraph.edge_index, use_subgraph=True, n_id=n_id)
        out_v, out_t = linear_v_t(out_v, out_t)

        nce_idx = torch.arange(
            min(subgraph.x.size(0), MAX_NCE),
            device=subgraph.x.device
        )

        loss_v = infoNCE_loss(
            out_v[nce_idx],
            subgraph.x[nce_idx, :data.v_dim]
        ) if config.modality != 'text' else 0

        loss_t = infoNCE_loss(
            out_t[nce_idx],
            subgraph.x[nce_idx, data.v_dim:]
        ) if config.modality != 'visual' else 0

        src, dst = subgraph.edge_label_index
        num_pos = src.size(0)

        pos_out = predictor(emb[src], emb[dst])

        dst_neg = torch.randint(
            0, subgraph.num_nodes,
            (num_pos,), device=subgraph.x.device
        )
        dst_neg = torch.where(dst_neg == dst, (dst_neg + 1) % subgraph.num_nodes, dst_neg)

        neg_out = predictor(emb[src], emb[dst_neg])

        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss \
               + config.task.lambda_v * loss_v \
               + config.task.lambda_t * loss_t \
               + config.task.get('lambda_model', 0.0) * loss_model

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_pos
        total_examples += num_pos

    return total_loss / total_examples


def run_lp(config):
    print(config)
    np.random.seed(config.seed)
    # 1. Load data
    data = load_data(
        config.dataset.graph_path, 
        config.dataset.v_emb_path, 
        config.dataset.t_emb_path, 
        val_ratio=config.dataset.lp_val_ratio, 
        test_ratio=config.dataset.lp_test_ratio, 
        num_neg=config.dataset.num_neg, 
        path=config.dataset.edge_split_path, 
        modality=config.modality
    ).to(config.device)

    # 2. Build model
    if config.model.name =="MLP":
        encoder = MLP(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GAT":
        encoder = GAT(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, heads=config.model.heads, dropout=config.model.dropout, att_dropout=config.model.att_dropout)
    elif config.model.name =="GCN":
        encoder = GCN(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GIN":
        encoder = GIN(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GraphSAGE":
        encoder = GraphSAGE(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="ChebNet":
        encoder = ChebNet(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, K=config.model.K, dropout=config.model.dropout)
    elif config.model.name == "RevGAT":
        encoder = RevGAT(in_feats=data.x.size(1),n_hidden=config.model.hidden_dim, n_layers=config.model.num_layers, n_heads=config.model.heads, activation=F.relu, dropout=config.model.dropout)
    elif config.model.name == "GCNII":
        encoder = GCNII(
            in_dim=data.x.size(1),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            alpha=config.model.alpha,
            theta=config.model.theta
        )
    elif config.model.name == "GATv2":
        encoder = GATv2(
            in_dim=data.x.size(1),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            heads=config.model.heads,         # Number of attention heads
            dropout=config.model.dropout,
            att_dropout=config.model.att_dropout # Dropout for attention coefficients
        )
    elif config.model.name == "MGNet":
        encoder = MGNet(
            in_dim=data.x.size(1),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            process=config.model.process,
            view_pooling=config.model.view_pooling
        )
    elif config.model.name == "LGMRec":
        encoder = LGMRec(
            in_dim=data.x.size(1),
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
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            num_experts=config.model.get('num_experts', 8),
            num_selected_experts=config.model.get('num_selected_experts', 2),
            num_layers=config.model.num_layers,
            feat_drop_rate=config.model.get('feat_drop_rate', 0.1),
            edge_mask_rate=config.model.get('edge_mask_rate', 0.1),
            gamma=config.model.get('gamma', 2.0),
            lambda_spd=config.model.get('lambda_spd', 0.5),
            dropout=config.model.dropout
        )
    elif config.model.name == "DMGC":
        encoder = DMGC(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            tau=config.model.get('tau', 1.0),
            lambda_cr=config.model.get('lambda_cr', 0.001),
            lambda_cm=config.model.get('lambda_cm', 1.0),
            dropout=config.model.dropout,
        )
    elif config.model.name == "DGF":
        encoder = DGF(
            v_feat_dim=data.v_dim,
            t_feat_dim=data.t_dim,
            hidden_dim=config.model.hidden_dim,
            alpha=config.model.get('alpha', 1.0),
            beta=config.model.get('beta', 1.0),
            num_layers=config.model.get('num_layers', 10),
            dropout=config.model.get('dropout', 0.0),
        )
    encoder.to(config.device)
    predictor = LinkPredictor(in_channels=config.model.hidden_dim, hidden_channels=config.task.predictor_hidden, out_channels=1, num_layers=config.task.predictor_layers, dropout=config.task.predictor_dropout).to(config.device)
    linear_v_t = Linear_v_t(config.model.hidden_dim, data.v_dim, data.t_dim).to(config.device)
    # 3. Training & Testing
    all_run_results = []  # Store the best test results dictionary for all runs
    
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) +
            list(predictor.parameters()) +
            list(linear_v_t.parameters()),
            lr=config.task.lr
        )
        encoder.reset_parameters() if hasattr(encoder, 'reset_parameters') else None
        predictor.reset_parameters() if hasattr(predictor, 'reset_parameters') else None
        
        best_mrr = 0
        best_test_results = {} # Best results for the current run
        
        # Training
        for epoch in range(config.task.n_epochs):
            train_loss = train(encoder, predictor, linear_v_t, data, config, data.x, data.adj_t, data.edge_split, optimizer, batch_size=config.task.batch_size)
            print(f"[Run {run+1}] Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            if epoch % 10 == 0 or epoch == config.task.n_epochs - 1:
                results = evaluate(encoder, predictor, data.x, data.adj_t, data.edge_split, config.dataset.num_neg, k_list=config.task.k_list)
                print("[VAL]")
                print(results)
                if results["MRR"] > best_mrr:
                    best_mrr = results["MRR"]
                    test_results = test(encoder, predictor, data.x, data.adj_t, data.edge_split, config.dataset.num_neg, k_list=config.task.k_list)
                    best_test_results = test_results
                    print(f"[TEST]")
                    print(test_results)

        # After the current run ends, add the best results to the list
        if best_test_results:
            all_run_results.append(best_test_results)

    # 4. Aggregate results from all runs and output mean and standard deviation
    if len(all_run_results) > 0:
        print("\n" + "="*20 + " Final Results " + "="*20)
        # Get all keys (e.g., 'Hits@1', 'Hits@3', 'MRR', etc.)
        metrics = all_run_results[0].keys()
        
        # Use logging to record final results (Hydra automatically saves to log file)
        log.info("="*60)
        log.info("[LP Task] Final Results")
        log.info(f"Model: {config.model.name} | Dataset: {config.dataset.name}")
        
        for metric in metrics:
            # Extract the metric value for each run
            values = [res[metric] for res in all_run_results]
            
            # Calculate mean and standard deviation, and convert to percentage
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            
            log.info(f"Average Test {metric} over {config.task.n_runs} runs: {mean_val:.2f} ± {std_val:.2f}")
            
            print(
                f"Average Test {metric} over {config.task.n_runs} runs: "
                f"{mean_val:.2f} ± {std_val:.2f}"
            )
        
        log.info("="*60)