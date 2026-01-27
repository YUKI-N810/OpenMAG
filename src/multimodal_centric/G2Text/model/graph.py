import torch.nn as nn
from model.models import GCN, GraphSAGE, GAT, MLP, GIN, ChebNet, LGMRec
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
from model.MMA import MMAConv
from model.DGF import DGF

class GNNAdapter(nn.Module):
    def __init__(self, cfg, embedding_dim):
        super().__init__()
        if cfg.model.name == 'MLP':
            self.gnn = MLP(embedding_dim, hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers, dropout=cfg.model.dropout)
        if cfg.model.name == 'GCN':
            self.gnn = GCN(embedding_dim, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout)
        elif cfg.model.name == 'GAT':
            self.gnn = GAT(embedding_dim, cfg.model.hidden_dim, cfg.model.num_layers, heads=cfg.model.heads, dropout=cfg.model.dropout) # heads 需要配置
        elif cfg.model.name == 'GraphSAGE':
            self.gnn = GraphSAGE(embedding_dim, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout)
        elif cfg.model.name == "LGMRec":
            self.gnn = LGMRec(
                in_dim=embedding_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout,
                hyper_num=cfg.model.hyper_num, 
                alpha=cfg.model.alpha      
            )
        elif cfg.model.name == "DGF":
            v_dim = embedding_dim // 2
            t_dim = embedding_dim - v_dim
            self.gnn = DGF(
                v_feat_dim=v_dim, 
                t_feat_dim=t_dim, 
                hidden_dim=cfg.model.hidden_dim,
                alpha=cfg.model.get('alpha', 1.0),
                beta=cfg.model.get('beta', 1.0),
                num_layers=cfg.model.get('num_layers', 1), 
                dropout=cfg.model.get('dropout', 0.0)
            )
        
        if cfg.model.hidden_dim != embedding_dim:
            self.proj = nn.Linear(cfg.model.hidden_dim, embedding_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x, edge_index):
        outputs = self.gnn(x, edge_index)
        aux_loss = 0.0
        if len(outputs) == 3:
            out = outputs[0]
        elif len(outputs) == 4:
            # DGF -> (out, x_v, x_t, loss)
            out = outputs[0]
            aux_loss = outputs[3]
        else:
            out = outputs[0]
        out = self.proj(out)
    
        return out, aux_loss