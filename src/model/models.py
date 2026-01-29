# models/encoder.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, ChebConv, GravNetConv, LGConv, GCN2Conv, GATv2Conv

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            input_dim = hidden_dim if i > 0 else in_dim
            self.linears.append(nn.Linear(input_dim, hidden_dim))

            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        h = x

        for i in range(self.num_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = F.dropout(h, p=self.dropout, training=self.training)

        x = self.linears[-1](h)

        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # Input Layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        # Hidden Layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x, edge_index, use_subgraph=False):
        for i, conv in enumerate(self.convs):     
            x = conv(x, edge_index.to(x.device))
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text 

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index, use_subgraph=False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, dropout, att_dropout=0):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=att_dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=att_dropout))

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index, use_subgraph=False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text
    
class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConv(mlp)
            self.convs.append(conv)

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)
        return x, x_vision, x_text

class ChebNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, K, dropout):
        super(ChebNet, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(ChebConv(in_dim, hidden_dim, K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) 
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)
        return x, x_vision, x_text

class HGNNLayer(nn.Module):
    def __init__(self, in_dim, hyper_num, n_layers=1, tau=0.5):
        super(HGNNLayer, self).__init__()
        self.hyper_num = hyper_num
        self.tau = tau
        self.n_layers = n_layers
        self.hyper_projector = nn.Linear(in_dim, hyper_num, bias=False)
        nn.init.xavier_uniform_(self.hyper_projector.weight)

    def forward(self, x):
        hyper_score = self.hyper_projector(x)
        H = F.gumbel_softmax(hyper_score, tau=self.tau, dim=1, hard=False)
        out = x
        for _ in range(self.n_layers):
            # Node -> HyperEdge (Aggregation)
            # lat: [N_hyper, Dim]
            lat = torch.mm(H.T, out)
            # HyperEdge -> Node (Distribution)
            # out: [N_nodes, Dim]
            out = torch.mm(H, lat)
        return out

class LGMRec(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, hyper_num=32, alpha=0.1):
        super(LGMRec, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.num_layers = num_layers
        self.feat_encoder = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
    
        self.hgnn = HGNNLayer(hidden_dim, hyper_num=hyper_num, n_layers=1)

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self.feat_encoder.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        x_emb = self.feat_encoder(x)
        x_emb = F.dropout(x_emb, p=self.dropout, training=self.training)
        # LightGCN: Layer 0 + Layer 1 + ... + Layer K
        embs_list = [x_emb]
        current_emb = x_emb
        
        for conv in self.convs:
            current_emb = conv(current_emb, edge_index)
            embs_list.append(current_emb)
            
        local_structure_emb = torch.stack(embs_list, dim=1).mean(dim=1)
        global_hyper_emb = self.hgnn(x_emb)
        final_emb = local_structure_emb + self.alpha * F.normalize(global_hyper_emb, p=2, dim=1)
        x_vision = F.dropout(F.relu(self.vision_head(final_emb)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(final_emb)), p=self.dropout, training=self.training)

        return final_emb, x_vision, x_text

class GCNII(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, alpha=0.1, theta=0.5):
        super(GCNII, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.alpha = alpha  
        self.theta = theta 
        
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCN2Conv(hidden_dim, alpha, theta, layer= _ + 1))
            
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for conv in self.convs: conv.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x_0 = x 
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x_0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text

from torch_geometric.nn import GATv2Conv

class GATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, dropout, att_dropout=0.1):
        super(GATv2, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lin_projs = nn.ModuleList()

        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=att_dropout, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.lin_projs.append(nn.Linear(in_dim, hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=att_dropout, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
            self.lin_projs.append(nn.Identity())


        if num_layers > 1:
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=att_dropout, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
            self.lin_projs.append(nn.Identity())
        self.out_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        for norm in self.norms: norm.reset_parameters()
        for lin in self.lin_projs: 
            if hasattr(lin, 'reset_parameters'): lin.reset_parameters()
        self.out_proj.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        x_in = x
        
        for i, conv in enumerate(self.convs):
            x_res = x_in
            x_in = conv(x_in, edge_index)
            if i == 0:
                x_res = self.lin_projs[i](x_res)
            x_in = x_in + x_res # ResNet
            x_in = self.norms[i](x_in)
            x_in = F.elu(x_in)
            x_in = F.dropout(x_in, p=self.dropout, training=self.training)
        x = self.out_proj(x_in) 
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text

class MHGAT(nn.Module):
    def __init__(self, v_dim, t_dim, hidden_dim, num_layers, dropout, heads=4):
        super(MHGAT, self).__init__()
        self.dropout = dropout
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.vis_projector = nn.Linear(v_dim, hidden_dim)
        self.txt_projector = nn.Linear(t_dim, hidden_dim)
        self.vis_gats = nn.ModuleList()
        self.txt_gats = nn.ModuleList()
        
        for _ in range(num_layers):
            self.vis_gats.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            )
            self.txt_gats.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            )
            
        self.modal_att = nn.Linear(hidden_dim * 2, 1)
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self.vis_projector.reset_parameters()
        self.txt_projector.reset_parameters()
        self.modal_att.reset_parameters()
        self.vision_head.reset_parameters()
        self.text_head.reset_parameters()
        for c in self.vis_gats: c.reset_parameters()
        for c in self.txt_gats: c.reset_parameters()

    def forward(self, x, edge_index, use_subgraph=False):
        x_v_raw = x[:, :self.v_dim]
        x_t_raw = x[:, self.v_dim:]
        h_v = F.relu(self.vis_projector(x_v_raw)) 
        h_v = F.dropout(h_v, p=self.dropout, training=self.training)
        
        h_t = F.relu(self.txt_projector(x_t_raw))
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        
        for i in range(len(self.vis_gats)):
            # --- Visual Stream ---
            h_v_res = h_v 
            h_v = self.vis_gats[i](h_v, edge_index)
            h_v = F.elu(h_v) + h_v_res # Residual
            h_v = F.dropout(h_v, p=self.dropout, training=self.training)
            
            # --- Text Stream ---
            h_t_res = h_t
            h_t = self.txt_gats[i](h_t, edge_index)
            h_t = F.elu(h_t) + h_t_res # Residual
            h_t = F.dropout(h_t, p=self.dropout, training=self.training)

        concat_h = torch.cat([h_v, h_t], dim=1) # [N, 2*hidden]
        att_score = self.modal_att(concat_h)    # [N, 1]
        beta = torch.sigmoid(att_score)         # Visual weights    
        final_emb = beta * h_v + (1 - beta) * h_t
        x_vision = F.dropout(F.relu(self.vision_head(h_v)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(h_t)), p=self.dropout, training=self.training)

        return final_emb, x_vision, x_text