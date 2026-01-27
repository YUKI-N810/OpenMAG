# src/model/GSMN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.gsmn.graph_model import VisualGraph, TextualGraph

class GSMN(nn.Module):
    """
    GSMN Adapter compatible with gnn_trainer.
    The initialization parameter interface remains consistent with other GNN models:
    - v_feat_dim: Image embedding dimension
    - t_feat_dim: Text embedding dimension
    - num_nodes: Number of nodes (optional, for reference only)
    - num_layers: Number of layers (might not be used internally)
    - hidden_dim: Internal hidden dim (projection + Graph internal conv)
    - out_dim: Output dim
    - dropout: Dropout probability

    Forward Output:
    - out: [N, out_dim*2] concatenated embedding
    - out_v: [N, hidden_dim] visual node-level embedding
    - out_t: [N, hidden_dim] textual node-level embedding
    """
    def __init__(self, v_feat_dim, t_feat_dim, num_nodes, num_layers, hidden_dim, out_dim, dropout):
        super(GSMN, self).__init__()
        self.v_feat_dim = v_feat_dim
        self.t_feat_dim = t_feat_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Projection layer: project raw features to hidden_dim
        self.img_proj = nn.Linear(v_feat_dim, hidden_dim)
        self.txt_proj = nn.Linear(t_feat_dim, hidden_dim)

        # Graph modules (retain original paper structure)
        try:
            # Here feat_dim, hid_dim, out_dim correspond to the dimensions of the internal linear layers of the Graph
            self.visual_graph = VisualGraph(feat_dim=hidden_dim, hid_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
            self.textual_graph = TextualGraph(feat_dim=hidden_dim, hid_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
            self.use_graph_modules = True
        except Exception as e:
            print("[GSMN] Warning: cannot init VisualGraph/TextualGraph - fallback to projection only.", e)
            self.visual_graph = None
            self.textual_graph = None
            self.use_graph_modules = False

        # Modality heads
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.zeros_(self.img_proj.bias)
        nn.init.xavier_uniform_(self.txt_proj.weight)
        nn.init.zeros_(self.txt_proj.bias)

        if self.visual_graph is not None:
            try:
                self.visual_graph.reset_parameters()
            except Exception:
                pass
        if self.textual_graph is not None:
            try:
                self.textual_graph.reset_parameters()
            except Exception:
                pass

        nn.init.xavier_uniform_(self.vision_head.weight)
        nn.init.zeros_(self.vision_head.bias)
        nn.init.xavier_uniform_(self.text_head.weight)
        nn.init.zeros_(self.text_head.bias)

    def forward(self, features, edge_index=None, bbox=None, depends=None, cap_lens=None):
        """
        features: [num_nodes, t_dim + v_dim]
        Output:
        - out: concatenated node embedding [N, out_dim*2]
        - out_v: visual embedding [N, hidden_dim]
        - out_t: textual embedding [N, hidden_dim]
        """
        device = features.device
        total_dim = features.size(1)
        expected = self.t_feat_dim + self.v_feat_dim
        if total_dim != expected:
            raise ValueError(f"[GSMN] features dim mismatch: got {total_dim} expected {expected} (t:{self.t_feat_dim}+v:{self.v_feat_dim})")

        # Split text/image
        txt_feat = features[:, :self.t_feat_dim]
        img_feat = features[:, self.t_feat_dim:]

        # Project to hidden_dim
        img_h = F.relu(self.img_proj(img_feat))
        txt_h = F.relu(self.txt_proj(txt_feat))

        # Graph module processing
        if self.use_graph_modules and self.visual_graph is not None and self.textual_graph is not None:
            # Adapter style: treat each node as a single region/word
            img_in = img_h.unsqueeze(1)
            txt_in = txt_h.unsqueeze(1)

            try:
                v_rep = torch.tanh(self.visual_graph.out_1(img_in.view(-1, img_in.size(-1))))
                t_rep = torch.tanh(self.textual_graph.out_1(txt_in.view(-1, txt_in.size(-1))))
            except Exception:
                v_rep = img_h
                t_rep = txt_h
        else:
            v_rep = img_h
            t_rep = txt_h

        # Modality heads
        out_v = F.dropout(F.relu(self.vision_head(v_rep)), p=self.dropout, training=self.training)
        out_t = F.dropout(F.relu(self.text_head(t_rep)), p=self.dropout, training=self.training)

        # Concatenated embedding
        out = torch.cat([out_v, out_t], dim=1)
        return out, out_v, out_t