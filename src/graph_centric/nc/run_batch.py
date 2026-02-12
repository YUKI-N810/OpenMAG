import dgl
import torch
import os
import time
import logging
from torch_geometric.data import Data
import numpy as np

log = logging.getLogger(__name__)
import pandas as pd
from model.models import GCN, GraphSAGE, GAT, MLP, GIN, ChebNet, LGMRec, GCNII, GATv2, MHGAT
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
from model.UniGraph2 import UniGraph2
from model.DMGC import DMGC
from model.DGF import DGF
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.amp import autocast, GradScaler
from utils.pretrained_model import TrainableBackbone
from transformers import AutoTokenizer

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
                # Pre-transform and store Tensor in RAM for faster training
                img_tensor = img_transforms(img)
                IMAGE_CACHE[p] = img_tensor
                success += 1
            except:
                pass
    print(f">>> Caching finished. {success}/{len(image_paths)} images loaded.")

class EndToEndModel(nn.Module):
    def __init__(self, gnn_model, vision_backbone, text_backbone, modality):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.gnn = gnn_model
        self.modality = modality
        
    def forward(self, x_txt, raw_images, input_ids, attention_mask, edge_index):
        if self.vision_backbone is not None and raw_images is not None:
            v_emb = self.vision_backbone(raw_images)
        elif self.modality == 'multimodal' or self.modality == 'visual':
            v_emb = x_txt[:, :768] # Assumption
        else:
            v_emb = torch.zeros(1, 768).to(edge_index.device) # Dummy

        if self.text_backbone is not None and input_ids is not None:
            t_emb = self.text_backbone(input_ids, attention_mask)
        else:
            t_emb = x_txt 

        if self.modality == 'text':
            v_emb = torch.zeros_like(v_emb)
        if self.modality == 'visual':
            t_emb = torch.zeros_like(t_emb)
        
        x_fused = torch.cat([v_emb, t_emb], dim=1)
        gnn_output = self.gnn(x_fused, edge_index)
        
        # Return results 
        if len(gnn_output) == 4:
            return gnn_output[0], gnn_output[1], gnn_output[2], gnn_output[3], v_emb, t_emb
        else:
            return gnn_output[0], gnn_output[1], gnn_output[2], v_emb, t_emb

def split_graph(nodes_num, train_ratio=0.6, val_ratio=0.2, fewshots=False, label=None):
    # Split dataset
    indices = np.random.permutation(nodes_num)
    if not fewshots:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_mask = torch.zeros(nodes_num, dtype=torch.bool)
        val_mask = torch.zeros(nodes_num, dtype=torch.bool)
        test_mask = torch.zeros(nodes_num, dtype=torch.bool)
        # indices = torch.Tensor(indices).to(torch.long)
        indices = torch.from_numpy(indices).to(torch.long)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
    return train_mask, val_mask, test_mask

def load_data(graph_path, v_emb_path, t_emb_path, image_path, train_ratio, val_ratio, fewshots=False, self_loop=True, undirected=True, modality="multimodal", text_noise_rate=0.0, image_missing_rate=0.0, edge_missing_rate=0.0, label_noise_rate=0.0, config=None):
    graph = dgl.load_graphs(graph_path)[0][0]
    
    # Edge robustness
    if edge_missing_rate > 0.0:
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        add_num = int(num_edges * edge_missing_rate)
        if add_num > 0:
            src_noise = torch.randint(0, num_nodes, (add_num,))
            dst_noise = torch.randint(0, num_nodes, (add_num,))
            graph.add_edges(src_noise, dst_noise)
            print(f">>> [Robustness] Added {add_num} NOISE edges (Active Corruption Rate: {edge_missing_rate})")
            
    # Self-loops, Undirected graph
    if undirected:
        graph = dgl.add_reverse_edges(graph)
    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    # Edges
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    is_visual_trainable = (v_emb_path == "TRAINABLE_MODE")
    is_text_trainable = (t_emb_path == "TRAINABLE_MODE")
    is_trainable = is_visual_trainable or is_text_trainable
    
    tokenized_data = {}
    
    if is_text_trainable:
        print(">>> [Data Loading] Mode: End-to-End Trainable (Raw Text)")
        try:
            base_dir = os.path.dirname(os.path.dirname(image_path))
            dataset_name = os.path.basename(base_dir) # e.g. Grocery
            csv_path = os.path.join(base_dir, f"{dataset_name}.csv")
            
            if not os.path.exists(csv_path):
                 # fallback attempt
                 csv_path = os.path.join(base_dir, f"{dataset_name.lower()}.csv")
                 
            print(f"Reading raw text from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Column name might vary by dataset (ref embedding_generator.py)
            text_col = 'text' if dataset_name in ["Movies", "Toys", "Grocery"] else 'caption'
            if text_col not in df.columns:
                 # fallback to first string column
                 text_col = df.select_dtypes(include=['object']).columns[0]
            
            raw_texts = df[text_col].astype(str).tolist()
            
            print(">>> [Data Loading] Pre-tokenizing text to accelerate training...")
            local_path = f"/root/autodl-tmp/hf_cache/{config.text_encoder}"
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            
            # max_length=128 is usually enough for e-commerce reviews, too long slows down
            encodings = tokenizer(raw_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            tokenized_data['input_ids'] = encodings['input_ids']
            tokenized_data['attention_mask'] = encodings['attention_mask']
            
            t_x = None
            t_dim = 768 # BERT base default
        except Exception as e:
            print(f"Error loading raw text: {e}")
            raise e
    else:
        t_x = torch.from_numpy(np.load(t_emb_path)).to(torch.float32)
        
        # Text robustness
        if text_noise_rate > 0.0:
            num_nodes = t_x.size(0)
            noise_num = int(num_nodes * text_noise_rate)
            if noise_num > 0:
                target_indices = torch.randperm(num_nodes)[:noise_num]
                perm_indices = torch.randperm(num_nodes)[:noise_num]
                t_x[target_indices] = t_x[perm_indices]
                print(f">>> [Robustness] Shuffled (Mismatched) Text Features for {noise_num} nodes (Rate: {text_noise_rate})")
        t_dim = t_x.size(1)

    image_paths = []
    if is_visual_trainable:
        print(">>> [Data Loading] Mode: End-to-End Trainable (Raw Images)")
        v_x = None 
        v_dim = 768 # ViT/Projected default
        
        num_nodes = graph.num_nodes()
        for i in range(num_nodes):
            p = os.path.join(image_path, f"{i}.jpg")
            if not os.path.exists(p):
                p = os.path.join(image_path, f"{i}.png")
            if not os.path.exists(p):
                 p = None 
            image_paths.append(p)
    else:
        # Load Frozen .npy
        v_x = torch.from_numpy(np.load(v_emb_path)).to(torch.float32)
        
        # Image robustness
        if image_missing_rate > 0.0:
            num_nodes = v_x.size(0)
            miss_num = int(num_nodes * image_missing_rate)
            if miss_num > 0:
                target_indices = torch.randperm(num_nodes)[:miss_num]
                perm_indices = torch.randperm(num_nodes)[:miss_num]
                # Shuffle
                v_x[target_indices] = v_x[perm_indices]
                print(f">>> [Robustness] Shuffled (Mismatched) Visual features for {miss_num} nodes (Rate: {image_missing_rate})")
        v_dim = v_x.size(1)
    
    if is_trainable:
        # Case A: Only Visual Trainable -> x = t_x (Frozen Text)
        if is_visual_trainable and not is_text_trainable:
            x = t_x
        # Case B: Only Text Trainable -> x = v_x (Frozen Visual)
        elif is_text_trainable and not is_visual_trainable:
            x = v_x
        # Case C: Both Trainable -> x use dummy or empty to prevent Loader error
        else:
            x = torch.zeros(graph.num_nodes(), 1) # Dummy placeholder
    else:
        # All Frozen
        if modality == 'text':
            v_x = torch.zeros_like(v_x)
        if modality == 'visual':
            t_x = torch.zeros_like(t_x)
            
        v_x = torch.nan_to_num(v_x, nan=0.0)
        t_x = torch.nan_to_num(t_x, nan=0.0)
        x = torch.cat([v_x, t_x], dim=1)

    print("Input Data Statistics:")
    print(f"Nodes: {graph.num_nodes()}")
    
    y = graph.ndata["label"]
    if "train_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"].to(torch.bool)
        val_mask = graph.ndata["val_mask"].to(torch.bool)
        test_mask = graph.ndata["test_mask"].to(torch.bool)
    else:
        train_mask, val_mask, test_mask = split_graph(graph.num_nodes(), train_ratio, val_ratio, fewshots, y)
    
    # Label Noise
    if label_noise_rate > 0.0:
        train_node_indices = torch.where(train_mask)[0]
        num_train = len(train_node_indices)
        num_noise = int(num_train * label_noise_rate)
        
        if num_noise > 0:
            perm = torch.randperm(num_train)[:num_noise]
            noise_indices = train_node_indices[perm]    
            num_classes = y.max().item() + 1
            random_labels = torch.randint(0, num_classes, (num_noise,), dtype=y.dtype)
            y[noise_indices] = random_labels
            
            print(f">>> [Robustness] Injected Label Noise to {num_noise} training nodes (Rate: {label_noise_rate})")
        
    data = Data(
        x=x,
        v_dim=v_dim,
        t_dim=t_dim,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    data.is_trainable = is_trainable
    data.aux_info = {}
    
    if is_visual_trainable:
        data.aux_info['image_paths'] = image_paths
        data.aux_info['visual_trainable'] = True
    else:
        data.aux_info['visual_trainable'] = False
        
    if is_text_trainable:
        # [Optimization] Store Tensor in aux_info
        data.aux_info['input_ids'] = tokenized_data['input_ids']
        data.aux_info['attention_mask'] = tokenized_data['attention_mask']
        data.aux_info['text_trainable'] = True
    else:
        data.aux_info['text_trainable'] = False
        
    return data

    #return Data(x=x, v_dim=v_x.size(1), t_dim=t_x.size(1), edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

class NodeClassifier(nn.Module):
    # Classification Head
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)

class GNNModel(nn.Module):
    # Embedding + Classification Head
    def __init__(self, encoder, classifier, dim_hidden, dim_v, dim_t):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.decoder_v = nn.Linear(dim_hidden, dim_v)
        self.decoder_t = nn.Linear(dim_hidden, dim_t)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
        self.decoder_v.reset_parameters()
        self.decoder_t.reset_parameters()

    def forward(self, x, edge_index, n_id=None):
        loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        kwargs = {}
        if isinstance(self.encoder, (Net, MGAT)):
            kwargs['n_id'] = n_id
            kwargs['use_subgraph'] = True
        if self.training and hasattr(self.encoder, "can_return_loss") and self.encoder.can_return_loss:
            x, x_v, x_t, loss = self.encoder(x, edge_index, **kwargs)
        else:
            x, x_v, x_t = self.encoder(x, edge_index, **kwargs)
        out = self.classifier(x)
        out_v = self.decoder_v(x_v)
        out_t = self.decoder_t(x_t)
        if self.training:
            return out, out_v, out_t, loss
        return out, out_v, out_t

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def calculate_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    Calculate Macro-F1 Score
    Args:
        preds: Predicted labels [N]
        labels: True labels [N]
        num_classes: Number of classes
    Returns:
        macro_f1: Average F1 score
    """
    # Initialize confusion matrix [num_classes, num_classes]
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    
    # Fill confusion matrix
    for p, l in zip(preds, labels):
        confusion_matrix[l, p] += 1
    
    # Calculate statistics for each class
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        # True Positive
        tp = confusion_matrix[cls, cls]
        
        # False Positive (Other classes predicted as this class)
        fp = confusion_matrix[:, cls].sum() - tp
        
        # False Negative (This class predicted as other classes)
        fn = confusion_matrix[cls, :].sum() - tp
        
        # Avoid division by zero
        precision[cls] = tp / (tp + fp + 1e-10)
        recall[cls] = tp / (tp + fn + 1e-10)
        
        # F1 Calculation
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls] + 1e-10)
    
    # Macro-F1 (Average across classes)
    macro_f1 = f1.mean().item()
    return macro_f1

def infoNCE_loss(out, orig_features, tau=0.07):
    """
    InfoNCE Loss Implementation
    - out: Feature embeddings output by model (batch_size, emb_dim)
    - orig_features: Original features (batch_size, feat_dim)
    - tau: Temperature coefficient
    """
    # 1. Feature Normalization
    out_norm = F.normalize(out, p=2, dim=1)
    orig_norm = F.normalize(orig_features, p=2, dim=1)
    
    # 2. Compute similarity matrix
    sim_matrix = torch.mm(out_norm, orig_norm.t()) / tau  # [batch_size, batch_size]
    
    # 3. Create labels (diagonal elements are positive samples)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # 4. Use Cross Entropy Loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

@torch.no_grad()
def evaluate(model, data, mask, config, num_classes):
    model.eval()
    
    is_trainable_mode = getattr(data, 'is_trainable', False)
    
    if is_trainable_mode:
        batch_size = 16
        neighbors = [5] * config.model.num_layers
    else:
        batch_size = config.task.batch_size
        neighbors = [config.task.num_neighbors] * config.model.num_layers
    # Create NeighborLoader for subgraph sampling
    loader = NeighborLoader(
        data,
        num_neighbors=neighbors,  # Number of neighbors sampled per layer
        input_nodes=mask,  # Sample only for evaluation nodes
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    is_v_train = data.aux_info.get('visual_trainable', False)
    is_t_train = data.aux_info.get('text_trainable', False)

    for batch in loader:  # Iterate over subgraphs
        n_id = batch.n_id
        if is_trainable_mode:
            node_ids = batch.n_id.cpu().numpy()
            
            # Prepare Images
            raw_imgs = None
            if is_v_train and 'image_paths' in data.aux_info:
                all_paths = data.aux_info['image_paths']
                batch_paths = [all_paths[i] for i in node_ids]
                img_list = []
                for p in batch_paths:
                    if p in IMAGE_CACHE:
                        img = IMAGE_CACHE[p]
                    else:
                        try:
                            if p is None: raise Exception
                            img = Image.open(p).convert('RGB')
                            img = img_transforms(img)
                        except:
                            img = torch.zeros(3, 224, 224)
                    img_list.append(img)
                raw_imgs = torch.stack(img_list).to(config.device)
            
            # Prepare Texts
            batch_input_ids = None
            batch_att_mask = None
            if is_t_train:
                all_input_ids = data.aux_info['input_ids']
                all_att_mask = data.aux_info['attention_mask']
                
                batch_input_ids = all_input_ids[node_ids].to(config.device)
                batch_att_mask = all_att_mask[node_ids].to(config.device)
            
            # Forward: out, out_v, out_t, [loss], [v_emb], [t_emb]
            res = model(batch.x.to(config.device), raw_imgs, batch_input_ids, batch_att_mask, batch.edge_index.to(config.device), n_id=n_id)
            out = res[0]
        else:
            out, out_v, out_t = model(batch.x.to(config.device), batch.edge_index.to(config.device), n_id=n_id)
            
        pred = out.argmax(dim=1)
        # correct += (pred[batch.train_mask] == batch.y[batch.train_mask]).sum().item()
        # total += batch.train_mask.sum().item()
        pred = pred[:batch.batch_size]
        y_true = batch.y[:batch.batch_size]

        all_preds.append(pred.cpu())
        all_labels.append(y_true.cpu())

        correct += (pred == y_true).sum().item()
        total += y_true.size(0)

    accuracy = correct / total if total > 0 else 0.0
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = calculate_f1(all_preds, all_labels, num_classes)


    return f1, accuracy

def train_and_eval(config, model, data, run_id=0):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    model = model.to(config.device)

    is_trainable_mode = getattr(data, 'is_trainable', False)

    if is_trainable_mode:
        param_groups = [
            {'params': model.gnn.parameters(), 'lr': config.task.lr, 'weight_decay': config.task.weight_decay}
        ]
        
        if model.vision_backbone is not None:
             param_groups.append({'params': model.vision_backbone.parameters(), 'lr': 1e-5})
             
        if model.text_backbone is not None:
             param_groups.append({'params': model.text_backbone.parameters(), 'lr': 1e-5})

        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.task.lr, weight_decay=config.task.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    data.x = data.x.cpu()
    
    if is_trainable_mode:
        batch_size = 4
        # Enforce limits on layers and neighbors to prevent VRAM explosion
        neighbors = [5] * config.model.num_layers
        accum_iter = 16
    else:
        batch_size = config.task.batch_size
        neighbors = [config.task.num_neighbors] * config.model.num_layers
        accum_iter = 1
        
    train_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        input_nodes=torch.where(data.train_mask)[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    best_val_acc = 0
    best_val_f1 = 0
    final_test_acc = 0
    final_test_f1 = 0
    
    is_v_train = data.aux_info.get('visual_trainable', False)
    is_t_train = data.aux_info.get('text_trainable', False)
    
    use_early_stop = config.task.early_stopping
    show_efficiency = config.task.show_efficiency
    
    if use_early_stop or show_efficiency:
        val_interval = 1
    else:
        val_interval = 10
    
    start_time = time.time()
    if 'cuda' in str(config.device):
        torch.cuda.reset_peak_memory_stats(config.device)
    
    patience = config.task.patience
    patience_counter = 0
    history = [] 
    actual_epochs = 0
    
    for epoch in tqdm(range(config.task.n_epochs), desc="Training"):
        actual_epochs = epoch + 1
        model.train()
        
        if is_trainable_mode:
            do_tuning = (epoch % 5 == 0)
            
            if model.vision_backbone:
                for param in model.vision_backbone.parameters():
                    param.requires_grad = do_tuning
            
            if model.text_backbone:
                for param in model.text_backbone.parameters():
                    param.requires_grad = do_tuning
            
            if epoch % 5 == 0:
                tqdm.write(f"Epoch {epoch}: Backbone Tuning {'ON' if do_tuning else 'OFF'}")
        
        total_loss = 0
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.task.n_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(config.device)
            n_id = batch.n_id
            batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=0.0, neginf=0.0)

            with autocast('cuda'):
                if is_trainable_mode:
                    node_ids = batch.n_id.cpu().numpy()
                    
                    raw_imgs = None
                    if is_v_train:
                        all_paths = data.aux_info['image_paths']
                        batch_paths = [all_paths[i] for i in node_ids]
                        
                        img_list = []
                        for p in batch_paths:
                            if p in IMAGE_CACHE:
                                img = IMAGE_CACHE[p]
                            else:
                                try:
                                    if p is None: raise Exception
                                    img = Image.open(p).convert('RGB')
                                    img = img_transforms(img)
                                except:
                                    img = torch.zeros(3, 224, 224)
                            img_list.append(img)
                        raw_imgs = torch.stack(img_list).to(config.device)
                    
                    batch_input_ids = None
                    batch_att_mask = None
                    if is_t_train:
                        all_input_ids = data.aux_info['input_ids']
                        all_att_mask = data.aux_info['attention_mask']
                        
                        batch_input_ids = all_input_ids[node_ids].to(config.device)
                        batch_att_mask = all_att_mask[node_ids].to(config.device)
                    res = model(batch.x, raw_imgs, batch_input_ids, batch_att_mask, batch.edge_index, n_id=n_id)
                    out, out_v, out_t = res[0], res[1], res[2]
                    
                    if len(res) >= 6:
                         loss_model = res[3]
                         target_visual = res[4]
                         target_text = res[5]
                    elif len(res) >= 5: # Compatibility with old logic
                         loss_model = res[3]
                         target_visual = res[4]
                         target_text = None
                    else:
                        loss_model = 0
                        target_visual = res[3]
                        target_text = None

                else:
                    # Frozen Mode
                    out, out_v, out_t, loss_model = model(batch.x, batch.edge_index, n_id=n_id)
                    target_visual = batch.x[:, :data.v_dim]
                    target_text = batch.x[:, data.v_dim:]
                
                # Loss Calculation
                current_batch_size = batch.batch_size
                loss_task = criterion(out[:current_batch_size], batch.y[:current_batch_size])
                loss_v = 0
                loss_t = 0
                
                if config.modality != 'text':
                     if is_v_train or not is_trainable_mode:
                        loss_v = infoNCE_loss(out_v[:current_batch_size], target_visual[:current_batch_size])

                if config.modality != 'visual':
                    if is_t_train:
                        # End2End Text
                        loss_t = infoNCE_loss(out_t[:current_batch_size], target_text[:current_batch_size])
                    elif not is_trainable_mode:
                        # Frozen
                        loss_t = infoNCE_loss(out_t[:current_batch_size], target_text[:current_batch_size])
                    else:
                        # Visual Trainable but Text Frozen
                        frozen_text = batch.x 
                        loss_t = infoNCE_loss(out_t[:current_batch_size], frozen_text[:current_batch_size])
                
                loss = loss_task + config.task.lambda_v * loss_v + config.task.lambda_t * loss_t + config.task.lambda_model * loss_model
                loss = loss / accum_iter

            scaler.scale(loss).backward()     
            total_loss += loss.item() * accum_iter * batch.train_mask.sum().item()
            # Gradient Accumulation Step
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()   
            if is_trainable_mode:
                del raw_imgs, batch_input_ids, batch_att_mask, res, out, out_v, out_t, target_visual, target_text  
        torch.cuda.empty_cache()

        loss = total_loss / len(train_loader)
        
        if epoch % val_interval == 0 or epoch == config.task.n_epochs - 1:
            val_f1, val_acc = evaluate(model, data, data.val_mask, config, data.y.max().item() + 1)
            test_f1, test_acc = evaluate(model, data, data.test_mask, config, data.y.max().item() + 1)
            
            current_time = time.time() - start_time
            history.append({
                'epoch': epoch,
                'time': current_time,
                'val_acc': val_acc,
                'test_acc': test_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                final_test_f1 = test_f1
                patience_counter = 0 
            else:
                patience_counter += 1
            
            if show_efficiency or use_early_stop:
                tqdm.write(f"[Run {run_id+1}] Epoch {epoch:03d} | Time: {current_time:.1f}s | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
            else:
                tqdm.write(f"[Run {run_id+1}] Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

            if use_early_stop and patience_counter >= patience:
                tqdm.write(f">>> Early Stopping triggered at Epoch {epoch}")
                break
    
    end_time = time.time()
    train_time = end_time - start_time
    
    peak_memory = 0.0
    if 'cuda' in str(config.device):
        #peak_memory = torch.cuda.max_memory_allocated(config.device) / 1024 / 1024
        peak_memory = torch.cuda.max_memory_reserved(config.device) / 1024 / 1024

    return best_val_acc, best_val_f1, final_test_acc, final_test_f1, train_time, peak_memory, actual_epochs, history
    #return best_val_acc, best_val_f1, final_test_acc, final_test_f1

def run_nc(config):
    print(config)
    np.random.seed(config.seed)
    data = load_data(
        config.dataset.graph_path, config.dataset.v_emb_path, config.dataset.t_emb_path, 
        config.dataset.image_path, config.dataset.train_ratio, config.dataset.val_ratio, 
        config.task.fewshots, config.task.self_loop, config.task.undirected, config.modality,
        text_noise_rate=config.task.text_noise_rate, 
        image_missing_rate=config.task.image_missing_rate, 
        edge_missing_rate=config.task.edge_missing_rate,
        label_noise_rate=config.task.label_noise_rate,
        config=config
    ).to(config.device)
    
    if getattr(data, 'is_trainable', False):
        if data.aux_info.get('visual_trainable', False) and 'image_paths' in data.aux_info:
            preload_images(data.aux_info['image_paths'])
            
    num_classes = data.y.max().item() + 1
    print("=== Num Classes ===")
    print("num_classes:", num_classes)
    in_dim = data.v_dim + data.t_dim
    if config.model.name =="MLP":
        encoder = MLP(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GAT":
        encoder = GAT(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, heads=config.model.heads, dropout=config.model.dropout, att_dropout=config.model.att_dropout)
    elif config.model.name =="GCN":
        encoder = GCN(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GIN":
        encoder = GIN(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GraphSAGE":
        encoder = GraphSAGE(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="ChebNet":
        encoder = ChebNet(in_dim=in_dim, hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, K=config.model.K, dropout=config.model.dropout)
    elif config.model.name == "RevGAT":
        encoder = RevGAT(in_feats=in_dim,n_hidden=config.model.hidden_dim, n_layers=config.model.num_layers, n_heads=config.model.heads, activation=F.relu, dropout=config.model.dropout)
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
            t_feat_dim=data.x.size(1)-768,
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
    classifier = NodeClassifier(in_dim=config.model.hidden_dim, num_classes=num_classes)
    gnn_model = GNNModel(encoder, classifier, config.model.hidden_dim, data.v_dim, data.t_dim)
    
    if getattr(data, 'is_trainable', False):
        print(f"=== Wrapper: End-to-End Trainable Mode ===")
        
        # [Optimization] Only train the last 2 layers, significantly faster
        unfrozen_layers = 2
        backbone_v = None
        if data.aux_info.get('visual_trainable', False):
             print(f" -> Initializing Visual Backbone: {config.visual_encoder}")
             backbone_v = TrainableBackbone(config.visual_encoder, target_dim=data.v_dim, num_unfrozen_layers=unfrozen_layers)
        backbone_t = None
        if data.aux_info.get('text_trainable', False):
             print(f" -> Initializing Text Backbone: {config.text_encoder}")
             backbone_t = TrainableBackbone(config.text_encoder, target_dim=data.t_dim, num_unfrozen_layers=unfrozen_layers)
             
        model = EndToEndModel(gnn_model, backbone_v, backbone_t, modality=config.modality)
    else:
        print("=== Wrapper: Frozen Features ===")
        model = gnn_model

    accs = []
    f1s = []
    
    times = []
    mems = []
    epochs_list = []
    all_histories = []
    
    '''for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        best_val_acc, best_val_f1, final_test_acc, final_test_f1 = train_and_eval(
            config, model, data, run
        )
        accs.append(final_test_acc)
        f1s.append(final_test_f1)'''
    
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        
        best_val_acc, best_val_f1, final_test_acc, final_test_f1, t_time, p_mem, n_epochs, history = train_and_eval(
            config, model, data, run
        )
        accs.append(final_test_acc)
        f1s.append(final_test_f1)
        
        times.append(t_time)
        mems.append(p_mem)
        epochs_list.append(n_epochs)
        all_histories.append(history)

    acc_mean = np.mean(accs) * 100
    acc_std  = np.std(accs) * 100

    f1_mean = np.mean(f1s)
    f1_std  = np.std(f1s)

    log.info("="*60)
    log.info("[NC Task] Final Results")
    log.info(f"Model: {config.model.name} | Dataset: {config.dataset.name}")
    log.info(f"Average Test Accuracy over {config.task.n_runs} runs: {acc_mean:.2f} ± {acc_std:.2f}")
    log.info(f"Average Test F1 over {config.task.n_runs} runs: {f1_mean:.4f} ± {f1_std:.4f}")
    log.info(f"Individual Accuracies: {[round(a * 100, 2) for a in accs]}")
    log.info("="*60)
    
    print(
        f"Average Test Accuracy over {config.task.n_runs} runs: "
        f"{acc_mean:.2f} ± {acc_std:.2f}"
    )

    print(
        f"Average Test F1 over {config.task.n_runs} runs: "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )
    
    if config.task.show_efficiency:
        print("\n=== Efficiency Analysis (Experimental Q9/Q10) ===")
        print(f"Model: {config.model.name}")
        print(f"Peak Memory (MB): {np.mean(mems):.2f} ± {np.std(mems):.2f}")
        print(f"Training Time (s): {np.mean(times):.2f} ± {np.std(times):.2f}")
        print(f"Training Epochs: {np.mean(epochs_list):.1f} ± {np.std(epochs_list):.1f}")
        
        # --- Data Aggregation Logic Start ---
        if len(all_histories) > 0:
            # 1. Find min epochs among runs to ensure alignment (Early Stop varies length)
            min_len = min([len(h) for h in all_histories])
            
            # 2. Extract data to build matrix [n_runs, min_len]
            # Time here is cumulative time at corresponding epoch per run
            times_matrix = np.zeros((config.task.n_runs, min_len))
            accs_matrix = np.zeros((config.task.n_runs, min_len))
            epochs_arr = np.arange(min_len) # 0, 1, 2...
            
            for r_idx, history in enumerate(all_histories):
                for e_idx in range(min_len):
                    times_matrix[r_idx, e_idx] = history[e_idx]['time']
                    accs_matrix[r_idx, e_idx] = history[e_idx]['test_acc']
            
            # 3. Calculate Mean and Std
            mean_time = np.mean(times_matrix, axis=0) # Mean time per epoch
            mean_acc = np.mean(accs_matrix, axis=0)   # Mean Acc per epoch
            std_acc = np.std(accs_matrix, axis=0)     # Acc Std per epoch
            
            print("\n--- Plotting Data (Mean & Std) ---")
            print("Epoch, Mean_Time, Mean_Acc, Std_Acc")
            for i in range(min_len):
                # Print each row: Epoch, Mean Time, Mean Acc, Std Acc
                print(f"{epochs_arr[i]}, {mean_time[i]:.2f}, {mean_acc[i]:.4f}, {std_acc[i]:.4f}")
        else:
            print("No history data found.")
            
        print("=================================================")