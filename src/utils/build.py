# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
import dgl
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd

# =======================
# ====== CONFIG =========
# =======================
CONFIG = {
    'root': '/root/autodl-tmp/MMAG/SemArt',
    'dataset': 'SemArt',
    'text': 'text_features/SemArt_text_xlm_roberta_base_768d.npy',
    'image': 'image_features/SemArt_image_clip_vit_large_patch14_768d.npy',
    'fusion': 'mean',          # mean / concat / weighted / none
    'weights': None,           
    'pca_dim': None,           
    'normalize': True,         
    'k': 10,                   
    'mutual': True,            
    'symmetrize': True,        
    'metric': 'cosine',        
    'n_jobs': 8,               
    'item_list': None,         
    'save_item_map': 'SemArt_item2idx.json',  
    'use_label': False          # 是否加载标签
}

# =======================
# ====== FUNCTIONS ======
# =======================
def load_embedding(path):
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def build_embeddings(cfg):
    feats = []
    item_ids = None

    # 加载 item_list
    if cfg['item_list'] is not None:
        if os.path.exists(cfg['item_list']):
            with open(cfg['item_list'], 'r', encoding='utf-8') as f:
                item_ids = [line.strip().split()[0] for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"item_list not found: {cfg['item_list']}")

    # 加载 embedding
    if cfg['text'] is not None:
        text_path = os.path.join(cfg['root'], cfg['text']) if not os.path.isabs(cfg['text']) else cfg['text']
        print("Loading text embedding:", text_path)
        feats.append(load_embedding(text_path))
    if cfg['image'] is not None:
        image_path = os.path.join(cfg['root'], cfg['image']) if not os.path.isabs(cfg['image']) else cfg['image']
        print("Loading image embedding:", image_path)
        feats.append(load_embedding(image_path))

    if len(feats) == 0:
        raise ValueError("至少指定一个模态 embedding：text 或 image")

    # 检查行数一致
    n_rows = feats[0].shape[0]
    for f in feats:
        if f.shape[0] != n_rows:
            raise ValueError(f"Embedding row mismatch: expected {n_rows}, got {f.shape[0]}")

    # 融合
    if cfg['fusion'] == 'mean':
        fused = np.mean(np.stack(feats, axis=0), axis=0)
    elif cfg['fusion'] == 'concat':
        fused = np.concatenate(feats, axis=1)
    elif cfg['fusion'] == 'weighted':
        if cfg['weights'] is None:
            raise ValueError("fusion=weighted 时必须指定 weights")
        ws = np.array(cfg['weights'], dtype=np.float32)
        if len(ws) != len(feats):
            raise ValueError("权重数量必须与模态数一致")
        stacked = np.stack(feats, axis=0)
        fused = np.tensordot(ws, stacked, axes=(0,0))
        fused = np.array(fused)
    elif cfg['fusion'] == 'none':
        assert len(feats) == 1, "fusion=none 只能在单模态时使用"
        fused = feats[0]
    else:
        raise ValueError("未知 fusion 方法")

    # PCA
    if cfg['pca_dim'] is not None:
        print(f"Applying PCA -> {cfg['pca_dim']} dim")
        pca = PCA(n_components=cfg['pca_dim'])
        fused = pca.fit_transform(fused)

    # L2归一化
    if cfg['normalize']:
        fused = l2_normalize(fused)

    return fused, item_ids

def build_knn_graph(feat, cfg):
    n = feat.shape[0]
    nn = NearestNeighbors(n_neighbors=cfg['k']+1, metric=cfg['metric'], n_jobs=cfg['n_jobs'])
    nn.fit(feat)
    distances, indices = nn.kneighbors(feat)

    sims = 1.0 - distances if cfg['metric'] == 'cosine' else -distances

    src_list, dst_list, w_list = [], [], []
    for i in range(n):
        neighs = indices[i]
        neigh_sims = sims[i]
        for idx_pos in range(1, len(neighs)):
            j = int(neighs[idx_pos])
            src_list.append(i)
            dst_list.append(j)
            w_list.append(float(neigh_sims[idx_pos]))

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    if cfg['mutual']:
        pairs = set(zip(src.tolist(), dst.tolist()))
        mask = [1 if (b,a) in pairs else 0 for a,b in zip(src.tolist(), dst.tolist())]
        mask = np.array(mask, dtype=bool)
        src = src[mask]; dst = dst[mask]; w = w[mask]

    if cfg['symmetrize']:
        src_rev, dst_rev, w_rev = dst.copy(), src.copy(), w.copy()
        src = np.concatenate([src, src_rev], axis=0)
        dst = np.concatenate([dst, dst_rev], axis=0)
        w = np.concatenate([w, w_rev], axis=0)

    # 合并重复边，取最大权重
    pairs = np.stack([src,dst], axis=1)
    dtype = np.dtype([('a', np.int64), ('b', np.int64)])
    structured = pairs.view(dtype)
    uniq, idx_first, counts = np.unique(structured, return_index=True, return_counts=True)
    if uniq.shape[0] < pairs.shape[0]:
        agg = {}
        for s,d,wt in zip(src.tolist(), dst.tolist(), w.tolist()):
            key = (s,d)
            if key in agg:
                if wt > agg[key]:
                    agg[key] = wt
            else:
                agg[key] = wt
        src = np.array([k[0] for k in agg.keys()], dtype=np.int64)
        dst = np.array([k[1] for k in agg.keys()], dtype=np.int64)
        w = np.array(list(agg.values()), dtype=np.float32)

    return src, dst, w

def load_labels(cfg, item_ids):
    """尝试加载标签，如果 item_info.csv 没有 label 列就自动生成"""
    if not cfg.get('use_label', True):
        print("标签加载关闭，跳过 label.")
        return None

    csv_path = os.path.join(cfg['root'], 'item_info.csv')
    if not os.path.exists(csv_path):
        print("未找到标签文件 item_info.csv，跳过 label.")
        return None

    df = pd.read_csv(csv_path)

    # 如果没有 label 列，用 tag 生成
    if 'label' not in df.columns:
        if 'tag' not in df.columns:
            raise ValueError("item_info.csv 缺少 'tag' 列，无法生成 label")
        unique_tags = sorted(df['tag'].dropna().unique())
        tag2label = {t:i for i,t in enumerate(unique_tags)}
        df['label'] = df['tag'].map(tag2label)
        df['label'] = df['label'].fillna(-1).astype(int)
        df.to_csv(csv_path, index=False)
        print(f"新增 label 列并保存到 {csv_path}")

    item_to_label = {str(row['item_id']): row['label'] for _, row in df.iterrows()}
    labels = [item_to_label.get(item_id, -1) for item_id in item_ids]
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"加载 {len(labels)} 个节点标签，缺失的标记为 -1")
    return labels

def build_attribute_edges(csv_path, column):
    df = pd.read_csv(csv_path)

    value2idx = {}
    src, dst = [], []

    for idx, val in enumerate(df[column].astype(str)):
        if val not in value2idx:
            value2idx[val] = []
        value2idx[val].append(idx)

    for _, idxs in value2idx.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            for j in idxs:
                if i != j:
                    src.append(i)
                    dst.append(j)

    return np.array(src), np.array(dst)

def main():
    cfg = CONFIG

    feat, item_ids = build_embeddings(cfg)
    n_items = feat.shape[0]
    print("Embedding shape:", feat.shape)

    if item_ids is None:
        item_ids = [str(i) for i in range(n_items)]

    
    
    knn_src, knn_dst, knn_w = build_knn_graph(feat, cfg)
    print(f"Built edges: {len(knn_src)}")
    csv_path = os.path.join(cfg['root'], 'SemArt.csv')
    author_src, author_dst = build_attribute_edges(csv_path, 'AUTHOR')


    author_w = torch.ones(len(author_src)) * 1.0




    all_src = np.concatenate([knn_src, author_src], axis=0)
    all_dst = np.concatenate([knn_dst, author_dst], axis=0)
    all_w   = np.concatenate([knn_w, author_w], axis=0)

    


    print(f"Added {len(author_src)} author edges")

    graph = dgl.graph(
        (all_src.tolist(), all_dst.tolist()),
        num_nodes=n_items
    )

    graph.edata['weight'] = torch.tensor(all_w, dtype=torch.float32)


    graph = dgl.add_self_loop(graph)
    print("Added self-loops to all nodes.")

    # 尝试加载标签
    labels = load_labels(cfg, item_ids)
    if labels is not None:
        graph.ndata['label'] = labels

    save_path = os.path.join(cfg['root'], f"{cfg['dataset']}Graph.pt")
    dgl.save_graphs(save_path, [graph])
    print("Saved graph to:", save_path)
    print(graph)

    # 保存 item2idx
    if cfg['save_item_map'] is not None:
        item2idx = {item_ids[i]: int(i) for i in range(len(item_ids))}
        save_map_path = os.path.join(cfg['root'], cfg['save_item_map'])
        with open(save_map_path, 'w', encoding='utf-8') as f:
            json.dump(item2idx, f, ensure_ascii=False, indent=2)
        print("Saved item2idx mapping to:", save_map_path)

if __name__ == '__main__':
    main()
