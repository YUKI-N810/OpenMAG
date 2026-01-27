import os
import pickle
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import dgl
from PIL import Image
from transformers import AutoTokenizer
from multimodal_centric.G2Text.language_modelling import utils
from torch_geometric.data import Data

MAGB_DATASETS = ['Toys', 'Movies', 'Grocery', 'RedditS']
MM_GRAPH_DATASETS = ['ele-fashion', 'sports', 'cloth']

def load_datasets(dataset_name: str, data_root: str = '/root/autodl-tmp/MMAG', split_seed: int = 42):
    if not isinstance(dataset_name, str):
        raise ValueError('dataset_name must be a string')

    csv_path = os.path.join(data_root, dataset_name, f'{dataset_name}.csv')
    df = pd.read_csv(csv_path)

    if dataset_name == 'Flickr30k':
        valid_mask = df['description1'].notna() & (df['description1'].astype(str).str.strip() != '') & (df['description1'].astype(str).str.lower() != 'nan')
    elif dataset_name in MAGB_DATASETS:
        valid_mask = df['description'].notna() & (df['description'].astype(str).str.strip() != '') & (df['description'].astype(str).str.lower() != 'nan')
    elif dataset_name in MM_GRAPH_DATASETS:
        valid_mask = df['text'].notna() & (df['text'].astype(str).str.strip() != '') & (df['text'].astype(str).str.lower() != 'nan')
    
    valid_indices = df[valid_mask].index.to_numpy()
    
    print(f"Total nodes: {len(df)}")
    print(f"Valid nodes (non-empty description): {len(valid_indices)}")
    print(f"Dropped {len(df) - len(valid_indices)} empty/invalid nodes from training set.")


    rng = np.random.RandomState(split_seed)
    rng.shuffle(valid_indices)
    
    n = len(valid_indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_idx = valid_indices[:n_train].tolist()
    val_idx = valid_indices[n_train:n_train + n_val].tolist()
    test_idx = valid_indices[n_train + n_val:].tolist()

    return df, train_idx, val_idx, test_idx

def build_prompt(title, neighbor_texts=None, dataset_name=None):

    if dataset_name == 'Flickr30k':
        task_str = ("Generate a detailed description for the image based on the context."
                    "Focus on specific attributes like colors, clothing patterns, actions, and background objects.")
        input_key = "Context"       
        neighbor_key = "Related Caption" 
    elif dataset_name in MM_GRAPH_DATASETS:
        task_str = "Generate the product title based on the image and related products."
        input_key = "Product Context" 
        neighbor_key = "Related Product"
    else:
        task_str = "Generate a natural-language description of the product node."
        input_key = "Title"
        neighbor_key = "Related Product"

    prompt = (
        f"### Task\n"
        f"{task_str}\n\n"
    )
    
    if neighbor_texts:
        prompt += "".join([f"- {neighbor_key} {i+1} Description: {t}\n"
                           for i, t in enumerate(neighbor_texts)])
    
    prompt += "\n### Output\nResult:"
    return prompt


class Datasets(torch.utils.data.Dataset):
    def __init__(self, cfg, df, id_list, tokenizer):
        self.dataset_name = cfg.dataset.name
        self.data_root = cfg.dataset.data_root
        self.path = os.path.join(self.data_root, self.dataset_name)
        self.image_path = os.path.join(self.path, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')
        graph_path = os.path.join(self.path, f'{self.dataset_name}Graph.pt')
        self.graph, _ = dgl.load_graphs(graph_path)
        self.graph = self.graph[0]
        
        self.context = cfg.task.context

        self.neighbor_dict = {}
        for i in range(self.graph.num_nodes()):
            neighbors = self.graph.successors(i).tolist() 
            self.neighbor_dict[i] = neighbors
        
        self.decoder_only = cfg.task.decoder_only
        self.neighbor_mode = cfg.task.neighbor_mode
        
        self.max_text_neighbors = cfg.task.max_text_neighbors
        self.max_image_neighbors = cfg.task.max_image_neighbors
        self.position_type = cfg.task.position_type
        
        self.df = df

        self.id_list = id_list 
        self.tokenizer = tokenizer

        self.max_input_length = cfg.task.max_input_length
        self.max_output_length = cfg.task.max_output_length
        self.text_model = cfg.task.text_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(cfg.task.text_model, use_fast=False)
        self.text_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.visual_feature_extractor = utils.get_feature_extractor_for_model(cfg.task.visual_model)

        self.n_text_tokens = cfg.task.n_text_tokens
        self.n_visual_tokens = cfg.task.n_visual_tokens

       
    def __len__(self):
        return len(self.id_list)
    
    def get_section_images(self, n_id: int):
        try:
            image_id = str(self.df.iloc[n_id]['id']) 
        except KeyError:
            print("Error: Column 'id' not found in CSV.")
            return None
        file_name = os.path.join(self.image_path, f"{image_id}.jpg")
        if os.path.exists(file_name):
            try:
                with Image.open(file_name) as img:
                    section_image = utils.get_pixel_values_for_model(self.visual_feature_extractor, img)
                    return section_image
            except Exception as e:
                print(f"Error encountered: {e}")
                return None
        return None

    def __getitem__(self, index):
        global_index = self.id_list[index]
        if self.neighbor_mode == "embedding":
            return self.get_embedding_item(global_index)

        d = self.df.iloc[global_index]
        
        images = []
        image_positions = []
        neighbor_texts = [] 
        
        if self.dataset_name == 'Flickr30k':
            parts = [str(d[f'description{i}']) for i in range(2, 6)]
            target_texts = " ".join([p for p in parts if p.lower() != 'nan'])
        elif self.dataset_name in MM_GRAPH_DATASETS:
            target_texts = ""
        else:
            target_texts = str(d['title'])
        
        if self.context == 'text_only':
            neighbors = self.neighbor_dict.get(global_index, [])
            neighbors = neighbors[:self.max_text_neighbors]

            for n_id in neighbors:
                n_row = self.df.iloc[n_id]
                if self.dataset_name == 'Flickr30k':
                    n_parts = [str(n_row[f'description{i}']) for i in range(1, 6)]
                    n_text = " ".join([p for p in n_parts if p.lower() != 'nan'])
                elif self.dataset_name in MM_GRAPH_DATASETS:
                    n_text = str(n_row['text'])
                else:
                    n_text = str(n_row['description'])
                neighbor_texts.append(n_text)

            inputs = build_prompt(
                title=target_texts,
                neighbor_texts=neighbor_texts
            )
            input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
        
        elif self.context == 'all':
            inputs = build_prompt(
                title=target_texts,
                neighbor_texts=[]
            )
            current_image = self.get_section_images(global_index)
            if current_image is None:
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.zeros((3,  224, 224), dtype=torch.float32))
            else:
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.tensor(current_image, dtype=torch.float32))
            
            max_text_length = self.max_input_length - self.n_visual_tokens
            input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            image_positions.append(input_ids.shape[0] + torch.arange(self.n_visual_tokens))
            input_ids = torch.cat([input_ids, visual_ids], dim=0)
            
            neighbors = self.neighbor_dict.get(global_index, [])
            neighbors = neighbors[:self.max_text_neighbors] 

            for n_id in neighbors:
                n_row = self.df.iloc[n_id]
                if self.dataset_name == 'Flickr30k':
                    n_parts = [str(n_row[f'description{i}']) for i in range(1, 6)]
                    neighbor_texts = " ".join([p for p in n_parts if p.lower() != 'nan'])
                elif self.dataset_name in MM_GRAPH_DATASETS:
                    neighbor_texts = str(n_row['text'])
                else:
                    neighbor_texts = str(n_row['description'])
                    
                neighbor_image = self.get_section_images(n_id)
                
                if neighbor_image is None:
                    visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                    neighbor_image = torch.zeros((3, 224, 224), dtype=torch.float32)
                else:
                    visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                    neighbor_image = torch.tensor(neighbor_image, dtype=torch.float32)
                
                max_text_length = self.max_input_length - input_ids.shape[0] - self.n_visual_tokens
                context_ids = self.tokenizer(neighbor_texts, max_length=max_text_length, padding="do_not_pad", truncation=False, return_tensors="pt").input_ids[0]
                if input_ids.shape[0] + context_ids.shape[0] + visual_ids.shape[0] > self.max_input_length:
                    break
                images.append(neighbor_image)
                image_positions.append(input_ids.shape[0] + context_ids.shape[0] + torch.arange(self.n_visual_tokens))
                input_ids = torch.cat([input_ids, context_ids, visual_ids], dim=0)
            
                
        if self.decoder_only:
            seq_len = input_ids.size(0)
            padding_len = self.max_input_length - seq_len
            
            if padding_len > 0:
                pads = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                padded_input_ids = torch.cat([pads, input_ids], dim=0)
                padded_attention_mask = torch.cat([torch.zeros(padding_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)], dim=0)
            else:
                padded_input_ids = input_ids[-self.max_input_length:]
                padded_attention_mask = torch.ones(self.max_input_length, dtype=torch.long)

            if self.dataset_name == 'Flickr30k':
                labels_str = str(d['description1'])
                if labels_str.lower() == 'nan': labels_str = ""
            elif self.dataset_name in MM_GRAPH_DATASETS:
                labels_str = str(d['text'])
            else:
                labels_str = str(d['description'])
            label_ids_list = self.tokenizer(
                labels_str, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_output_length - 1
            ).input_ids
            
            label_ids_list.append(self.tokenizer.eos_token_id)
            label_tensor = torch.tensor(label_ids_list, dtype=torch.long)
            
            l_seq_len = label_tensor.size(0)
            l_pad_len = self.max_output_length - l_seq_len
            
            if l_pad_len > 0:
                l_pads = torch.full((l_pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                padded_labels = torch.cat([label_tensor, l_pads], dim=0)
                
                l_ignore = torch.full((l_pad_len,), -100, dtype=torch.long)
                padded_loss_labels = torch.cat([label_tensor, l_ignore], dim=0)
                
                label_mask = torch.cat([torch.ones(l_seq_len, dtype=torch.long), torch.zeros(l_pad_len, dtype=torch.long)], dim=0)
            else:
                padded_labels = label_tensor[:self.max_output_length]
                padded_loss_labels = padded_labels
                label_mask = torch.ones(self.max_output_length, dtype=torch.long)

            final_input_ids = torch.cat([padded_input_ids, padded_labels], dim=0) 
            final_attention_mask = torch.cat([padded_attention_mask, label_mask], dim=0)
            prompt_ignore = torch.full((self.max_input_length,), -100, dtype=torch.long)
            final_labels = torch.cat([prompt_ignore, padded_loss_labels], dim=0)

            result = {
                "input_ids": final_input_ids,
                "attention_mask": final_attention_mask,
                "labels": final_labels
            }
            
        else:
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            labels_with_ignore_index = torch.LongTensor([label if label != 0 else -100 for label in labels])
            result = {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": labels_with_ignore_index}
            
        if self.context == 'all':
            images = torch.stack(images, dim=0)
            image_positions = torch.cat(image_positions, dim=0)
            result["images"] = images
            result["image_positions"] = image_positions
    
        return result
    
    def get_embedding_item(self, global_index):
        d = self.df.iloc[global_index]
        
        if self.dataset_name == 'Flickr30k':
            target_texts = "Generate a detailed description for the image."
        elif self.dataset_name in MM_GRAPH_DATASETS:
            target_texts = "Product Item"   
        else:
            target_texts = str(d['title'])
            
        inputs = build_prompt(title=target_texts, neighbor_texts=[]) 
        
        current_image = self.get_section_images(global_index)
        if current_image is None:
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img_tensor = torch.tensor(current_image, dtype=torch.float32)
            
        avail_len = self.max_input_length - self.n_visual_tokens
        if not self.decoder_only:
             avail_len -= 1

        text_ids = self.tokenizer(inputs, padding="do_not_pad", truncation=True, max_length=avail_len, return_tensors="pt").input_ids[0]

        has_eos = False
        if not self.decoder_only:
            if len(text_ids) > 0 and text_ids[-1] == self.tokenizer.eos_token_id:
                text_ids = text_ids[:-1]
                has_eos = True
        visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
        target_input_ids = torch.cat([text_ids, visual_ids], dim=0)
        image_positions = text_ids.shape[0] + torch.arange(self.n_visual_tokens)

        if not self.decoder_only and has_eos:
            eos_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            target_input_ids = torch.cat([target_input_ids, eos_tensor], dim=0)

        if self.decoder_only:
            seq_len = target_input_ids.size(0)
            padding_len = self.max_input_length - seq_len
            if padding_len > 0:
                pads = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                padded_input_ids = torch.cat([pads, target_input_ids], dim=0)
                padded_attention_mask = torch.cat([torch.zeros(padding_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)], dim=0)
                image_positions = image_positions + padding_len
            else:
                padded_input_ids = target_input_ids[-self.max_input_length:]
                padded_attention_mask = torch.ones(self.max_input_length, dtype=torch.long)

            if self.dataset_name == 'Flickr30k':
                labels_str = str(d['description1'])
                if labels_str.lower() == 'nan': labels_str = ""
            elif self.dataset_name in MM_GRAPH_DATASETS:
                labels_str = str(d['text'])
            else:
                labels_str = str(d['description'])
            label_tokens = self.tokenizer(labels_str, add_special_tokens=False, truncation=True, max_length=self.max_output_length-1).input_ids
            label_tokens.append(self.tokenizer.eos_token_id)
            label_tensor = torch.tensor(label_tokens, dtype=torch.long)
            
            l_seq_len = label_tensor.size(0)
            l_pad_len = self.max_output_length - l_seq_len
            if l_pad_len > 0:
                l_pads = torch.full((l_pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                padded_labels = torch.cat([label_tensor, l_pads], dim=0)
                label_mask = torch.ones(self.max_output_length, dtype=torch.long) 
                l_ignore = torch.full((l_pad_len,), -100, dtype=torch.long)
                padded_loss_labels = torch.cat([label_tensor, l_ignore], dim=0)
            else:
                padded_labels = label_tensor[:self.max_output_length]
                padded_loss_labels = padded_labels
                label_mask = torch.ones(self.max_output_length, dtype=torch.long)

            final_input_ids = torch.cat([padded_input_ids, padded_labels], dim=0)
            final_attention_mask = torch.cat([padded_attention_mask, label_mask], dim=0)
            prompt_ignore = torch.full((self.max_input_length,), -100, dtype=torch.long)
            final_labels = torch.cat([prompt_ignore, padded_loss_labels], dim=0)
            
        else:
            # Encoder-Decoder right padding
            model_inputs = self.tokenizer.pad({"input_ids": [target_input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            final_input_ids = model_inputs.input_ids[0]
            final_attention_mask = model_inputs.attention_mask[0]
            
            if not isinstance(image_positions, torch.Tensor):
                image_positions = torch.tensor(image_positions, dtype=torch.long)
            
            valid_img_pos = image_positions[image_positions < self.max_input_length]
            final_attention_mask[valid_img_pos] = 1 

            if self.dataset_name == 'Flickr30k':
                labels_str = str(d['description1'])
                if labels_str.lower() == 'nan': labels_str = ""
            elif self.dataset_name in MM_GRAPH_DATASETS:
                labels_str = str(d['text'])
            else:
                labels_str = str(d['description'])
            
            labels = self.tokenizer(labels_str, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            final_labels = torch.LongTensor([label if label != self.tokenizer.pad_token_id else -100 for label in labels])


        neighbor_texts = []
        neighbor_images = []
        
        position_texts = [] 
        position_images = []
        
        location_texts = []
        location_images = []
        
        location = 0 
        edge_list = [] 

        neighbors = self.neighbor_dict.get(global_index, [])
        if global_index in neighbors:
            neighbors.remove(global_index)
        neighbors = neighbors[:self.max_text_neighbors] 

        for n_idx, n_id in enumerate(neighbors):
            n_row = self.df.iloc[n_id]
            n_desc = ""
            
            if self.dataset_name == 'Flickr30k':
                n_parts = [str(n_row[f'description{i}']) for i in range(1, 6)]
                n_desc = " ".join([p for p in n_parts if p.lower() != 'nan'])             
            elif self.dataset_name in MM_GRAPH_DATASETS:
                n_desc = str(n_row['text'])
            else:
                n_desc = str(n_row['description'])

            if not n_desc or n_desc.lower() == 'nan': n_desc = ""

            neighbor_texts.append(n_desc)
            position_texts.append(n_idx + 1) 
            location_texts.append(location)
            
            edge_list.append((0, location + 1)) 
            
            location += 1

            n_image = self.get_section_images(n_id)
            if n_image is not None:
                neighbor_images.append(torch.tensor(n_image, dtype=torch.float32))
            else:
                print(f"[Warning] No image found for neighbor id {n_id} (global index). Using zero tensor.")
                neighbor_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            
            position_images.append(n_idx + 1)
            location_images.append(location)
            
            edge_list.append((0, location + 1))
            edge_list.append((location - 1 + 1, location + 1))
            
            location += 1

        total_neighbor_nodes = len(neighbor_texts) + len(neighbor_images) 
        node_num = 1 + total_neighbor_nodes
        
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.LongTensor(edge_list).t().contiguous()

        lpe = None
        graph_data = None

        if self.position_type == 'laplacian':
            node_value = torch.zeros((node_num))
            g_data = Data(x=node_value, edge_index=edge_index)
            try:
                lpe = utils.compute_LPE(g_data, k=8)
            except:
                lpe = torch.zeros((node_num, 8)) 

        elif self.position_type == 'gnn':
            graph_data = edge_index

        while len(neighbor_texts) < self.max_text_neighbors:
            neighbor_texts.append("")
            position_texts.append(0) 
            location_texts.append(location)
            location += 1
        
        while len(neighbor_images) < self.max_image_neighbors:
            neighbor_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            position_images.append(0)
            location_images.append(location)
            location += 1
                   
        tokenizer_to_use = self.text_tokenizer if hasattr(self, 'text_tokenizer') else self.tokenizer
        neighbor_encodings = tokenizer_to_use(
            neighbor_texts, 
            max_length=32, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        result = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "labels": final_labels,
            
            "images": img_tensor.unsqueeze(0),
            "image_positions": image_positions,
            
            "neighbor_input_ids": neighbor_encodings.input_ids,
            "neighbor_attention_mask": neighbor_encodings.attention_mask,
            "neighbor_pos_ids": torch.LongTensor(position_texts),
            "text_locations": torch.LongTensor(location_texts),
            
            "neighbor_images": torch.stack(neighbor_images, dim=0),
            "neighbor_images_pos_ids": torch.LongTensor(position_images),
            "image_locations": torch.LongTensor(location_images)
        }
        
        if self.position_type == 'laplacian' and lpe is not None:
            result["lpe"] = lpe
        if self.position_type == 'gnn' and graph_data is not None:
            result["graph"] = graph_data

        return result
    
def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

    elem = batch[0]

    if 'images' in elem:
        max_images = max([b['images'].shape[0] for b in batch])
        images_list = []
        image_positions_list = []

        for b in batch:
            current_img_num = b['images'].shape[0]
            n_pad = max_images - current_img_num
            
            if n_pad > 0:
                pad_imgs = torch.zeros((n_pad, 3, 224, 224), dtype=b['images'].dtype)
                images_list.append(torch.cat([b['images'], pad_imgs], dim=0))
                
                if current_img_num > 0:
                    tokens_per_image = b['image_positions'].shape[0] // current_img_num
                else:
                    tokens_per_image = 0
                pad_positions = torch.zeros((n_pad * tokens_per_image,), dtype=b['image_positions'].dtype)
                image_positions_list.append(torch.cat([b['image_positions'], pad_positions], dim=0))
            else:
                images_list.append(b['images'])
                image_positions_list.append(b['image_positions'])

        result['images'] = torch.stack(images_list, dim=0)
        result['image_positions'] = torch.stack(image_positions_list, dim=0)

    if 'neighbor_images' in elem:
        result['neighbor_input_ids'] = torch.stack([b['neighbor_input_ids'] for b in batch])
        result['neighbor_attention_mask'] = torch.stack([b['neighbor_attention_mask'] for b in batch])
        result['neighbor_pos_ids'] = torch.stack([b['neighbor_pos_ids'] for b in batch])
        result['text_locations'] = torch.stack([b['text_locations'] for b in batch])
        
        result['neighbor_images'] = torch.stack([b['neighbor_images'] for b in batch])
        result['neighbor_images_pos_ids'] = torch.stack([b['neighbor_images_pos_ids'] for b in batch])
        result['image_locations'] = torch.stack([b['image_locations'] for b in batch])
        
        if 'lpe' in elem:
            result['lpe'] = torch.stack([b['lpe'] for b in batch])
        
        if 'graph' in elem:
            nodes_per_sample = 1 + result['neighbor_pos_ids'].shape[1] + result['neighbor_images_pos_ids'].shape[1]
            
            big_edge_index_list = []
            for i, b in enumerate(batch):
                edge_index = b['graph'] 
                offset = i * nodes_per_sample
                shifted_edge_index = edge_index + offset
                
                big_edge_index_list.append(shifted_edge_index)
            
            result['graph'] = torch.cat(big_edge_index_list, dim=1)

    return result