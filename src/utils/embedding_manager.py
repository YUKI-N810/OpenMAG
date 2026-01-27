import os
import glob
import torch
from omegaconf import OmegaConf, open_dict
import sys
from pathlib import Path
from .embedding_generator import build_dataset

TRAINABLE_ENCODERS = ["vig", "tnt", "bert", "vit"]

ENCODER_MAP = {
    # multimodal encoders
    "qwen2.5": "Qwen2.5-VL-3B-Instruct",
    "qwen2":   "Qwen2-VL-7B-Instruct",
    "clip":    "clip-vit-large-patch14",
    "llamavl": "Llama-3.2-11B-Vision-Instruct",
    # text encoders
    "llama3.2":  "Llama3.2-1B-Instruct",
    "bert":    "bert-base-nli-mean-tokens",
    "roberta": "xlm-roberta-base",
    "t5":      "t5-large",
    "opt":     "facebook-opt-125m",
    # visual encoders
    "swin":    "swinv2-large-patch4-window12-192-22k",
    "vit":     "vit-base-patch16-224",
    "dino":    "dinov2-large",
    "imagebind": "imagebind-huge",
    "convnext": "convnextv2-base-22k-224",
    
    "vig":      "vig", # Trainable
    "tnt":      "tnt", # Trainable
}


TEXT_ENCODERS = [
    "qwen2.5", "qwen2", "clip", "bert", "roberta", "t5", "opt", "llama3.2"
]

VISUAL_ENCODERS = [
    "qwen2.5", "qwen2", "clip", "swin", "vit", "dino", "imagebind", "convnext", 
    "vig", "tnt"
]

def full_name(short_name, modality='text'):
    if not short_name:
        return None
    
    short_name_lower = short_name.lower()
    
    if modality == 'text':
        whitelist = TEXT_ENCODERS
    else:
        whitelist = VISUAL_ENCODERS
        
    matched_key = None
    for key in whitelist:
        if key in short_name_lower:
            matched_key = key
            break
            
    if not matched_key:
        available_options = "\n\t".join(whitelist)
        
        other_modality = 'visual' if modality == 'text' else 'text'
        other_list = VISUAL_ENCODERS if modality == 'text' else TEXT_ENCODERS
        extra_hint = ""
        
        for k in other_list:
            if k in short_name_lower:
                extra_hint = f"\n(Hint: '{short_name}' is valid for [{other_modality}_encoder], but not for [{modality}_encoder].)\n"
                break

        error_msg = (
            f"\n\nIn 'config': Invalid {modality} encoder '{short_name}'\n"
            f"{extra_hint}\n"
            f"Available options for [{modality}_encoder]:\n\t{available_options}\n"
        )
        raise ValueError(error_msg)

    return ENCODER_MAP.get(matched_key, matched_key)

def find_embedding(data_root, dataset_name, encoder_name, modality):
    if not encoder_name:
        return None  
    sanitized_encoder = encoder_name.replace('.', '_').replace('-', '_')
    feature_dir = os.path.join(data_root, dataset_name, f"{modality}_features")
    filename = f"{dataset_name}_{modality}_{sanitized_encoder}_768d.npy"
    return os.path.join(feature_dir, filename)

def ensure_embeddings(cfg):
    data_root = cfg.dataset.data_root
    dataset_name = cfg.dataset.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    user_req_text_trainable = cfg.text_trainable
    user_req_visual_trainable = cfg.visual_trainable

    # --- Text Encoder ---
    if hasattr(cfg, 'text_encoder') and cfg.text_encoder:
        encoder_name = full_name(cfg.text_encoder, modality='text')
        
        is_trainable_text = False
        if encoder_name:
            for keyword in TRAINABLE_ENCODERS:
                if keyword in encoder_name.lower():
                    if keyword == 'vit' and 'clip' in encoder_name.lower():
                        continue
                    if user_req_text_trainable:
                        is_trainable_text = True
                    break

        if is_trainable_text:
            print(f"\n[Embedding Manager] Detected Text Trainable Encoder: {encoder_name}")
            print(f" -> End-to-end fine-tuning mode. Skipping offline .npy generation.")
            with open_dict(cfg):
                cfg.dataset.t_emb_path = "TRAINABLE_MODE"
                cfg.text_encoder = encoder_name
        else:
            target_path = find_embedding(data_root, dataset_name, encoder_name, 'text')
            
            if os.path.exists(target_path):
                print(f"[Embedding Manager] Detected existing local file: {target_path}")
            else:
                print(f"[Embedding Manager] File not found, initiating generator...")
                print(f"   - Target path: {target_path}")
                
                generator = build_dataset(
                    rootdir=data_root,
                    dataset_name=dataset_name,
                    text_encoder=encoder_name,
                    visual_encoder=None,
                    device=device
                )
                generator.generate_text_embeddings(batch_size=cfg.batch_size)

            with open_dict(cfg):
                cfg.dataset.t_emb_path = target_path
                cfg.text_encoder = encoder_name
                print(f"Textual Embedding Path: {target_path}")

    # --- Visual Encoder ---
    if hasattr(cfg, 'visual_encoder') and cfg.visual_encoder:
        encoder_name = full_name(cfg.visual_encoder, modality='visual')
        
        is_trainable = False
        if encoder_name:
            for keyword in TRAINABLE_ENCODERS:
                if keyword in encoder_name.lower():
                    if keyword == 'vit' and 'clip' in encoder_name.lower():
                        continue
                    
                    if user_req_visual_trainable:
                        is_trainable = True
                    break
        
        if is_trainable:
            print(f"\n[Embedding Manager] Detected Visual Trainable Encoder: {encoder_name}")
            print(f" -> This is an End-to-End training model.")
            print(f" -> Skipping offline .npy generation step.")
            
            with open_dict(cfg):
                cfg.dataset.v_emb_path = "TRAINABLE_MODE"
                cfg.visual_encoder = encoder_name
            
        else:
            target_path = find_embedding(data_root, dataset_name, encoder_name, 'image')
            
            if os.path.exists(target_path):
                print(f"[Embedding Manager] Detected existing local file: {target_path}")
            else:
                print(f"[Embedding Manager] File not found, initiating generator...")
                print(f"   - Target path: {target_path}")
                
                generator = build_dataset(
                    rootdir=data_root,
                    dataset_name=dataset_name,
                    text_encoder=None,
                    visual_encoder=encoder_name,
                    device=device
                )
                generator.generate_visual_embeddings(batch_size=cfg.batch_size)

            with open_dict(cfg):
                cfg.dataset.v_emb_path = target_path
                cfg.visual_encoder = encoder_name
                print(f"Visual Embedding Path: {target_path}")
            
    return cfg