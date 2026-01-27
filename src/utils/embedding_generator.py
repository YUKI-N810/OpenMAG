import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import torch.nn as nn
from dgl.data.utils import load_graphs
from PIL import Image
from tqdm import tqdm
from .pretrained_model import Qwen2VLExtractor, Qwen25VLExtractor, Llama32Extractor, CLIPExtractor, RobertaExtractor, SwinExtractor, OPTExtractor, BertExtractor, ViTExtractor, DINOExtractor, T5Extractor, ConvNextV2Extractor, ImageBindExtractor
from abc import ABC, abstractmethod
import argparse

class BaseEmbeddingGenerator(ABC):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        self.rootdir = rootdir
        self.dataset_name = dataset_name
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.device = device
        
        self.datadir = os.path.join(self.rootdir, self.dataset_name)
        self.csv_file = f"{self.dataset_name}.csv"
        self.df = pd.read_csv(os.path.join(self.datadir, self.csv_file))
        self.metadata = self.df.copy()
        self.num_nodes = len(self.df)

        if text_encoder:
            self.text_extractor = self._init_extractor(text_encoder)
            self.raw_texts = self.get_raw_text()
        if visual_encoder:
            self.visual_extractor = self._init_extractor(visual_encoder)
            self.raw_images = self.get_raw_image()

    def _init_extractor(self, encoder_name):
        if encoder_name == "Qwen2.5-VL-3B-Instruct":
            extractor = Qwen25VLExtractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(2048, 768).to(self.device)
        elif encoder_name == "Qwen2-VL-7B-Instruct":
            extractor = Qwen2VLExtractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(3584, 768).to(self.device)
        elif encoder_name == "Llama3.2-1B-Instruct":
            extractor = Llama32Extractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(2048, 768).to(self.device)
        elif encoder_name == "clip-vit-large-patch14":
            extractor = CLIPExtractor(encoder_name, self.device)
        elif encoder_name == "vit-base-patch16-224":
            extractor = ViTExtractor(encoder_name, self.device)
        elif encoder_name == "xlm-roberta-base":
            extractor = RobertaExtractor(encoder_name, self.device)
        elif encoder_name == "bert-base-nli-mean-tokens":
            extractor = BertExtractor(encoder_name, self.device)
        elif encoder_name == "t5-large":
            extractor = T5Extractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(1024, 768).to(self.device)
        elif encoder_name == "swinv2-large-patch4-window12-192-22k":
            extractor = SwinExtractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(1536, 768).to(self.device)
        elif encoder_name == "dinov2-large":
            extractor = DINOExtractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(1024, 768).to(self.device)
        elif encoder_name == "imagebind-huge":
            extractor = ImageBindExtractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(1024, 768).to(self.device)
        elif encoder_name == "convnextv2-base-22k-224":
            extractor = ConvNextV2Extractor(encoder_name, self.device)
            self.proj_layer = torch.nn.Linear(1024, 768).to(self.device)
        elif encoder_name == "facebook-opt-125m":
            extractor = OPTExtractor(encoder_name, self.device)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        return extractor

    @abstractmethod
    def get_raw_text(self, start_idx=None, end_idx=None):
        """Return list of texts"""
        pass

    @abstractmethod
    def get_raw_image(self, start_idx=None, end_idx=None):
        """Return list of PIL images"""
        pass

    def generate_text_embeddings(self, batch_size=128):
        print(f"Dataset {self.dataset_name} loaded with {self.num_nodes} nodes")
        print(f"Using text encoder {self.text_encoder} on {self.device}")

        all_text_feats = []

        for start_idx in tqdm(range(0, self.num_nodes, batch_size), desc="Text Embedding Progress", ncols=100):
            end_idx = min(start_idx + batch_size, self.num_nodes)
            
            batch_texts = self.raw_texts[start_idx:end_idx]
            text_feats = self.text_extractor.extract_text_features(batch_texts) 
            if hasattr(self, "proj_layer"):
                text_feats = self.proj_layer(text_feats.to(self.device)).cpu().detach()
            
            all_text_feats.append(text_feats)

            del batch_texts, text_feats

        all_text_feats = torch.cat(all_text_feats, dim=0)

        text_dir = os.path.join(self.rootdir, self.dataset_name, "text_features")
        os.makedirs(text_dir, exist_ok=True)
        text_path = os.path.join(text_dir, f"{self.dataset_name}_text_{self.text_encoder.replace('.', '_').replace('-', '_')}_{all_text_feats.shape[1]}d.npy")
        np.save(text_path, all_text_feats.numpy())
        print(f"Saved text features → {text_path}")

        return all_text_feats
    
    def generate_visual_embeddings(self, batch_size=128):
        print(f"Dataset {self.dataset_name} loaded with {self.num_nodes} nodes")
        print(f"Using visual encoder {self.visual_encoder} on {self.device}")

        all_image_feats = []

        for start_idx in tqdm(range(0, self.num_nodes, batch_size), desc="Batch progress", ncols=100):
            end_idx = min(start_idx + batch_size, self.num_nodes)
            
            # Get list of image paths for the current batch (List[str])
            batch_paths = self.raw_images[start_idx:end_idx]

            # Pass path list directly, Extractor handles reading and batch inference internally
            batch_feats = self.visual_extractor.extract_image_features(batch_paths)

            # Projection layer processing
            if hasattr(self, "proj_layer"):
                batch_feats = self.proj_layer(batch_feats.to(self.device))
            
            all_image_feats.append(batch_feats.cpu().detach())

            del batch_paths, batch_feats
            torch.cuda.empty_cache()

        all_image_feats = torch.cat(all_image_feats, dim=0)

        # Save results
        image_dir = os.path.join(self.rootdir, self.dataset_name, "image_features")
        os.makedirs(image_dir, exist_ok=True)

        model_safe_name = self.visual_encoder.replace('.', '_').replace('-', '_').replace('/', '_')
        image_path = os.path.join(image_dir, f"{self.dataset_name}_image_{model_safe_name}_{all_image_feats.shape[1]}d.npy")
        
        np.save(image_path, all_image_feats.numpy())
        print(f"Saved image features → {image_path}")
        
        return all_image_feats
        
class MAGB(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        col = 'text' if self.dataset_name in ["Movies", "Toys", "Grocery"] else 'caption'
        raw_text = self.df[col].astype(str).to_list()
        return raw_text

    def get_raw_image(self):
        
        raw_image = []
        raw_image_path = os.path.join(self.datadir, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')
        
        # Assume self.df has an 'id' column corresponding to image filenames
        for img_id in tqdm(self.df['id'], desc=f"Loading images for {self.dataset_name}"):
            img_name = f"{img_id}.jpg"
            img_path = os.path.join(raw_image_path, img_name)
            raw_image.append(img_path if os.path.exists(img_path) else None)
        return raw_image

class NineRec(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        raw_text = (
            self.df['text_cn'].fillna('') + ' ' +
            self.df['text_en'].fillna('')
        ).tolist()
        return raw_text

    def get_raw_image(self):
        raw_image = []
        raw_image_path = os.path.join(self.datadir, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')
        
        # Assume self.df has an 'id' column corresponding to image filenames
        for img_id in tqdm(self.df['id'], desc=f"Loading images for {self.dataset_name}"):
            img_name = f"{img_id}.jpg"
            img_path = os.path.join(raw_image_path, img_name)
            raw_image.append(img_path if os.path.exists(img_path) else None)
        return raw_image
    
class mmgraph(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        col = 'text'
        texts = self.df[col].astype(str).to_list()
        raw_text = []
        for t in tqdm(texts, desc=f"Loading text for {self.dataset_name}", ncols=100):
            raw_text.append(t)
        return raw_text

    def get_raw_image(self):
        raw_image = []
        raw_image_path = os.path.join(self.datadir, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')
        
        for img_id in tqdm(self.df['id'], desc=f"Loading images for {self.dataset_name}"):
            img_name = f"{img_id}.jpg"
            img_path = os.path.join(raw_image_path, img_name)
            raw_image.append(img_path if os.path.exists(img_path) else None)
        return raw_image

class Flickr30kDataset(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        col = 'description1'
        texts = self.df[col].astype(str).to_list()
        raw_text = []
        for t in tqdm(texts, desc=f"Loading text for {self.dataset_name}", ncols=100):
            raw_text.append(t)
        return raw_text
    
    def get_raw_image(self):
        raw_image = []
        raw_image_path = os.path.join(self.datadir, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')

        for img_id in tqdm(self.df['id'], desc=f"Loading images for {self.dataset_name}"):
            img_name = f"{img_id}.jpg"
            img_path = os.path.join(raw_image_path, img_name)
            raw_image.append(img_path if os.path.exists(img_path) else None)
        return raw_image
    '''def generate_text_embeddings(self, batch_size=32):
        all_text_feats = []

        for start in tqdm(range(0, self.num_nodes, batch_size)):
            end = min(start + batch_size, self.num_nodes)
            batch_caps = self.raw_texts[start:end]

            batch_avg_feats = []
            for caps in batch_caps:
                feats = self.text_extractor.extract_text_features(caps)
                feats = feats.mean(dim=0)
                if hasattr(self, "proj_layer"):
                    feats = self.proj_layer(feats.to(self.device))
                batch_avg_feats.append(feats.cpu())

            all_text_feats.append(torch.stack(batch_avg_feats))

        all_text_feats = torch.cat(all_text_feats, dim=0)

        text_dir = os.path.join(self.rootdir, self.dataset_name, "text_features")
        os.makedirs(text_dir, exist_ok=True)
        path = os.path.join(
            text_dir,
            f"{self.dataset_name}_text_{self.text_encoder.replace('/', '_')}_768d.npy"
        )
        np.save(path, all_text_feats.numpy())

        return all_text_feats'''


class AdsDataset(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        # Text column for Ads dataset is 'text'
        raw_text = self.df['text'].astype(str).to_list()
        return raw_text

    def get_raw_image(self):
        raw_image = []
        raw_image_path = os.path.join(self.datadir, f'{self.dataset_name}Images_extracted', f'{self.dataset_name}Images')
        
        # Images for Ads dataset are based on 'id' column
        for img_id in tqdm(self.df['id'], desc=f"Loading images for {self.dataset_name}"):
            # Try different image formats
            img_found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_name = f"{img_id}{ext}"
                img_path = os.path.join(raw_image_path, img_name)
                raw_image.append(img_path if os.path.exists(img_path) else None)
            
            if not img_found:
                # If image does not exist, create a blank image
                raw_image.append(Image.new('RGB', (224, 224), color='white'))
        
        return raw_image

class SemArtDataset(BaseEmbeddingGenerator):
    def __init__(self, rootdir, dataset_name, text_encoder, visual_encoder, device):
        super().__init__(rootdir, dataset_name, text_encoder, visual_encoder, device)

    def get_raw_text(self):
        return self.df['DESCRIPTION'].astype(str).to_list()

    def get_raw_image(self):
        raw_image = []
        raw_image_path = os.path.join(
            self.datadir,
            "SemArtImages_extracted",
            "SemArtImages"
        )

        for img_file in self.df['IMAGE_FILE']:
            img_path = os.path.join(raw_image_path, img_file)
            raw_image.append(img_path if os.path.exists(img_path) else None)

        return raw_image

def build_dataset(rootdir, dataset_name, text_encoder, visual_encoder, device):
    if dataset_name in ["Movies", "Toys", "Grocery", "RedditS"]:
        return MAGB(rootdir, dataset_name, text_encoder, visual_encoder, device)
    elif dataset_name in ["DY", "KU", "TN", "QB", "Bili_Cartoon", "Bili_Dance", "Bili_Food", "Bili_Movie", "Bili_Music"]:
        return NineRec(rootdir, dataset_name, text_encoder, visual_encoder, device)
    elif dataset_name in ["books-nc", "ele-fashion", "cloth", "sports"]:
        return mmgraph(rootdir, dataset_name, text_encoder, visual_encoder, device)
    elif dataset_name == "Flickr30k":
        return Flickr30kDataset(rootdir, dataset_name, text_encoder, visual_encoder, device)
    elif dataset_name in ["MultiMET_ads","MultiMET_twitter","MultiMET_facebook"]:
        return AdsDataset(rootdir, dataset_name, text_encoder, visual_encoder, device)
    elif dataset_name == "SemArt":  
        return SemArtDataset(rootdir, dataset_name, text_encoder, visual_encoder, device)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for different datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--rootdir", type=str, default="/root/autodl-tmp/MMAG", help="Root directory for datasets")
    parser.add_argument("--text_encoder", type=str, default="", help="Text encoder name")
    parser.add_argument("--visual_encoder", type=str, default="", help="Visual encoder name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding generation")
    args = parser.parse_args()

    dataset = build_dataset(
        rootdir=args.rootdir,
        dataset_name=args.dataset_name,
        text_encoder=args.text_encoder,
        visual_encoder=args.visual_encoder,
        device=args.device
    )
    
    print(f"Dataset {args.dataset_name} initialized, generating embeddings...")
    #text_feats, image_feats = dataset.generate_and_save_embeddings(batch_size=args.batch_size)
    if args.text_encoder:
        text_feats = dataset.generate_text_embeddings(batch_size=args.batch_size)
    if args.visual_encoder:
        image_feats = dataset.generate_visual_embeddings(batch_size=args.batch_size)
    print("Embedding generation completed.")

    meta_dir = os.path.join(args.rootdir, args.dataset_name, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    
    meta_path = os.path.join(meta_dir, f"{args.dataset_name}_metadata.csv")
    dataset.metadata.to_csv(meta_path, index=False)
    
    print(f"Saved metadata → {meta_path}")
    
if __name__ == "__main__":
    main()