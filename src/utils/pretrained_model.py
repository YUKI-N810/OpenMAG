import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, \
    CLIPModel, CLIPProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, T5EncoderModel
from tqdm import tqdm
from model.visual_encoder.ViG.vig import vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========================================
# Multimodal Extractors  
# ========================================
class Qwen2VLExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        
        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        texts = []
        for t in text:
            msg = [{"role": "user", "content": [{"type": "text", "text": t}]}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=False))

        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        feats = out.hidden_states[-1].mean(dim=1).float()
        return feats.cpu()

    def extract_image_features(self, image):
        if not isinstance(image, list):
            image = [image]

        texts = []
        pil_images = []

        for img in image:
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            else:
                pil_img = img.convert("RGB")
            
            pil_images.append(pil_img)

            msg = [{"role": "user", "content": [{"type": "image", "source": pil_img}]}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=False))

        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()

class Qwen25VLExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]

        texts = []
        for t in text:
            msg = [{"role": "user", "content": [{"type": "text", "text": t}]}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=False))

        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        feats = out.hidden_states[-1].mean(dim=1).float()
        return feats.cpu()

    def extract_image_features(self, image):
        if not isinstance(image, list):
            image = [image]

        texts = []
        pil_images = []

        for img in image:
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            else:
                pil_img = img.convert("RGB")
            
            pil_images.append(pil_img)
            
            msg = [{"role": "user", "content": [{"type": "image", "source": pil_img}]}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=False))

        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()

class CLIPExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = CLIPModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = CLIPProcessor.from_pretrained(local_path)
        self.model.eval()

        self.hidden_size = self.model.config.projection_dim
        
        self.max_text_len = self.model.config.text_config.max_position_embeddings

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len
        ).to(self.device)

        with torch.no_grad():
            text_feats = self.model.get_text_features(**inputs)
        
        return text_feats.float().cpu()

    def extract_image_features(self, image):
        if not isinstance(image, list):
            image = [image]

        pil_images = []
        for img in image:
            if img is None:
                pil_images.append(Image.new('RGB', (224, 224), color=(0, 0, 0)))
            elif isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            image_feats = self.model.get_image_features(**inputs)
            
        return image_feats.float().cpu()

# ========================================
# Textual Extractors  
# ========================================
class Llama32Extractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        feats = out.hidden_states[-1].mean(dim=1).float()
        return feats.cpu()
    
class BertExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,  
            device_map=device
        )
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        last_hidden_state = out.last_hidden_state
        
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        feats = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        
        return feats.cpu()
    
class RobertaExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,        
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        feats = out.hidden_states[-1].mean(dim=1).float()
        return feats.cpu()

class OPTExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        feats = out.hidden_states[-1].mean(dim=1).float()
        return feats.cpu()

class T5Extractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = T5EncoderModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
        self.hidden_size = self.model.config.d_model  

    def extract_text_features(self, text):
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        
        feats = sum_embeddings / sum_mask
        
        return feats.cpu()

# ========================================
# Visual Extractors  
# ========================================
class SwinExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = AutoModel.from_pretrained(
            local_path,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)

        self.hidden_size = self.model.config.hidden_size

    def extract_image_features(self, image):
        if not isinstance(image, list): image = [image]
        pil_images = []
        for img in image:
            if isinstance(img, str): pil_images.append(Image.open(img).convert("RGB"))
            else: pil_images.append(img.convert("RGB"))

        # Swin expects float32 input
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()
    
class ViTExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)

        self.hidden_size = self.model.config.hidden_size

    def extract_image_features(self, image):
        if not isinstance(image, list): image = [image]
        pil_images = []
        for img in image:
            if isinstance(img, str): pil_images.append(Image.open(img).convert("RGB"))
            else: pil_images.append(img.convert("RGB"))

        # Convert to bfloat16
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device).to(torch.bfloat16) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()
    
class DINOExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)

        self.hidden_size = self.model.config.hidden_size

    def extract_image_features(self, image):
        if not isinstance(image, list): image = [image]
        pil_images = []
        for img in image:
            if isinstance(img, str): pil_images.append(Image.open(img).convert("RGB"))
            else: pil_images.append(img.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device).to(torch.bfloat16) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()

class ImageBindExtractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(local_path)

        self.hidden_size = 1024

    def extract_image_features(self, image):
        if not isinstance(image, list): image = [image]
        pil_images = []
        for img in image:
            if isinstance(img, str): pil_images.append(Image.open(img).convert("RGB"))
            else: pil_images.append(img.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device).to(torch.bfloat16) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        return out.hidden_states[-1].mean(dim=1).float().cpu()
    
class ConvNextV2Extractor():
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(local_path)

        self.hidden_size = 1024

    def extract_image_features(self, image):
        if not isinstance(image, list): image = [image]
        pil_images = []
        for img in image:
            if isinstance(img, str): pil_images.append(Image.open(img).convert("RGB"))
            else: pil_images.append(img.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device).to(torch.bfloat16) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
             feats = out.pooler_output
        else:
             feats = out.last_hidden_state.mean(dim=[-2, -1])
             
        return feats.float().cpu()

class TrainableBackbone(nn.Module):
    def __init__(self, model_name, target_dim=768, num_unfrozen_layers=3):
        super(TrainableBackbone, self).__init__()
        self.model_name = model_name
        self.feat_dim = 0 
        self.is_text_model = False
        self.tokenizer = None
        
        ckpt_path = None    
        local_path = f"/root/autodl-tmp/hf_cache/{model_name}"

        # ------------------- Text Models -------------------
        if 'bert' in model_name:
            print(f"[TrainableBackbone] Loading BERT: {model_name}")
            self.backbone = AutoModel.from_pretrained(local_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.feat_dim = self.backbone.config.hidden_size
            self.is_text_model = True
            if num_unfrozen_layers >= 0:
                self._freeze_layers(self.backbone.encoder.layer, num_unfrozen_layers)
            
        # ------------------- Visual Models (ViT) -------------------
        elif 'vit' in model_name:
            print(f"[TrainableBackbone] Loading ViT: {model_name}")
            self.backbone = AutoModel.from_pretrained(local_path)
            # ViT usually uses [CLS] token or pooler_output (768)
            self.feat_dim = self.backbone.config.hidden_size 
            if num_unfrozen_layers >= 0:
                self._freeze_layers(self.backbone.encoder.layer, num_unfrozen_layers)
        
        # ------------------- ViG Series -------------------
        elif 'vig' in model_name:
            if 'vig_ti' in model_name:
                self.backbone = vig_ti_224_gelu()
                ckpt_path = '/root/autodl-tmp/hf_cache/ViG/vig_ti_74.5.pth'
            elif 'vig_s' in model_name:
                self.backbone = vig_s_224_gelu()
                ckpt_path = '/root/autodl-tmp/hf_cache/ViG/vig_s_80.6.pth'
            elif 'vig_b' in model_name:
                self.backbone = vig_b_224_gelu()
                ckpt_path = '/root/autodl-tmp/hf_cache/ViG/vig_b_82.6.pth'

            self.feat_dim = 1000
            
            # Logic for loading local weights
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"[TrainableBackbone] Loading local checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[TrainableBackbone] Weights loaded. Msg: {msg}")

        # ------------------- TNT Series -------------------
        elif 'tnt' in model_name:
            if 'tnt_s' in model_name:
                timm_name = 'tnt_s_patch16_224'
                ckpt_path = '/root/autodl-tmp/hf_cache/TNT/tnt_s_81.5.pth.tar'
                self.feat_dim = 384 
            elif 'tnt_b' in model_name:
                timm_name = 'tnt_b_patch16_224'
                ckpt_path = '/root/autodl-tmp/hf_cache/TNT/tnt_b_82.9.pth.tar'
                self.feat_dim = 640 
            
            self.backbone = timm.create_model(timm_name, pretrained=False)
            self.backbone.reset_classifier(0)

            # Logic for loading local weights
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"[TrainableBackbone] Loading local checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[TrainableBackbone] Weights loaded. Msg: {msg}")
        
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # =========================================================
        # 4. Unified Projection Layer (Projector) -> target_dim
        # =========================================================
        self.projector = nn.Linear(self.feat_dim, target_dim)

        # =========================================================
        # 5. Enable Full Gradients (End-to-End Trainable)
        # =========================================================
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True
            
    def _freeze_layers(self, layers, num_unfrozen):
        """
        Freeze all layers except the last num_unfrozen layers
        """
        total_layers = len(layers)
        freeze_until = total_layers - num_unfrozen
        
        print(f" -> [Layer Freezing] Total layers: {total_layers}. Freezing first {freeze_until} layers. Training last {num_unfrozen} layers.")
        
        # Freeze Embeddings
        if hasattr(self.backbone, 'embeddings'):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
                
        # Freeze preceding Transformer Layers
        for i in range(freeze_until):
            for param in layers[i].parameters():
                param.requires_grad = False
                
        # Ensure Projector and Pooler (if present) are trainable
        if hasattr(self.backbone, 'pooler') and self.backbone.pooler is not None:
             for param in self.backbone.pooler.parameters():
                param.requires_grad = True

    def forward(self, x, attention_mask=None):
        # Text mode
        if self.is_text_model:
            # [Optimization] Support passing input_ids (Tensor) directly
            if isinstance(x, torch.Tensor) and x.dtype in [torch.long, torch.int]:
                # x is input_ids, attention_mask must be passed
                out = self.backbone(input_ids=x, attention_mask=attention_mask)
            else:
                # Original logic: pass list[str]
                if isinstance(x, list):
                    inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.backbone.device)
                    out = self.backbone(**inputs)
                else:
                    raise ValueError("Input to Text Backbone must be List[str] or Tensor(input_ids)")
            
            # Pooling
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                features = out.pooler_output
            else:
                last_hidden_state = out.last_hidden_state
                # If passing Tensor, ensure mask device is correct
                if attention_mask is None:
                    # mask exists in inputs only when input is list; if taking tensor branch, external mask is required
                     mask = torch.ones_like(x).float() # Fallback (Not recommended)
                else:
                     mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                
                features = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        else:
            # Visual mode
            out = self.backbone(x)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                features = out.pooler_output
            elif hasattr(out, 'last_hidden_state'):
                features = out.last_hidden_state.mean(dim=1)
            else:
                features = out 
                
            if isinstance(features, (tuple, list)): features = features[0]
            
        return self.projector(features)