"""
TextPooler code block is adapted from:
- Repository: Bert model in Huggingface Transformers
- Author: Huggingface
- Link: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    RobertaModel,
    CLIPVisionModel,
    CLIPModel
)

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    IA3Config,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
import torch.nn.functional as F
from .graph import GNNAdapter


class TextPooler(nn.Module):
    """
    Pool the hidden state corresponding to the first token.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttentionModel(nn.Module):
    """
    SelfAttentionModel is a wrapper around a pretrained language model.
    It supports the encoder-decoder models (e.g., T5) and decoder-only models (e.g., OPT).
    """
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.context = cfg.task.context
        self.decoder_only = cfg.task.decoder_only
        self.neighbor_mode = cfg.task.neighbor_mode
        self.position_type = cfg.task.position_type
        self.n_text_tokens = cfg.task.n_text_tokens
        self.n_visual_tokens = cfg.task.n_visual_tokens
        self.tokenizer = tokenizer

        if "t5" in cfg.task.generation_model:
            peft_task_type = TaskType.SEQ_2_SEQ_LM
            config = AutoConfig.from_pretrained(cfg.task.generation_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(cfg.task.generation_model, config=config)
        elif "opt" in cfg.task.generation_model or "llama" in cfg.task.generation_model.lower():
            peft_task_type = TaskType.CAUSAL_LM
            config = AutoConfig.from_pretrained(cfg.task.generation_model)
            model = AutoModelForCausalLM.from_pretrained(cfg.task.generation_model, config=config)
        else:
            raise ValueError(f"SelfAttentionModel does not support {cfg.task.generation_model}.")

        if cfg.task.peft_type == "none":
            self.lm = model
        else:
            if cfg.task.peft_type == "lora":
                if "t5" in cfg.task.generation_model.lower():
                    target_modules = ["q", "v"]
                    #target_modules = ["q", "v", "k", "o", "wi", "wo"]
                elif "opt" in cfg.task.generation_model.lower() or "llama" in cfg.task.generation_model.lower():
                    target_modules = ["q_proj", "v_proj"]
                else:
                    target_modules = ["q_proj", "v_proj"]
                
                peft_config = LoraConfig(
                    r=cfg.task.lora_r,
                    lora_alpha=cfg.task.lora_alpha,
                    target_modules=target_modules, 
                    lora_dropout=cfg.task.lora_dropout,
                    bias="none",
                    modules_to_save=["lm_head"],
                )
            elif cfg.task.peft_type == "prefix":
                peft_config = PrefixTuningConfig(
                    task_type=peft_task_type,
                    inference_mode=False,
                    num_virtual_tokens=64,
                    prefix_projection=True, 
                    encoder_hidden_size=config.hidden_size,
                    token_dim=config.hidden_size
                )
            elif cfg.task.peft_type == "prompt":
                peft_config = PromptTuningConfig(
                    task_type=peft_task_type,
                    num_virtual_tokens=cfg.task.num_virtual_tokens,            
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Analyze the multimodal neighbor context to generate a detailed image description:", 
                    tokenizer_name_or_path=cfg.task.generation_model,
                )
                
            elif cfg.task.peft_type == "ia3":
                if "llama" in cfg.task.generation_model.lower():
                    peft_config = IA3Config(
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=["k_proj", "v_proj", "down_proj"], 
                        feedforward_modules=["down_proj"],
                    )
                else:
                    # Original configuration for OPT
                    peft_config = IA3Config(
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=["q_proj", "v_proj", "down_proj", "fc2"],
                        feedforward_modules=["down_proj", "fc2"],
                    )
            
                self.lm = get_peft_model(model, peft_config)
                self.lm.print_trainable_parameters()
            else:
                raise ValueError(f"SelfAttentionModel does not support {cfg.task.peft_type}.")
            self.lm = get_peft_model(model, peft_config)

        self.input_embeddings = self.lm.get_input_embeddings()

        self.text_model = None
        if self.neighbor_mode == "embedding":
            config = AutoConfig.from_pretrained(cfg.task.text_model)
            embedding_dim = self.input_embeddings.embedding_dim * cfg.task.n_text_tokens
            if "roberta" in cfg.task.text_model:
                self.text_model = RobertaModel.from_pretrained(cfg.task.text_model, config=config)
            elif "CLIP" in cfg.task.text_model:
                self.text_model = CLIPModel.from_pretrained(cfg.task.text_model, config=config)
            self.text_pooler = TextPooler(config)
            self.text_embeddings = nn.Linear(config.hidden_size, embedding_dim)
            if cfg.task.position_type != "none":
                self.text_position_embeddings = nn.Embedding(cfg.task.max_output_length + 1, embedding_dim) 
            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False

        self.visual_model = None
        if self.context in ("session_all", "all"):
            # embedding_dim = self.input_embeddings.embedding_dim * cfg.task.n_visual_tokens
            embedding_dim = self.input_embeddings.embedding_dim
            self.visual_model = CLIPVisionModel.from_pretrained(cfg.task.visual_model)
            # self.visual_embeddings = nn.Linear(self.visual_model.config.hidden_size, embedding_dim)
            self.visual_embeddings = nn.Sequential(
            nn.Linear(self.visual_model.config.hidden_size, self.visual_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.visual_model.config.hidden_size, embedding_dim)
    )
            if cfg.task.position_type != "none":
                self.visual_position_embeddings = nn.Embedding(cfg.task.max_output_length + 1, embedding_dim) 
            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False
        
        if self.position_type == "laplacian":
            if self.context in ("section_only", "section_all", "text_only") or self.neighbor_mode == "raw":
                raise ValueError(f"[Laplacian PE] neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")
            k = 1 + cfg.task.max_text_neighbors + cfg.task.max_image_neighbors - 5
            embedding_dim = self.input_embeddings.embedding_dim * cfg.task.n_text_tokens
            self.lpe_embeddings = nn.Linear(k, embedding_dim)

        if self.position_type == "gnn":
            embedding_dim = self.input_embeddings.embedding_dim * cfg.task.n_text_tokens
            self.gnn = GNNAdapter(cfg, embedding_dim)
        
        if self.cfg.task.freeze_lm:
            print("Freezing the LM.")
            self.lm.eval()
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

    def get_text_embs(self, input_ids, attention_mask, pos_ids=None):
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = self.text_pooler(outputs.last_hidden_state)
        text_embs = self.text_embeddings(encoder_outputs)

        if self.position_type != "none" and pos_ids is not None:
            pos_ids = pos_ids.reshape(-1)
            text_embs = text_embs + self.text_position_embeddings(pos_ids)

        text_embs = text_embs.reshape(text_embs.shape[0], self.n_text_tokens, -1)
        return text_embs.reshape(batch_size, neighbor_num, self.n_text_tokens, -1)

    def get_visual_embs(self, pixel_values, pos_ids=None):
        batch_size, neighbor_num, pixel, width, height = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, pixel, width, height)

        outputs = self.visual_model(pixel_values)
        #encoder_outputs = outputs.pooler_output
        #visual_embs = self.visual_embeddings(encoder_outputs)
        patch_features = outputs.last_hidden_state[:, 1:, :]
        
        if patch_features.shape[1] != self.n_visual_tokens:
            patch_features = patch_features.transpose(1, 2)
            patch_features = F.adaptive_avg_pool1d(patch_features, self.n_visual_tokens)
            patch_features = patch_features.transpose(1, 2)
        visual_embs = self.visual_embeddings(patch_features)

        if self.position_type != "none" and pos_ids is not None:
            pos_ids = pos_ids.reshape(-1)
            visual_embs = visual_embs + self.visual_position_embeddings(pos_ids)

        visual_embs = visual_embs.reshape(visual_embs.shape[0], self.n_visual_tokens, -1)
        return visual_embs.reshape(batch_size, neighbor_num, self.n_visual_tokens, -1)

    def train(self, mode=True):
        super(SelfAttentionModel, self).train(mode=mode)
        if self.cfg.task.freeze_lm:
            self.lm.eval()
        if self.text_model is not None:
            self.text_model.eval()
        if self.visual_model is not None:
            self.visual_model.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images=None,
        image_positions=None,
        neighbor_input_ids=None,
        neighbor_attention_mask=None,
        neighbor_pos_ids=None,
        text_locations=None,
        neighbor_images=None,
        neighbor_images_pos_ids=None,
        image_locations=None,
        lpe=None,
        graph=None
    ):

        if self.neighbor_mode == "raw" and self.context in ("session", "text_only"):
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "raw" and self.context in ("session_all", "all"):
            input_embs = self.input_embeddings(input_ids)
            visual_embs = self.get_visual_embs(images)

            batch_size, seq_len, hidden_dim = input_embs.shape
            batch_idx = torch.arange(batch_size)[:, None]
            input_embs[batch_idx, image_positions] = visual_embs.reshape(batch_size, -1, hidden_dim)

            if self.decoder_only:
                labels[batch_idx, image_positions] = -100

            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "embedding" and self.context in ("session", "text_only"):
            batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
            neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            neighbor_embeds = neighbor_embeds.reshape(batch_size, neighbor_num * self.n_text_tokens, -1)
            neighbor_attention_mask = neighbor_pos_ids > 0
            neighbor_attention_mask = torch.repeat_interleave(neighbor_attention_mask, repeats=self.n_text_tokens, dim=1)

            input_embs = self.input_embeddings(input_ids)

            if self.decoder_only:
                input_embs = torch.cat((neighbor_embeds, input_embs), dim=1)
                attention_mask = torch.cat((neighbor_attention_mask, attention_mask), dim=1)
                
                neighbor_labels = -100 * torch.ones((batch_size, neighbor_num * self.n_text_tokens), dtype=labels.dtype).to(labels.device)
                labels = torch.cat((neighbor_labels, labels), dim=1)
            else:
                input_embs = torch.cat((input_embs, neighbor_embeds), dim=1)
                attention_mask = torch.cat((attention_mask, neighbor_attention_mask), dim=1)

            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "embedding" and self.context in ("session_all", "all"):
            # 1. Calculate self input Embeddings
            input_embs = self.input_embeddings(input_ids)

            if images is not None and image_positions is not None:
                self_visual_embs = self.get_visual_embs(images)
                batch_size, seq_len, hidden_dim = input_embs.shape
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = self_visual_embs.reshape(batch_size, -1, hidden_dim)

            # 2. Calculate neighbor Embeddings
            text_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            visual_embeds = self.get_visual_embs(neighbor_images, neighbor_images_pos_ids)
            
            # 3. Dynamic alignment: Calculate max Token count (compatible with n_text_tokens=4 and n_visual_tokens=256)
            max_tokens = max(text_embeds.shape[2], visual_embeds.shape[2])
            batch_size, total_neighbor_num = text_embeds.shape[0], text_embeds.shape[1] + visual_embeds.shape[1]

            neighbor_embeds = torch.zeros((batch_size, total_neighbor_num, max_tokens, hidden_dim), device=text_embeds.device, dtype=text_embeds.dtype)
            neighbor_attention_mask = torch.zeros((batch_size, total_neighbor_num, max_tokens), device=text_embeds.device, dtype=text_embeds.dtype)

            # 4. Pad text neighbors
            text_pad_len = max_tokens - text_embeds.shape[2]
            if text_pad_len > 0:
                text_embeds_padded = F.pad(text_embeds, (0, 0, 0, text_pad_len))
                text_mask_padded = F.pad(neighbor_attention_mask[:, :text_embeds.shape[1]], (0, text_pad_len)) # Here mask is usually generated from pos_ids, simplified by direct padding
                # Note: For code simplicity, reconstructing the mask directly might be safer, as follows:
                text_mask_origin = (neighbor_pos_ids > 0).unsqueeze(-1).expand(-1, -1, self.n_text_tokens)
                text_mask_padded = F.pad(text_mask_origin, (0, text_pad_len))
            else:
                text_embeds_padded = text_embeds
                text_mask_padded = (neighbor_pos_ids > 0).unsqueeze(-1).expand(-1, -1, self.n_text_tokens)

            neighbor_embeds[:, :text_embeds.shape[1]] = text_embeds_padded
            text_attention_mask_raw = (neighbor_pos_ids > 0).unsqueeze(-1).expand(-1, -1, self.n_text_tokens)
            if text_pad_len > 0:
                text_embeds = F.pad(text_embeds, (0, 0, 0, text_pad_len))
                text_attention_mask_raw = F.pad(text_attention_mask_raw, (0, text_pad_len))
            
            vis_pad_len = max_tokens - visual_embeds.shape[2]
            if vis_pad_len > 0:
                visual_embeds = F.pad(visual_embeds, (0, 0, 0, vis_pad_len))
            
            visual_attention_mask_raw = (neighbor_images_pos_ids > 0).unsqueeze(-1).expand(-1, -1, self.n_visual_tokens)
            if vis_pad_len > 0:
                visual_attention_mask_raw = F.pad(visual_attention_mask_raw, (0, vis_pad_len))

            # Execute assignment
            batch_idx = torch.arange(batch_size)[:, None]
            neighbor_embeds[batch_idx, text_locations] = text_embeds
            neighbor_embeds[batch_idx, image_locations] = visual_embeds
            
            target_dtype = neighbor_attention_mask.dtype
            neighbor_attention_mask[batch_idx, text_locations] = text_attention_mask_raw.to(dtype=target_dtype)
            neighbor_attention_mask[batch_idx, image_locations] = visual_attention_mask_raw.to(dtype=target_dtype)

            neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, hidden_dim)
            neighbor_attention_mask = neighbor_attention_mask.reshape(batch_size, -1)

            # 3. Handle GNN and Node Fusion
            if self.context == "all":
                if self.position_type == "laplacian":
                    lpe_embeddings = self.lpe_embeddings(lpe)
                    lpe_embeddings = lpe_embeddings.reshape(batch_size, total_neighbor_num + 1, n_tokens, hidden_dim)
                    neighbor_embeds = neighbor_embeds + lpe_embeddings[:, 1:].reshape(batch_size, -1, hidden_dim)
                
                elif self.position_type == "gnn":
                    batch_size, total_seq_len, hidden_dim = neighbor_embeds.shape
                    n_tokens = self.n_text_tokens
                    neighbors_len = total_seq_len // n_tokens
                    
                    # Construct Target Node (Node 0): Directly from input_embs after image injection
                    # Simple Mean Pooling + Repeat to adapt to GNN input format
                    target_node_feat = input_embs.mean(dim=1).unsqueeze(1).repeat(1, n_tokens, 1)
                    
                    neighbor_feats_4d = neighbor_embeds.reshape(batch_size, neighbors_len, n_tokens, hidden_dim)
                    
                    # GNN Input: [Target Node, Neighbor Nodes]
                    all_nodes_feats = torch.cat([target_node_feat.unsqueeze(1), neighbor_feats_4d], dim=1)
                    all_nodes_flat = all_nodes_feats.reshape(-1, n_tokens * hidden_dim)
                    
                    #gnn_output = self.gnn(all_nodes_flat, graph)
                    gnn_output, gnn_loss = self.gnn(all_nodes_flat, graph)
                    total_nodes = 1 + neighbors_len
                    gnn_output = gnn_output.reshape(batch_size, total_nodes, n_tokens, hidden_dim)
                    
                    # Extract updated Self features (Node 0) and Neighbor features (Node 1:)
                    new_self_feat = gnn_output[:, 0, :, :].reshape(batch_size, -1, hidden_dim)
                    new_neighbor_feats = gnn_output[:, 1:, :, :].reshape(batch_size, -1, hidden_dim)
                    
                    # Concatenate updated Self features with Neighbors, placed before Prompt as Context
                    # Input becomes: [Neighbors, Updated_Self, Original_Prompt_With_Image]
                    neighbor_embeds = torch.cat([new_neighbor_feats, new_self_feat], dim=1)
                    
                    # Synchronize Mask update: Add Mask for Self part (all 1s)
                    self_mask = torch.ones((batch_size, new_self_feat.shape[1]), device=neighbor_attention_mask.device)
                    neighbor_attention_mask = torch.cat([neighbor_attention_mask, self_mask], dim=1)


            if self.decoder_only:
                input_embs = torch.cat((neighbor_embeds, input_embs), dim=1)
                attention_mask = torch.cat((neighbor_attention_mask, attention_mask), dim=1)

                neighbor_labels = -100 * torch.ones((batch_size, neighbor_embeds.shape[1]), dtype=labels.dtype).to(labels.device)
                labels = torch.cat((neighbor_labels, labels), dim=1)
            else:
                input_embs = torch.cat((input_embs, neighbor_embeds), dim=1)
                attention_mask = torch.cat((attention_mask, neighbor_attention_mask), dim=1)

            model_kwargs = {
                'inputs_embeds': input_embs,
                'attention_mask': attention_mask,
                'labels': labels
            }
            #if not self.decoder_only:
            #    model_kwargs['input_ids'] = torch.zeros((input_embs.shape[0], input_embs.shape[1]), dtype=torch.long, device=input_embs.device)
            if not self.decoder_only:
                if self.cfg.task.peft_type == 'prefix':
                    model_kwargs['input_ids'] = torch.zeros((input_embs.shape[0], input_embs.shape[1]), dtype=torch.long, device=input_embs.device)
                else:
                    pass

            #return self.lm(**model_kwargs)
            outputs = self.lm(**model_kwargs)
            
            # If in training mode and DGF returns a valid loss, add it to the main task loss
            if self.training and self.context == "all" and self.position_type == "gnn":
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    # Ensure gnn_loss is a tensor and on the correct device
                    if isinstance(gnn_loss, torch.Tensor):
                         outputs.loss = outputs.loss + gnn_loss
            
            return outputs

        else:
            raise ValueError(f"Neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")
    
    def generate(self, input_ids, attention_mask, **kwargs):
        images = kwargs.pop('images', None)
        image_positions = kwargs.pop('image_positions', None)
        neighbor_input_ids = kwargs.pop('neighbor_input_ids', None)
        neighbor_attention_mask = kwargs.pop('neighbor_attention_mask', None)
        neighbor_pos_ids = kwargs.pop('neighbor_pos_ids', None)
        text_locations = kwargs.pop('text_locations', None)
        neighbor_images = kwargs.pop('neighbor_images', None)
        neighbor_images_pos_ids = kwargs.pop('neighbor_images_pos_ids', None)
        image_locations = kwargs.pop('image_locations', None)
        lpe = kwargs.pop('lpe', None)
        graph = kwargs.pop('graph', None)

        input_embs = None

        if self.neighbor_mode == "raw" and self.context in ("session_all", "all"):
            input_embs = self.input_embeddings(input_ids)
            if images is not None and image_positions is not None:
                visual_embs = self.get_visual_embs(images)
                batch_size, seq_len, hidden_dim = input_embs.shape
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = visual_embs.reshape(batch_size, -1, hidden_dim)

        elif self.neighbor_mode == "raw":
             input_embs = self.input_embeddings(input_ids)

        elif self.neighbor_mode == "embedding":
            input_embs = self.input_embeddings(input_ids)

            if self.context in ("session_all", "all") and images is not None and image_positions is not None:
                self_visual_embs = self.get_visual_embs(images)
                batch_size, seq_len, hidden_dim = input_embs.shape
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = self_visual_embs.reshape(batch_size, -1, hidden_dim)
            
            neighbor_embeds = None
            neighbor_mask = None
            
            if self.context in ("session", "text_only") and neighbor_input_ids is not None:
                batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
                neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
                neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, input_embs.shape[-1])
                valid_neighbors = (neighbor_input_ids != self.tokenizer.pad_token_id).any(dim=-1)
                neighbor_mask = torch.repeat_interleave(valid_neighbors, repeats=self.n_text_tokens, dim=1)
                neighbor_mask = neighbor_mask.reshape(batch_size, -1)

            elif self.context in ("session_all", "all") and neighbor_input_ids is not None:
                text_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
                visual_embeds = self.get_visual_embs(neighbor_images, neighbor_images_pos_ids)
                
                max_tokens = max(text_embeds.shape[2], visual_embeds.shape[2])
                
                batch_size, text_neighbor_num = text_embeds.shape[0], text_embeds.shape[1]
                visual_neighbor_num = visual_embeds.shape[1]
                total_neighbor_num = text_neighbor_num + visual_neighbor_num
                hidden_dim = text_embeds.shape[-1]

                # 2. Initialize container with max_tokens
                neighbor_embeds = torch.zeros((batch_size, total_neighbor_num, max_tokens, hidden_dim), device=input_embs.device, dtype=input_embs.dtype)
                
                # 3. Construct Mask container (Note: mask shape alignment is also needed during generation phase)
                # For simplicity, initialize with all 0s, fill later
                neighbor_mask_container = torch.zeros((batch_size, total_neighbor_num, max_tokens), device=input_embs.device, dtype=input_embs.dtype)
                
                # 4. Pad and assign text (Padding 8 -> 256)
                text_pad_len = max_tokens - text_embeds.shape[2]
                if text_pad_len > 0:
                    text_embeds_padded = F.pad(text_embeds, (0, 0, 0, text_pad_len))
                else:
                    text_embeds_padded = text_embeds
                
                target_dtype = neighbor_mask_container.dtype
                # Construct validity Mask for text
                valid_text = (neighbor_input_ids != self.tokenizer.pad_token_id).any(dim=-1) # [batch, n_neighbors]
                text_valid_mask = valid_text.unsqueeze(-1).expand(-1, -1, max_tokens) # Simply expand to max_tokens
                # (A more precise way is 1 only for the first n_text_tokens, 0 for padding, but all 1s is fine for attention as long as padding is 0 vectors)
                # For rigor, we do this:
                text_len = text_embeds.shape[2]
                text_valid_mask = torch.zeros((batch_size, text_neighbor_num, max_tokens), device=input_embs.device, dtype=target_dtype)
                # Set first text_len positions to valid_text values
                text_valid_mask[:, :, :text_len] = valid_text.unsqueeze(-1).expand(-1, -1, text_len).to(dtype=target_dtype)

                # 5. Pad and assign images (Theoretically not needed, but for code robustness)
                vis_pad_len = max_tokens - visual_embeds.shape[2]
                if vis_pad_len > 0:
                    visual_embeds_padded = F.pad(visual_embeds, (0, 0, 0, vis_pad_len))
                else:
                    visual_embeds_padded = visual_embeds
                
                valid_visual = (neighbor_images_pos_ids > 0) # [batch, n_neighbors]
                # Image mask: First visual_len positions are valid
                visual_len = visual_embeds.shape[2]
                visual_valid_mask = torch.zeros((batch_size, visual_neighbor_num, max_tokens), device=input_embs.device, dtype=target_dtype)
                visual_valid_mask[:, :, :visual_len] = valid_visual.unsqueeze(-1).expand(-1, -1, visual_len).to(dtype=target_dtype)

                # 6. Put into container
                batch_idx = torch.arange(batch_size)[:, None]
                neighbor_embeds[batch_idx, text_locations] = text_embeds_padded
                neighbor_embeds[batch_idx, image_locations] = visual_embeds_padded
                
                neighbor_mask_container[batch_idx, text_locations] = text_valid_mask
                neighbor_mask_container[batch_idx, image_locations] = visual_valid_mask
                
                # 7. Reshape
                neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, hidden_dim)
                neighbor_mask = neighbor_mask_container.reshape(batch_size, -1)
                
                if self.position_type == "gnn" and graph is not None:
                     batch_size, total_seq_len, hidden_dim = neighbor_embeds.shape
                     n_tokens = self.n_text_tokens
                     neighbors_len = total_seq_len // n_tokens
                     
                     target_node_feat = input_embs.mean(dim=1).unsqueeze(1).repeat(1, n_tokens, 1)
                     neighbor_feats_4d = neighbor_embeds.reshape(batch_size, neighbors_len, n_tokens, hidden_dim)
                     all_nodes_feats = torch.cat([target_node_feat.unsqueeze(1), neighbor_feats_4d], dim=1)
                     all_nodes_flat = all_nodes_feats.reshape(-1, n_tokens * hidden_dim)
                     
                     gnn_output, _ = self.gnn(all_nodes_flat, graph)
                     gnn_output = gnn_output.reshape(batch_size, 1 + neighbors_len, n_tokens, hidden_dim)
                     
                     new_self_feat = gnn_output[:, 0, :, :].reshape(batch_size, -1, hidden_dim)
                     new_neighbor_feats = gnn_output[:, 1:, :, :].reshape(batch_size, -1, hidden_dim)
                     neighbor_embeds = torch.cat([new_neighbor_feats, new_self_feat], dim=1)
                     self_mask = torch.ones((batch_size, new_self_feat.shape[1]), device=neighbor_mask.device)
                     neighbor_mask = torch.cat([neighbor_mask, self_mask], dim=1)

            if neighbor_embeds is not None:
                if neighbor_mask is None:
                    neighbor_mask = torch.ones((batch_size, neighbor_embeds.shape[1]), device=input_embs.device)
                if neighbor_mask.dim() == 3:
                    neighbor_mask = neighbor_mask.reshape(batch_size, -1)

                if self.decoder_only:
                    input_embs = torch.cat((neighbor_embeds, input_embs), dim=1)
                    attention_mask = torch.cat((neighbor_mask, attention_mask), dim=1)
                else:
                    input_embs = torch.cat((input_embs, neighbor_embeds), dim=1)
                    attention_mask = torch.cat((attention_mask, neighbor_mask), dim=1)

        generate_kwargs = {
            'inputs_embeds': input_embs,
            'attention_mask': attention_mask,
            **kwargs
        }

        if self.decoder_only:
            batch_size, seq_len, _ = input_embs.shape
            dummy_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_embs.device)
            if self.tokenizer.pad_token_id is not None:
                dummy_ids.fill_(self.tokenizer.pad_token_id)
            generate_kwargs['input_ids'] = dummy_ids
        else:
            pass

        return self.lm.generate(**generate_kwargs)