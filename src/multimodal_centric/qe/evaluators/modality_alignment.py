# -*- coding: utf-8 -*-
"""
Modality Alignment Evaluator
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Dict, Any

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator
from src.multimodal_centric.qe.evaluators.modality_matching import calculate_clip_score
from src.multimodal_centric.qe.scripts.prepare_alignment_data import run_stage1_parallel, run_stage2_static_parallel
from src.utils.storage_manager import StorageManager

class AlignmentEvaluator(BaseEvaluator):
    """
    Responsible for executing fine-grained modality alignment evaluation tasks.
    This evaluator relies on a pre-processed data file containing all necessary
    pre-computed features (phrase embeddings, region embeddings, etc.).
    """
    def __init__(self, config, gnn_model: torch.nn.Module):
        super().__init__(config, gnn_model)
        self.config = config
        self.dataset_name = config.dataset.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t_name = config.text_encoder.replace("/", "_").replace("-", "_")
        v_name = config.visual_encoder.replace("/", "_").replace("-", "_")
        filename = f"{self.dataset_name}_{t_name}_{v_name}_align.pt"
        
        self.preprocessed_data_dir = Path(__file__).resolve().parent.parent / "scripts" / "ground_truth"
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessed_data_path = self.preprocessed_data_dir / filename
        self.ground_truth_path = self.preprocessed_data_dir / f"{self.dataset_name}_ground_truth.jsonl" # GT file is generic, no need to change
        
        self._storage = StorageManager(self.config)

    def _run_preprocessing_pipeline(self):
        """
        Automatically run the Stage 1 and Stage 2 preprocessing pipeline.
        """
        print("\n" + "="*50)
        print(f"Auto-Trigger: Preprocessed data not found at {self.preprocessed_data_path}")
        print("Starting preprocessing pipeline... This may take a while.")
        print("="*50 + "\n")

        if run_stage1_parallel is None or run_stage2_static_parallel is None:
            raise ImportError("Cannot run preprocessing because the scripts could not be imported. Please check your project structure.")

        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            workers_per_gpu = [4] * num_gpus # Default strategy: 4 workers per card
        else:
            workers_per_gpu = [0] 

        print(f"Info: Detected {num_gpus} GPUs. Using workers_per_gpu strategy: {workers_per_gpu}")

        # 2. Ensure Multiprocessing start method is spawn (CUDA compatibility)
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass # Pass if already set

        # 3. Run Stage 1 (Generate Ground Truth)
        # Even if jsonl exists, just to be safe, if the final pt is missing, we check jsonl
        if not self.ground_truth_path.exists():
            print(f"Step 1/2: Generating Ground Truth ({self.ground_truth_path})...")
            run_stage1_parallel(
                dataset_name=self.dataset_name, 
                output_file=self.ground_truth_path, 
                workers_per_gpu=workers_per_gpu,
                config=self.config
            )
        else:
            print(f"Step 1/2: Ground Truth file found ({self.ground_truth_path}), skipping generation.")

        # 4. Run Stage 2 (Feature Extraction)
        print(f"Step 2/2: Extracting Features to ({self.preprocessed_data_path})...")
        run_stage2_static_parallel(
            ground_truth_file=self.ground_truth_path,
            output_file=self.preprocessed_data_path,
            config=self.config,
            workers_per_gpu=workers_per_gpu
        )
        
        print("\n" + "="*50)
        print("Preprocessing Completed. Resuming Evaluation.")
        print("="*50 + "\n")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Execute the complete fine-grained alignment evaluation process.
        """
        if not self.preprocessed_data_path.exists():
            try:
                self._run_preprocessing_pipeline()
            except Exception as e:
                import traceback
                print(f"Critical Error during auto-preprocessing: {e}")
                print(traceback.format_exc())
                return {"error": f"Preprocessing failed: {str(e)}"}
            
            if not self.preprocessed_data_path.exists():
                return {"error": "Preprocessing finished but output file is still missing."}

        print(f"Loading preprocessed data from '{self.preprocessed_data_path}'...")
        preprocessed_data = torch.load(self.preprocessed_data_path)
        
        if not preprocessed_data:
            print("Error: Preprocessed data file is empty.")
            return {"error": "Preprocessed data file is empty."}

        print("Loading global features for GNN enhancement...")
        global_image_embeds = self._storage.get_embedding(self.config.dataset.v_emb_path)
        global_text_embeds = self._storage.get_embedding(self.config.dataset.t_emb_path)
        if global_image_embeds is None or global_text_embeds is None:
            raise ValueError("Failed to load global embeddings for evaluation.")
        
        global_features = np.concatenate((global_text_embeds, global_image_embeds), axis=1)
        global_features_tensor = torch.from_numpy(global_features).float().to(self.device)
        
        edge_index = self._storage.load_graph().edge_index.to(self.device)

        all_scores = []
        invalid_pairs_count = 0
        
        for item in tqdm(preprocessed_data, desc="Evaluating Alignment Scores"):
            node_index = item["node_index"]
            phrase_embedding = item["phrase_embedding"]
            region_embedding = item["region_embedding"]

            # Perform numerical checks using torch.isnan and torch.isinf
            if torch.isnan(phrase_embedding).any() or torch.isinf(phrase_embedding).any() or \
               torch.isnan(region_embedding).any() or torch.isinf(region_embedding).any():
                invalid_pairs_count += 1
                continue

            phrase_embedding = phrase_embedding.to(self.device)
            region_embedding = region_embedding.to(self.device)

            # 1. Construct GNN input
            if phrase_embedding.shape != region_embedding.shape:
                min_dim = min(phrase_embedding.shape[0], region_embedding.shape[0])
                phrase_embedding = phrase_embedding[:min_dim]
                region_embedding = region_embedding[:min_dim]

            local_feature_pair = torch.cat([phrase_embedding, region_embedding]).unsqueeze(0)
            
            expected_dim = global_features_tensor.shape[1]
            if local_feature_pair.shape[1] != expected_dim:
                print(f"Warning: Local feature dimension {local_feature_pair.shape[1]} does not match global feature dimension {expected_dim}. Skipping this pair.")
                invalid_pairs_count += 1
                continue

            # 2. Replace corresponding rows in global features
            temp_features = global_features_tensor.clone()
            temp_features[node_index, :] = local_feature_pair
            
            # 3. GNN Enhancement
            with torch.no_grad():
                _, enhanced_img, enhanced_txt = self.gnn_model(temp_features, edge_index)
            
            # 4. Calculate Score
            enhanced_local_img = enhanced_img[node_index].cpu().numpy()
            enhanced_local_txt = enhanced_txt[node_index].cpu().numpy()
            score = calculate_clip_score(enhanced_local_img, enhanced_local_txt)
            if score is not None:
                all_scores.append(score)
            else:
                invalid_pairs_count += 1

        if invalid_pairs_count > 0:
            print(f"Warning: During evaluation, {invalid_pairs_count} feature pairs were skipped due to invalid values or dimension mismatches.")

        if not all_scores:
            print("Error: Failed to calculate any valid alignment scores.")
            return {"error": "No valid alignment scores could be calculated."}

        mean_alignment_score = np.nanmean(all_scores)
        print(f"Evaluation completed. The average alignment score on {len(all_scores)} valid local feature pairs is: {mean_alignment_score:.4f}")
        
        return {"mean_alignment_score": float(mean_alignment_score)}