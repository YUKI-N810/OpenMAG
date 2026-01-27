# -*- coding: utf-8 -*-
"""
Modality Matching Evaluator

This module defines the MatchingEvaluator class, used to perform modality matching evaluation tasks.
It inherits from BaseEvaluator and implements specific evaluation logic.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator

def calculate_clip_score(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray
) -> Optional[float]:
    """
    Calculates and returns the CLIP-score between a pair of image and text embeddings.
    Returns None if either embedding is a zero vector.
    """
    img_norm = np.linalg.norm(image_embedding)
    txt_norm = np.linalg.norm(text_embedding)

    if img_norm == 0 or txt_norm == 0:
        return None

    image_embedding = image_embedding / img_norm
    text_embedding = text_embedding / txt_norm
    return 100.0 * np.dot(image_embedding, text_embedding)


class MatchingEvaluator(BaseEvaluator):
    """
    Responsible for performing modality matching evaluation tasks.
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        Initialize the modality matching evaluator.
        """
        super().__init__(config, gnn_model)

    def evaluate(self) -> Dict[str, float]:
        """
        Execute the complete evaluation process.
        """
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("Error: Failed to get enhanced embeddings.")
            return {"error": "Failed to get enhanced embeddings."}

        enhanced_image_embeddings, enhanced_text_embeddings = enhanced_embeddings
        
        scores = [
            calculate_clip_score(img, txt) 
            for img, txt in zip(enhanced_image_embeddings, enhanced_text_embeddings)
        ]
        
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        if not valid_scores:
            print("Error: No valid samples to score.")
            return {"error": "No valid samples to score."}

        mean_score = np.mean(valid_scores)
        
        print(f"Evaluation completed. The average CLIP-score on {len(valid_scores)} valid samples is: {mean_score:.4f}")
        if num_invalid > 0:
            print(f"  (Warning: {num_invalid} zero-vector samples were ignored during evaluation)")
        
        return {"mean_clip_score": float(mean_score)}