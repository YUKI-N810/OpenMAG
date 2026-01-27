# -*- coding: utf-8 -*-
"""
Prepare preprocessed data for modality alignment task (Parallel Version).

This script integrates two core stages and uses multiprocessing to accelerate processing:
1.  (Stage 1) Generate Ground Truth: 
    -   Extract noun phrases from raw data.
    -   Use GroundingDINO to find bounding boxes corresponding to phrases.
    -   Save results as a .jsonl file.
2.  (Stage 2) Extract and Cache Features:
    -   Read the .jsonl ground truth file.
    -   Extract text embeddings and image region embeddings for each (phrase, bounding box) pair.
    -   Save all information into a unified .pt file.
"""
import sys
import os
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import torch
import spacy
from PIL import Image
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import torch.multiprocessing as mp
import time
import yaml
import traceback
from hydra import initialize, compose
from omegaconf import OmegaConf

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.utils.storage_manager import StorageManager
from src.utils.pretrained_model import (
    Qwen2VLExtractor, Qwen25VLExtractor, Llama32Extractor, CLIPExtractor, 
    RobertaExtractor, SwinExtractor, OPTExtractor, BertExtractor, ViTExtractor, DINOExtractor, 
    T5Extractor, ConvNextV2Extractor, ImageBindExtractor
)

def init_extractor_wrapper(encoder_name, device):
    """
    Initialize Extractor and projection layer (if needed) based on name.
    Logic exactly replicates BaseEmbeddingGenerator._init_extractor
    """
    extractor = None
    proj_layer = None
    
    if not encoder_name:
        return None, None

    if encoder_name == "Qwen2.5-VL-3B-Instruct":
        extractor = Qwen25VLExtractor(encoder_name, device)
        proj_layer = torch.nn.Linear(2048, 768).to(device)
    elif encoder_name == "Qwen2-VL-7B-Instruct":
        extractor = Qwen2VLExtractor(encoder_name, device)
        proj_layer = torch.nn.Linear(3584, 768).to(device)
    elif encoder_name == "Llama3.2-1B-Instruct":
        extractor = Llama32Extractor(encoder_name, device)
        proj_layer = torch.nn.Linear(2048, 768).to(device)
    elif encoder_name == "clip-vit-large-patch14":
        extractor = CLIPExtractor(encoder_name, device)
    elif encoder_name == "vit-base-patch16-224":
        extractor = ViTExtractor(encoder_name, device)
    elif encoder_name == "xlm-roberta-base":
        extractor = RobertaExtractor(encoder_name, device)
    elif encoder_name == "bert-base-nli-mean-tokens":
        extractor = BertExtractor(encoder_name, device)
    elif encoder_name == "t5-large":
        extractor = T5Extractor(encoder_name, device)
        proj_layer = torch.nn.Linear(1024, 768).to(device)
    elif encoder_name == "swinv2-large-patch4-window12-192-22k":
        extractor = SwinExtractor(encoder_name, device)
    elif encoder_name == "dinov2-large":
        extractor = DINOExtractor(encoder_name, device)
        proj_layer = torch.nn.Linear(1024, 768).to(device)
    elif encoder_name == "imagebind-huge":
        extractor = ImageBindExtractor(encoder_name, device)
        proj_layer = torch.nn.Linear(1024, 768).to(device)
    elif encoder_name == "convnextv2-base-22k-224":
        extractor = ConvNextV2Extractor(encoder_name, device)
        proj_layer = torch.nn.Linear(1024, 768).to(device)
    elif encoder_name == "facebook-opt-125m":
        extractor = OPTExtractor(encoder_name, device)
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")
        
    return extractor, proj_layer

def worker_stage1(task_queue, result_queue, device_id, config):
    """Worker process function for Stage 1 (Simplified, no debug output)."""
    device = f"cuda:{device_id}"
    model_id = "/root/autodl-tmp/hf_cache/grounding-dino-base"
    _storage = StorageManager(config)

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    except Exception:
        print(f"[Worker {device_id}] Model loading failed:\n{traceback.format_exc()}", flush=True)
        return

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        print(f"[Worker {device_id}] spaCy loading failed:\n{traceback.format_exc()}", flush=True)
        return

    max_text_length = model.config.text_config.max_position_embeddings // 2

    while True:
        task = task_queue.get()
        if task is None:
            break

        node_index, dataset_name = task
        status = "unknown_error"
        grounding_results = []
        error_info = ""
        raw_data = None

        try:
            # Step A: Get Data
            raw_data = _storage.get_raw_data_by_index(dataset_name, node_index)
            
            if not raw_data or not raw_data.get("image_path") or not raw_data.get("text"):
                status = "skipped_no_raw_data"
            else:
                image_path, text = raw_data["image_path"], raw_data["text"]

                if not os.path.exists(image_path):
                    status = "image_not_found"
                else:
                    # Step B: Read Image
                    image = Image.open(image_path).convert("RGB")
                    
                    # Step C: spaCy Processing
                    doc = nlp(text)
                    phrases = [chunk.text for chunk in doc.noun_chunks]

                    if not phrases:
                        status = "no_noun_phrases"
                    else:
                        # Step D: Model Inference
                        text_for_grounding = ". ".join(phrases)
                        inputs = processor(
                            images=image,
                            text=text_for_grounding,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=max_text_length
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)

                        results = processor.post_process_grounded_object_detection(
                            outputs, inputs.input_ids,
                            threshold=0.4, text_threshold=0.25,
                            target_sizes=[image.size[::-1]]
                        )
                        
                        grounding_results = [
                            {"phrase": label, "box": box.cpu().numpy().tolist()}
                            for label, box in zip(results[0]["labels"], results[0]["boxes"])
                        ]
                        status = "success" if grounding_results else "grounding_failed"

        except Exception as e:
            status = "processing_error"
            error_msg = str(e)
            
            # Filter out non-critical error prints, only print key info
            print(f"  > Error Type: {type(e).__name__}", flush=True)
            print(f"  > Error Message: {error_msg}", flush=True)
            print(f"  > Stack Trace:\n{traceback.format_exc()}", flush=True)
            
            # flush=True is critical here!
            sys.stdout.flush()
            
        # Keep a concise key result output
        print(f"[Worker {device_id}] Completed task node_index={node_index}, status={status}", flush=True)
        result_queue.put((node_index, status, raw_data, grounding_results, error_info))



def run_stage1_parallel(dataset_name: str, output_file: Path, workers_per_gpu: List[int], config):
    """Execute the parallel version of Stage 1."""
    print("\n--- [Stage 1] Start parallel generation of ground truth file ---")
    _storage = StorageManager(config)
    try:
        node_ids = _storage.load_node_ids(dataset_name)
        num_nodes = len(node_ids)
    except FileNotFoundError as e:
        print(f"Error: Failed to load node ID file: {e}")
        sys.exit(1)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for i in range(num_nodes):
        task_queue.put((i, dataset_name))

    processes = []
    device_map = []
    for gpu_id, num_workers in enumerate(workers_per_gpu):
        for _ in range(num_workers):
            device_map.append(gpu_id)
    
    for _ in range(len(device_map)):
        task_queue.put(None)

    for rank, device_id in enumerate(device_map):
        p = mp.Process(target=worker_stage1, args=(task_queue, result_queue, device_id, config))
        p.start()
        processes.append(p)

    stats = defaultdict(int)
    results_buffer = {}
    failed_nodes_info = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(num_nodes), desc="Stage 1: Generating ground truth"):
            node_index, status, raw_data, grounding_results, error_info = result_queue.get()
            stats["total_nodes"] += 1
            stats[status] += 1
            if status == "success":
                record = {"node_index": node_index, "node_id": node_ids[node_index], "image_path": raw_data["image_path"], "text": raw_data["text"], "grounding": grounding_results}
                results_buffer[node_index] = record
            elif error_info:
                failed_nodes_info.append(f"  - Node {node_index}: Status={status}\n     Error details:\n{error_info}\n")

        for i in sorted(results_buffer.keys()):
            f.write(json.dumps(results_buffer[i], ensure_ascii=False) + "\n")

    for p in processes: p.join()

    print("\n--- [Stage 1] Diagnostics ---")
    print(f"Total nodes processed: {stats['total_nodes']}, Successfully generated ground truth: {stats['success']}")
    print("Failed/Skipped reasons:")
    for reason, count in stats.items():
        if reason not in ["total_nodes", "success"]: 
            print(f"  - {reason}: {count}")
    
    if failed_nodes_info:
        print("\n--- Detailed error report ---")
        for info in failed_nodes_info:
            print(info)

    print(f"--- [Stage 1] Completed, ground truth file saved to: {output_file} ---")
    return output_file

def worker_stage2(tasks: List[Dict], device_id: int, config, temp_file_path: Path):
    """Worker process function for Stage 2 (Static Allocation Version)."""
    device = f"cuda:{device_id}"
    
    text_encoder_name = config.text_encoder
    visual_encoder_name = config.visual_encoder

    print(f"[Worker {device_id}] Initializing models: Text={text_encoder_name}, Visual={visual_encoder_name}")

    text_extractor, text_proj = init_extractor_wrapper(text_encoder_name, device)
    visual_extractor, visual_proj = init_extractor_wrapper(visual_encoder_name, device)

    results = []
    # Use position=device_id to display progress bars for each worker on separate lines
    if text_extractor is None or visual_extractor is None:
        print(f"[Worker {device_id}] CRITICAL ERROR: Failed to initialize extractors. Check encoder names!")
        return # Exit immediately to avoid invalid loop

    results = []
    
    for task in tqdm(tasks, desc=f"Worker (GPU:{device_id}) Stage 2", position=device_id+1):
        try:
            image_path = task.get("image_path")
            if not image_path or not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            
            for grounding_pair in task["grounding"]:
                phrase = grounding_pair["phrase"]
                box = grounding_pair["box"]

                # --- 1. Text Features ---
                phrase_embed = text_extractor.extract_text_features(phrase) # Returns Tensor
                if text_proj is not None:
                    phrase_embed = text_proj(phrase_embed.to(device))
                
                # Move back to CPU, ensure it is a Tensor
                phrase_embed = phrase_embed.detach().cpu().squeeze()

                # --- 2. Visual Features ---
                region_image = image.crop((box[0], box[1], box[2], box[3]))
                region_embed = visual_extractor.extract_image_features(region_image) # Returns Tensor
                if visual_proj is not None:
                    region_embed = visual_proj(region_embed.to(device))
                
                # Move back to CPU
                region_embed = region_embed.detach().cpu().squeeze()
                
                if isinstance(phrase_embed, np.ndarray):
                    phrase_embed = torch.from_numpy(phrase_embed)
                if isinstance(region_embed, np.ndarray):
                    region_embed = torch.from_numpy(region_embed)

                results.append({
                    "node_index": task["node_index"],
                    "phrase": phrase,
                    "box": box,
                    "phrase_embedding": phrase_embed.float(),
                    "region_embedding": region_embed.float()
                })
        except Exception as e:
            # [Debug Suggestion] Do not continue silently, print the first few errors to check
            # print(f"[Worker {device_id}] Error: {e}") 
            continue
            
    # Pre-save check
    if len(results) == 0:
        print(f"[Worker {device_id}] Warning: No results generated! Check image paths or model execution.")
    else:
        print(f"[Worker {device_id}] Saving {len(results)} results.")
        
    torch.save(results, temp_file_path)

def run_stage2_static_parallel(ground_truth_file: Path, output_file: Path, config, workers_per_gpu: List[int]):
    """Execute the static parallel version of Stage 2."""
    print("\n--- [Stage 2] Start static parallel extraction and caching of features ---")
    if not ground_truth_file.exists():
        print(f"Error: Ground truth file not found, please run stage 1 first."); sys.exit(1)

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    device_map = [gpu_id for gpu_id, num_workers in enumerate(workers_per_gpu) for _ in range(num_workers)]
    total_workers = len(device_map)
    
    # Statically split the task list into N sub-lists
    chunks = [tasks[i::total_workers] for i in range(total_workers)]
    
    temp_dir = output_file.parent / f"temp_stage2_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)
    
    processes = []
    for i in range(total_workers):
        temp_file = temp_dir / f"part_{i}.pt"
        p = mp.Process(target=worker_stage2, args=(chunks[i], device_map[i], config, temp_file))
        p.start()
        processes.append(p)
        
    for p in processes: p.join()
    
    # Merge results from all temporary files
    final_data = []
    print("--- [Stage 2] All worker processes completed, merging results... ---")
    for temp_file_path in sorted(temp_dir.glob("part_*.pt")):
        final_data.extend(torch.load(temp_file_path))
        os.remove(temp_file_path)
    os.rmdir(temp_dir)
    
    torch.save(final_data, output_file)
    print(f"--- [Stage 2] Completed, processed {len(final_data)} feature pairs, preprocessed features saved to: {output_file} ---")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for modality alignment task (parallel version).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to process (e.g., toys). This must match a file in configs/dataset/.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "2"], help="Stage to execute. '1' for ground truth, '2' for features.")
    parser.add_argument("--ground_truth_file", type=Path, default=None, help="Manually specify the input ground truth file path for Stage 2.")
    parser.add_argument("--output_file", type=Path, default=None, help="Manually specify the output file path for the final preprocessed data for Stage 2.")
    parser.add_argument("--workers-per-gpu", type=int, nargs='+', required=True, help="Specify the number of workers to start per GPU, e.g., '1 3 3 1'.")
    args = parser.parse_args()

    relative_config_path = "../../../../configs"
    with initialize(config_path=relative_config_path, job_name="alignment_preprocess", version_base="1.2"):
        cfg = compose(config_name="config", overrides=[f"dataset={args.dataset.lower()}"])
        config = OmegaConf.to_container(cfg, resolve=True)
    
    dataset_name = cfg.dataset.name

    base_output_dir = Path(__file__).resolve().parent / "ground_truth"
    default_gt_file = base_output_dir / f"{dataset_name}_ground_truth.jsonl"
    default_output_file = base_output_dir / f"{dataset_name}_alignment_preprocessed.pt"

    ground_truth_file_to_use = args.ground_truth_file or default_gt_file
    final_output_file = args.output_file or default_output_file

    if args.stage in ["all", "1"]:
        generated_gt_path = run_stage1_parallel(dataset_name, default_gt_file, args.workers_per_gpu)
        if args.stage == "all":
            run_stage2_static_parallel(generated_gt_path, final_output_file, config, args.workers_per_gpu)
    elif args.stage == "2":
        run_stage2_static_parallel(ground_truth_file_to_use, final_output_file, config, args.workers_per_gpu)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to ensure CUDA is correctly initialized in child processes
    mp.set_start_method("spawn", force=True)
    main()