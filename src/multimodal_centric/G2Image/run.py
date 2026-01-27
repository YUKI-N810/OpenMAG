import os
import sys
from omegaconf import OmegaConf, open_dict

from .process import preprocess_dataset
from .train import main as train_entry
from .test import main as test_entry

def run_g2image(cfg):
    print(f"=== Running Task: G2Image (Dataset: {cfg.dataset.name}) ===")

    with open_dict(cfg):
        dataset_name = cfg.dataset.name
        model_name = cfg.model.name if "name" in cfg.model else "model"
        base_output_dir = cfg.task.output_dir
        base_logging_dir = cfg.task.logging_dir
        base_test_output_dir = cfg.task.test_output_dir if "test_output_dir" in cfg.task else None
        base_model_dir = cfg.task.model_dir if "model_dir" in cfg.task else None

        # train
        cfg.task.output_dir = os.path.join(base_output_dir, dataset_name, model_name)
        cfg.task.train_data_dir = os.path.join(cfg.dataset.data_root, dataset_name, "train")
        cfg.task.save_data_path = os.path.join(cfg.dataset.data_root, dataset_name)
        cfg.task.test_dir = os.path.join(cfg.dataset.data_root, dataset_name, "test")
        cfg.task.logging_dir = os.path.join(base_logging_dir, dataset_name, model_name)
        cfg.task.pretrained_image_embedding_dir = cfg.dataset.v_emb_path
        cfg.task.image_path = cfg.dataset.image_path

        # test
        if base_test_output_dir:
            cfg.task.test_output_dir = os.path.join(base_test_output_dir, dataset_name, model_name)
        if base_model_dir in (None, "", base_output_dir):
            cfg.task.model_dir = cfg.task.output_dir
        cfg.task.test_dir = os.path.join(cfg.dataset.data_root, dataset_name, "test")

    asin_map_path = os.path.join(cfg.task.save_data_path, "processed", "node_2_asin.pt")
    train_meta_path = os.path.join(cfg.task.save_data_path, "train", "metadata.jsonl")
    test_meta_path = os.path.join(cfg.task.save_data_path, "test", "metadata.jsonl")

    def check_file_valid(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    is_processed = (
        check_file_valid(asin_map_path) and 
        check_file_valid(train_meta_path) and 
        check_file_valid(test_meta_path)
    )

    if cfg.task.mode == 'train':
        if not is_processed:
            print("\n>>> Starting Preprocessing...")
            preprocess_dataset(cfg) 
        print("\n>>> Starting Training...")
        os.makedirs(cfg.task.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.task.output_dir, 'hydra_config.yaml'))
        train_entry(cfg)

    elif cfg.task.mode == 'test':
        print("\n>>> Starting Testing...")
        test_entry(cfg)