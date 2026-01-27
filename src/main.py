# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from pathlib import Path

from utils.embedding_manager import ensure_embeddings

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg = ensure_embeddings(cfg)
    
    if cfg.task.name in ["nc", "lp", "gc", "cl", "g2text", "g2image"]:
        if 'params' in cfg.model:
            OmegaConf.set_struct(cfg.model, False)
            merged_config = OmegaConf.merge(cfg.model, cfg.model.params)
            cfg.model = merged_config
            OmegaConf.set_struct(cfg.model, True)

    if cfg.task.name == "nc":
        from graph_centric.nc.run_batch import run_nc
        run_nc(cfg)
    elif cfg.task.name == "lp":
        from graph_centric.lp.run_batch import run_lp
        run_lp(cfg)
    elif cfg.task.name == "cl":
        from graph_centric.cl.run_batch import run_clustering
        run_clustering(cfg)
    elif cfg.task.name in ["modality_matching", "modality_retrieval", "modality_alignment"]:
        from multimodal_centric.qe.run import run_qe
        run_qe(cfg)
    elif cfg.task.name == "g2text":
        from multimodal_centric.G2Text.language_modelling.run import run_g2text
        run_g2text(cfg)
    elif cfg.task.name == "g2image":
        from multimodal_centric.G2Image.run import run_g2image
        run_g2image(cfg)

if __name__ == "__main__":
    main()
