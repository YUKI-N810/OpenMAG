import os
import math
import argparse
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoImageProcessor,
)
from torchvision import transforms
# from torchmetrics.image.fid import FrechetInceptionDistance
# from .GraphAdapter import PadToSquare
# from .infer_pipeline import InstructG2IPipeline
from .GraphAdapter import PadToSquare
from .infer_pipeline import InstructG2IPipeline
import wandb



def read_data(test_dir):
    data = []
    with open(os.path.join(test_dir, 'metadata.jsonl')) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            tmp = json.loads(line)
            # data.append({
            #     'text': tmp['text'],
            #     'center_image': Image.open(os.path.join(test_dir, tmp['center'])).convert("RGB"),
            #     'neighbor_image': [Image.open(os.path.join(test_dir, fname)).convert("RGB") for fname in tmp[args.neighbor_key]]
            # })
            data.append({
                'text': tmp['text'],
                'center_image': tmp['center'],
                'neighbor_image': tmp['neighbors']
            })
    return data

def main(cfg):
    import logging
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    
    args = cfg.task
    test_output_dir = getattr(args, "test_output_dir", None)
    if not test_output_dir:
        test_output_dir = os.path.join(args.model_dir, "test_outputs")
        args.test_output_dir = test_output_dir
    os.makedirs(test_output_dir, exist_ok=True)

    # Evaluator
    # clip_id = "openai/clip-vit-large-patch14"
    # dino_id = "facebook/dinov2-large"
    clip_id = os.path.join(args.cache_dir,"clip-vit-large-patch14")
    dino_id = os.path.join(args.cache_dir,"dinov2-large")
    clip_model = CLIPModel.from_pretrained(clip_id, cache_dir=args.cache_dir).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id, cache_dir=args.cache_dir)
    dino_model = AutoModel.from_pretrained(dino_id, cache_dir=args.cache_dir).to(args.device)
    dino_processor = AutoImageProcessor.from_pretrained(dino_id, cache_dir=args.cache_dir)
    if args.if_wandb:
        wandb.init(
            project="Instructg2i",
            name="{}".format(args.wandb_init_name),
            mode="online",
            config=args
        )

    print(args.neighbor_key)

    # read the data(读metadata.jsonl)
    print('Reading data...')
    dataset = read_data(args.test_dir)
    
    mapping_path = os.path.join(args.save_data_path, 'processed', 'node_2_asin.pt')
    print(f"Loading node_2_asin mapping from {mapping_path}...")
    node_2_asin = torch.load(mapping_path)

    # image transformation function
    neighbor_transforms = transforms.Compose(
                [
                    PadToSquare(fill=(args.resolution, args.resolution, args.resolution), padding_mode='constant'),
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution)
                ]
        )

    def neighbor_transform_func(neighbor_images, gt_image):
        neighbor_image = [neighbor_transforms(n_img) for n_img in neighbor_images]
        neighbor_image += [neighbor_transforms(Image.fromarray(np.uint8(np.zeros_like(np.array(gt_image)))).convert('RGB'))] * (args.neighbor_num - len(neighbor_image))
        return neighbor_image
    
    def neighbor_transform_func_mine(neighbor_images):
        neighbor_images += ['Empty'] * (args.neighbor_num - len(neighbor_images))
        return neighbor_images

    def neighbor_mask_func(neighbor_images):
        neighbor_mask = [1] * len(neighbor_images)
        neighbor_mask += [0] * (args.neighbor_num - len(neighbor_mask))
        return neighbor_mask

    # init the pipeline
    print('Loading diffusion model...')
    pipe_graph2img = InstructG2IPipeline.from_pretrained(args.model_dir, args.neighbor_num, device=args.device, args=args, cfg=cfg)

    # run inference
    print('Scoring...')
    img_clip_scores = []
    dinov2_scores = []

    print(f'Total testing data:{len(dataset)}, max index: {args.max_index}')
    assert args.max_index <= len(dataset)
    num_diff_iter = math.ceil(args.max_index / args.diffusion_infer_batch_size)
    num_score_iter = math.ceil(args.max_index / args.score_batch_size)

    # diffusion model inference
    gt_images = []
    gen_images = []
    center_ids = []
    for idx in tqdm(range(num_diff_iter), desc="Diffusion model inference"):
        start = idx * args.diffusion_infer_batch_size
        end = min(args.max_index, (idx + 1) * args.diffusion_infer_batch_size)

        # get current batch data
        texts = [dataset[idd]['text'] for idd in range(start, end)]
        # neighbor_images = [neighbor_transform_func(dataset[idd]["neighbor_image"][:args.neighbor_num], dataset[idd]["center_image"]) for idd in range(start, end)]
        neighbor_images = [neighbor_transform_func_mine(dataset[idd]["neighbor_image"][:args.neighbor_num]) for idd in range(start, end)]

        neighbor_masks = [neighbor_mask_func(dataset[idd]["neighbor_image"][:args.neighbor_num]) for idd in range(start, end)]
                
        gen_image = pipe_graph2img(prompt=texts, neighbor_image=neighbor_images, neighbor_mask=torch.LongTensor(neighbor_masks), num_inference_steps=args.num_inference_steps).images
        if args.if_wandb:
            wandb.log({
                "example_image": wandb.Image(gen_image)
                })

        # prepare for later score calculation
        gt_images_idx = [dataset[idd]["center_image"] for idd in range(start, end)]
        center_ids.extend(gt_images_idx)
        for gt_idx in gt_images_idx:
            file_name = node_2_asin[int(gt_idx)] 
            load_path = os.path.join(args.image_path, f"{file_name}.jpg")
            gt_images.append(Image.open(load_path).convert('RGB'))
        gen_images.extend(gen_image)
        
        # if idx == 1:break

    for idx in tqdm(range(num_score_iter), desc="Score model inference"):
        start = idx * args.score_batch_size
        end = min(args.max_index, (idx + 1) * args.score_batch_size) 

        # clip scores
        gt_image_dp = clip_processor(images=[gt_images[idd] for idd in range(start, end)], return_tensors="pt", padding=True, truncation=True, max_length=77)
        gen_image_dp = clip_processor(images=[gen_images[idd] for idd in range(start, end)], return_tensors="pt", padding=True, truncation=True, max_length=77)

        gt_image_dp = {k: v.to(args.device) for k, v in gt_image_dp.items()}
        gen_image_dp = {k: v.to(args.device) for k, v in gen_image_dp.items()}
        
        with torch.no_grad():
            gt_image_features = clip_model.get_image_features(**gt_image_dp)
            gen_image_features = clip_model.get_image_features(**gen_image_dp)

            gt_image_features = gt_image_features / gt_image_features.norm(p=2, dim=1, keepdim=True)
            gen_image_features = gen_image_features / gen_image_features.norm(p=2, dim=1, keepdim=True)
            
            img_clip_score = torch.nn.functional.relu(torch.diagonal(torch.matmul(gt_image_features, gen_image_features.t()), 0))
                    
            img_clip_scores.extend(img_clip_score.tolist())

        # dino-v2 score
        gt_image_dp_dino = dino_processor(images=[gt_images[idd] for idd in range(start, end)], return_tensors="pt")
        gen_image_dp_dino = dino_processor(images=[gen_images[idd] for idd in range(start, end)], return_tensors="pt")

        gt_image_dp_dino = {k: v.to(args.device) for k, v in gt_image_dp_dino.items()}
        gen_image_dp_dino = {k: v.to(args.device) for k, v in gen_image_dp_dino.items()}

        with torch.no_grad():
            gt_image_features_dino = dino_model(**gt_image_dp_dino).pooler_output
            gen_image_features_dino = dino_model(**gen_image_dp_dino).pooler_output

            gt_image_features_dino = gt_image_features_dino / gt_image_features_dino.norm(p=2, dim=1, keepdim=True)
            gen_image_features_dino = gen_image_features_dino / gen_image_features_dino.norm(p=2, dim=1, keepdim=True)

            dino_score = torch.nn.functional.relu(torch.diagonal(torch.matmul(gt_image_features_dino, gen_image_features_dino.t()), 0))
            dinov2_scores.extend(dino_score.tolist())

    # fid = FrechetInceptionDistance()
    # fid_transforms = transforms.Compose(
    #         [
    #             PadToSquare(fill=(args.resolution, args.resolution, args.resolution), padding_mode='constant'),
    #             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(args.resolution),
    #             transforms.ToTensor(),
    #             transforms.ConvertImageDtype(torch.uint8)
    #         ]
    # )
    # gt_images_tensor = torch.stack([fid_transforms(tmp_img) for tmp_img in gt_images])
    # gen_images_tensor = torch.stack([fid_transforms(tmp_img) for tmp_img in gen_images])
    # fid.update(gt_images_tensor, real=True)
    # fid.update(gen_images_tensor, real=False)
    # fid_score = fid.compute().item()

    print('**************************** Ground Truth-based Metrics **************************************')
    clip_mean = float(np.mean(img_clip_scores))
    dino_mean = float(np.mean(dinov2_scores))
    print(f"Generated Image v.s. Ground Truth Image CLIP score: {clip_mean}")
    print(f"Generated Image v.s. Ground Truth Image Dino-v2 score: {dino_mean}")
    # print(f"Generated Image v.s. Ground Truth Image FiD score: {fid_score}")

    metrics = {
        "clip_mean": clip_mean,
        "dino_mean": dino_mean,
        "num_samples": int(args.max_index),
    }
    with open(os.path.join(test_output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    np.savez(
        os.path.join(test_output_dir, "scores.npz"),
        clip_scores=np.array(img_clip_scores, dtype=np.float32),
        dino_scores=np.array(dinov2_scores, dtype=np.float32),
    )

    if args.if_save_image:
        images_dir = os.path.join(test_output_dir, "generated")
        os.makedirs(images_dir, exist_ok=True)
        for idx, img in enumerate(gen_images):
            center_id = center_ids[idx] if idx < len(center_ids) else str(idx)
            img.save(os.path.join(images_dir, f"{center_id}.png"))

    if args.if_wandb:
        wandb.finish()

if __name__=='__main__':
    main()

        