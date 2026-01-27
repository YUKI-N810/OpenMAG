#!/usr/bin/env python
# coding=utf-8

""" Finetuning summary generation models"""
from collections import OrderedDict
import json
import os
import random
import sys
import tqdm
import wandb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import time
from time import perf_counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torchmetrics import BLEUScore
from torchmetrics.text import ROUGEScore
from warmup_scheduler import GradualWarmupScheduler
from torch.amp import autocast

from datasets import load_dataset
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    HfArgumentParser,
    set_seed,
    get_scheduler,
)
from transformers.optimization import Adafactor
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

#from wikiweb2m import load_wikiweb2m, WikiWeb2M
from multimodal_centric.G2Text.mydatasets import load_datasets, Datasets, collate_fn
from multimodal_centric.G2Text.cider import Cider

from multimodal_centric.G2Text.language_modelling import utils
from multimodal_centric.G2Text.model import SelfAttentionModel, CrossAttentionModel

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only display errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

best_acc1 = 0  # Variable to keep track of best model so far.

def run_g2text(cfg):
    print(OmegaConf.to_yaml(cfg))

    # 1. Set log directory
    i = 0
    base_log_dir = cfg.task.log_dir
    log_dir = os.path.join(base_log_dir, f'{cfg.task.run_name}_{i}')
    while os.path.exists(log_dir):
        i += 1
        log_dir = os.path.join(base_log_dir, f'{cfg.task.run_name}_{i}')
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. The save_dir here will be injected into cfg and passed to worker later
    cfg.task.save_dir = os.path.join(log_dir, 'ckpt.pth.tar')

    # 3. Wandb initialization
    # Convert OmegaConf to dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project=cfg.task.project, name=cfg.task.run_name, config=config_dict)
    
    print(f'Logging to {log_dir}.')

    # 4. Set random seed
    if cfg.task.seed is not None:
        random.seed(cfg.task.seed)
        torch.manual_seed(cfg.task.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. CUDNN deterministic is on.')

    # 5. Start multiprocessing
    ngpus_per_node = torch.cuda.device_count()
    
    dist_port = random.randint(20000, 60000)
    cfg.task.dist_url = f'tcp://127.0.0.1:{dist_port}'
    print(f"Using distributed url: {cfg.task.dist_url}")
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, log_dir, run))

def main_worker(gpu, world_size, cfg, log_dir, run):

    # Variable to keep track of best model so far.
    global best_acc1
    print("Use GPU: {} for training".format(gpu))
    dist_url = getattr(cfg.task, 'dist_url', 'tcp://127.0.0.1:1337')
    
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=gpu)

    # Prepare pretrained model
    if "t5" in cfg.task.generation_model:
        # encoder-decoder models
        cfg.task.decoder_only = False
        tokenizer = AutoTokenizer.from_pretrained(cfg.task.generation_model, use_fast=False)
        model = SelfAttentionModel(cfg, tokenizer)
    elif "opt" in cfg.task.generation_model or "llama" in cfg.task.generation_model.lower():
        # decoder-only models
        cfg.task.decoder_only = True
        tokenizer = AutoTokenizer.from_pretrained(cfg.task.generation_model, use_fast=False)
        tokenizer.padding_side = "left"
        
        # === Key addition: Fix Llama pad_token ===
        if "llama" in cfg.task.generation_model.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # =======================================
        
        model = SelfAttentionModel(cfg, tokenizer)
    elif "mpt" in cfg.task.generation_model:
        # OPT models with newly added cross-attention layers
        cfg.task.decoder_only = True
        cfg.task.generation_model = cfg.task.generation_model.replace("mpt", "opt")
        tokenizer = AutoTokenizer.from_pretrained(cfg.task.generation_model, use_fast=False)
        tokenizer.padding_side = "left"
        model = CrossAttentionModel(cfg, tokenizer)

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    if cfg.task.fp16:
        model = model.float()
    elif cfg.task.bf16:
        model = model.bfloat16()

    # Wandb logging
    if gpu % world_size == 0:
        _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
        run.watch(model)
        run.config.update({"total_params": total_trainable_params + total_nontrainable_params})
        run.config.update({"trainable_params": total_trainable_params})
        run.config.update({"non_trainable_params": total_nontrainable_params})

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if "t5" in cfg.task.generation_model:
        print('Using Adafactor as the optimizer.')
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=cfg.task.learning_rate)
        scheduler = None
    elif "opt" in cfg.task.generation_model or "llama" in cfg.task.generation_model.lower():
        print(f'Using AdamW. PEFT Type: {cfg.task.peft_type}')
        optimizer_cls = torch.optim.AdamW
        
        # Parameter grouping container
        projector_params = [] # Visual projection layer (requires large LR: 1e-3 ~ 2e-3)
        prompt_params = []    # Prompt Tuning Soft Tokens
        adapter_params = []   # <--- Added: Adapter/IA3 params (requires large LR: 3e-3 ~ 8e-3)
        normal_params = []    # Others (LoRA, etc.)

        # Iterate over all parameters requiring gradients
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 1. Visual Projector
            if "visual_embeddings" in name or "visual_model" in name:
                projector_params.append(param)
            
            # 2. Prompt Tuning (if you still keep it)
            elif "prompt_embeddings" in name or "soft_prompts" in name:
                prompt_params.append(param)
            
            # 3. === Added: Adapter (IA3) parameter capture ===
            # IA3 parameters usually contain "ia3" keyword
            elif "ia3" in name:
                adapter_params.append(param)
                # print(f"Found Adapter Param: {name}") # For debugging
            # ======================================

            # 4. Others (e.g., LoRA or Bias)
            else:
                normal_params.append(param)
        
        # Define learning rates for different groups
        lr_projector = cfg.task.learning_rate * 10 
        lr_prompt = 0.01 
        lr_adapter = 5e-3 # <--- Added: IA3 recommended learning rate (0.005)
        
        # Construct parameter list
        optim_groups = []
        
        if projector_params:
            print(f"Projector Params: {len(projector_params)}, LR: {lr_projector}")
            optim_groups.append({'params': projector_params, 'lr': lr_projector})
            
        if prompt_params:
            print(f"Prompt Tuning Params: {len(prompt_params)}, LR: {lr_prompt}")
            optim_groups.append({'params': prompt_params, 'lr': lr_prompt})

        # === Added: Add Adapter group ===
        if adapter_params:
            print(f"Adapter (IA3) Params: {len(adapter_params)}, LR: {lr_adapter} (Boosted)")
            optim_groups.append({'params': adapter_params, 'lr': lr_adapter})
        # ============================

        if normal_params:
            print(f"Other Trainable Params: {len(normal_params)}, LR: {cfg.task.learning_rate}")
            optim_groups.append({'params': normal_params, 'lr': cfg.task.learning_rate})

        if not optim_groups:
            raise ValueError("No trainable parameters found! Check your PEFT config.")

        optimizer = optimizer_cls(
            optim_groups, 
            betas=(cfg.task.adam_beta1, cfg.task.adam_beta2),
            weight_decay=cfg.task.weight_decay, 
            eps=1e-8
        )
        
        # Scheduler remains unchanged
        scheduler_steplr = StepLR(optimizer, step_size=(cfg.task.lr_schedule_step_size * cfg.task.steps_per_epoch) // cfg.task.grad_accumulation_steps, gamma=cfg.task.lr_schedule_gamma)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=cfg.task.lr_warmup_steps, after_scheduler=scheduler_steplr)

    # Detecting last checkpoint.
    if cfg.task.resume:
        checkpoint_path = os.path.join(cfg.task.log_dir, cfg.task.resume, 'ckpt.pth.tar')
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(checkpoint_path, map_location=loc, weights_only=False)
            cfg.task.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc {})".format(checkpoint_path, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True

    # Prepare Dataset
    start_time = perf_counter()
    #train_data, val_data, test_data, id_list = load_wikiweb2m(cfg.task)
    df, train_idx, val_idx, test_idx = load_datasets(cfg.dataset.name, cfg.dataset.data_root)
    print(f'Loading datasets done: {perf_counter()-start_time}')
    start_time = perf_counter()
    train_dataset = Datasets(cfg, df, train_idx, tokenizer)
    val_dataset = Datasets(cfg, df, val_idx, tokenizer)
    test_dataset = Datasets(cfg, df, test_idx, tokenizer)
    print(f'Initialize datasets: {perf_counter()-start_time}')
    print(f'Training with {len(train_dataset)} examples, validating with {len(val_dataset)} examples, testing with {len(test_dataset)} examples.')

    # Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    
    # Dataloader
    start_time = perf_counter()
    if cfg.task.context == 'text_only':
        train_loader = DataLoader(train_dataset, batch_size=cfg.task.per_device_train_batch_size,
            shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=cfg.task.per_device_val_batch_size,
            shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=val_sampler)
        test_loader = DataLoader(test_dataset, batch_size=cfg.task.per_device_val_batch_size,
            shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.task.per_device_train_batch_size,
                shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=train_sampler, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=cfg.task.per_device_val_batch_size,
                shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=val_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg.task.per_device_val_batch_size,
                shuffle=False, num_workers=cfg.task.dataloader_num_workers, prefetch_factor=10, pin_memory=False, sampler=test_sampler, collate_fn=collate_fn)
    print(f'Initialize dataloaders: {perf_counter()-start_time}')

    if cfg.task.test:
        evaluate_loop(test_loader, model, tokenizer, cfg.task.start_epoch, cfg, run, prefix="test")
        return
    total_time = 0
    if torch.distributed.get_rank() % world_size == 0:
        epoch_iter = tqdm(range(cfg.task.start_epoch, cfg.task.epochs),
                        desc="Epochs",
                        total=(cfg.task.epochs - cfg.task.start_epoch),
                        leave=True)
    else:
        epoch_iter = range(cfg.task.start_epoch, cfg.task.epochs)

    for epoch in epoch_iter:
        start_time = time.time()
        if epoch == 0:
            evaluate_loop(val_loader, model, tokenizer, epoch-1, cfg, run)

        # train for one epoch
        train_sampler.set_epoch(epoch)
        train_loop(train_loader, model, tokenizer, optimizer, epoch, scheduler, cfg, run)

        # evaluate on validation set
        acc1 = evaluate_loop(val_loader, model, tokenizer, epoch, cfg, run)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if gpu % world_size == 0 and (is_best or epoch == 0):
            # Only save non-frozen parameters.
            stripped_state_dict = {
                k: v for k, v in model.state_dict().items() if
                ('.text_model' not in k and '.visual_model' not in k)
            }
            stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
            state = {
                'epoch': epoch,
                'best_acc1': acc1,
                'state_dict': stripped_state_dict,
                'optimizer' : optimizer.state_dict(),
            }
            if scheduler is not None:
                state['scheduler'] = scheduler.state_dict()
            print('=> save best val model ...', cfg.task.save_dir)
            torch.save(state, cfg.task.save_dir)
        epoch_time = time.time() - start_time
        total_time += epoch_time
        # Update tqdm postfix (optional) — rank0 only
        if torch.distributed.get_rank() % world_size == 0:
            epoch_iter.set_postfix({'epoch_time': f'{epoch_time:.1f}s', 'best_acc1': f'{best_acc1:.4f}'})
        print(f"Epoch {epoch} time: {epoch_time}s")
    print(f"Total time: {total_time}s")
    # Test
    checkpoint_path = os.path.join(log_dir, 'ckpt.pth.tar')
    print("=> loading best val checkpoint '{}'".format(checkpoint_path))
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(checkpoint_path, map_location=loc, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded best val checkpoint '{}'".format(checkpoint_path))
    evaluate_loop(test_loader, model, tokenizer, cfg.task.epochs, cfg, run, "test")

def train_loop(train_loader, model, tokenizer, optimizer, epoch, scheduler, cfg, run):
    """
    Train loop for one epoch.
    Args:
        train_loader (DataLoader): Training dataloader.
        model (nn.Module): Model to train.
        tokenizer (PreTrainedTokenizer): Tokenizer.
        optimizer (Optimizer): Optimizer to use.
        epoch (int): Current epoch.
        scheduler (Scheduler): Scheduler to use.
        args (Arguments): Arguments.
        run (wandb run): Wandb run.
    """
    gpu, world_size = dist.get_rank(), dist.get_world_size()
    ngpus_per_node = torch.cuda.device_count()

    # Metrics
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    forward_time = utils.AverageMeter('Forward', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')

    # Progress bar (rank0 only)
    if gpu == 0:
        batch_iter = tqdm(range(cfg.task.steps_per_epoch),
                        desc=f"Train Epoch {epoch}",
                        total=cfg.task.steps_per_epoch,
                        leave=False)
    else:
        batch_iter = range(cfg.task.steps_per_epoch)


    # Additional loss just to record the summary loss on decoder-only models
    if cfg.task.decoder_only:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    end = time.time()
    for i, batch in zip(batch_iter, train_loader):
        data_time.update(time.time() - end)
        batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items()}
        forward_start = time.time()
        use_dtype = torch.bfloat16 if cfg.task.bf16 else torch.float32
        
        with autocast(device_type='cuda', dtype=use_dtype, enabled=cfg.task.bf16):
            outputs = model(**batch)
            loss = outputs.loss

        forward_time.update(time.time() - forward_start)
        
        losses.update(loss.item(), batch["input_ids"].size(0))
        loss = loss / cfg.task.grad_accumulation_steps
        loss.backward()

        # Update weights every cfg.task.grad_accumulation_steps
        if ((i + 1) % cfg.task.grad_accumulation_steps == 0) or (i == cfg.task.steps_per_epoch - 1):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                if cfg.task.grad_clip > 2:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.task.grad_clip)
            optimizer.zero_grad()

            # Log metrics every update step
            actual_step = (epoch * cfg.task.steps_per_epoch + i + 1) // cfg.task.grad_accumulation_steps
            if actual_step == 1 or actual_step % cfg.task.print_freq == 0:
                losses.all_reduce()
                batch_time.all_reduce()
                data_time.all_reduce()
                forward_time.all_reduce()
                ex_per_sec = (cfg.task.per_device_train_batch_size / batch_time.avg) * ngpus_per_node

                # Log only on the first GPU
                if gpu % world_size == 0:
                    #progress.display(i + 1)
                    run.log({"train/loss": losses.avg}, step=actual_step)
                    run.log({"metrics/total_secs_per_batch": batch_time.avg}, step=actual_step)
                    run.log({"metrics/data_secs_per_batch": data_time.avg}, step=actual_step)
                    run.log({"metrics/total_secs_captioning": forward_time.avg}, step=actual_step)
                    run.log({"metrics/examples_per_sec": ex_per_sec}, step=actual_step)

                losses.reset()
                batch_time.reset()
                data_time.reset()
                forward_time.reset()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == cfg.task.steps_per_epoch - 1:
            break


def evaluate_loop(val_loader, model, tokenizer, epoch, cfg, run, prefix="val"):
    """
    Evaluate loop.
    Args:
        val_loader (DataLoader): Validation dataloader.
        model (nn.Module): Model to evaluate.
        tokenizer (PreTrainedTokenizer): Tokenizer.
        epoch (int): Current epoch.
        args (Arguments): Arguments.
        run (wandb run): Wandb run.
        prefix (str): Prefix to use for logging.   
    """
    tokenizer.padding_side='left'
    gpu, world_size = dist.get_rank(), dist.get_world_size()
    ngpus_per_node = torch.cuda.device_count()

    # Three metrics to evaluate summarization: BLEU, ROUGE, CIDEr
    bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3, 4]]
    rouge_scorer = ROUGEScore()
    cider_scorer = Cider()
    actual_step = ((epoch + 1) * cfg.task.steps_per_epoch) // cfg.task.grad_accumulation_steps

    batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.AVERAGE)
    losses = utils.AverageMeter('Loss', ':.4e', utils.Summary.AVERAGE)
    bleu1 = utils.AverageMeter('BLEU@1', ':6.2f', utils.Summary.AVERAGE)
    bleu2 = utils.AverageMeter('BLEU@2', ':6.2f', utils.Summary.AVERAGE)
    bleu3 = utils.AverageMeter('BLEU@3', ':6.2f', utils.Summary.AVERAGE)
    bleu4 = utils.AverageMeter('BLEU@4', ':6.2f', utils.Summary.AVERAGE)
    rouge1 = utils.AverageMeter('ROUGE@1', ':6.2f', utils.Summary.AVERAGE)
    rouge2 = utils.AverageMeter('ROUGE@2', ':6.2f', utils.Summary.AVERAGE)
    rougeL = utils.AverageMeter('ROUGE@L', ':6.2f', utils.Summary.AVERAGE)
    rougeLsum = utils.AverageMeter('ROUGE@Lsum', ':6.2f', utils.Summary.AVERAGE)
    cider = utils.AverageMeter('CIDER', ':6.2f', utils.Summary.AVERAGE)

    if prefix == 'test':
        # Test phase: Must run through the entire DataLoader, cannot truncate
        max_steps = len(val_loader)
    else:
        # Validation phase: Run only the number of steps specified by parameters (e.g., 50) for speed, but cannot exceed total length
        max_steps = min(cfg.task.val_steps_per_epoch, len(val_loader))

    if gpu % world_size == 0:
        batch_iter = tqdm(range(max_steps),
                             desc=f"{prefix.capitalize()} Epoch {epoch}",
                             total=max_steps,
                             mininterval=1.0,
                             leave=False)
    else:
        batch_iter = range(max_steps)

    # Additional loss just to record the summary loss on decoder-only models
    if cfg.task.decoder_only:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    # Switch to evaluate mode
    model.eval()
        
    with torch.no_grad():
        end = time.time()
        all_generated_captions = []
        all_gt_captions = []
        max_to_display = 10

        for i, batch in zip(batch_iter, val_loader):
            batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            if cfg.task.decoder_only:
                # Only consider loss on reference summary just like encoder-decoder models
                # To maintain consistency in Loss evaluation, outputs.loss can be used directly here, or keep the original logic
                # But considering we mainly care about Metric, taking outputs.loss directly here is safest
                loss = outputs.loss
            else:
                labels = batch['labels']
                loss = outputs.loss
            losses.update(loss.item(), batch["input_ids"].size(0))

            # ==========================
            # 1. Unified extraction of multimodal parameters
            # ==========================
            gen_kwargs = {
                "images": batch.get("images"),
                "image_positions": batch.get("image_positions"),
                "neighbor_input_ids": batch.get("neighbor_input_ids"),
                "neighbor_attention_mask": batch.get("neighbor_attention_mask"),
                "neighbor_pos_ids": batch.get("neighbor_pos_ids"),
                "text_locations": batch.get("text_locations"),
                "neighbor_images": batch.get("neighbor_images"),
                "neighbor_images_pos_ids": batch.get("neighbor_images_pos_ids"),
                "image_locations": batch.get("image_locations"),
                "lpe": batch.get("lpe"),
                "graph": batch.get("graph"),
            }
            # Filter out None parameters
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            # ==========================
            # 2. Branch processing generation
            # ==========================
            if cfg.task.decoder_only:
                # === OPT (Decoder-Only) ===
                prompt_ids = batch["input_ids"][:, :cfg.task.max_input_length].contiguous()
                prompt_mask = batch["attention_mask"][:, :cfg.task.max_input_length].contiguous()

                generated_ids = model.module.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=cfg.task.max_output_length,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True,  # Fix generation not stopping
                    repetition_penalty=1.5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_kwargs
                )
                
                if generated_ids.shape[1] > cfg.task.max_input_length:
                    generated_ids = generated_ids[:, cfg.task.max_input_length:].contiguous()
            else:
                # === T5 (Encoder-Decoder) ===
                enc_ids = batch["input_ids"][:, :cfg.task.max_input_length].contiguous()
                enc_mask = batch["attention_mask"][:, :cfg.task.max_input_length].contiguous()

                generated_ids = model.module.generate(
                    input_ids=enc_ids,
                    attention_mask=enc_mask,
                    max_new_tokens=cfg.task.max_output_length,
                    do_sample=False,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=2,
                    length_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_kwargs 
                )
            
            generated_ids = torch.where(generated_ids == tokenizer.pad_token_id, torch.tensor(tokenizer.eos_token_id, device=generated_ids.device), generated_ids)

            all_generated_ids = [torch.zeros_like(generated_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_generated_ids, generated_ids)
            all_generated_ids[dist.get_rank()] = generated_ids
            generated_ids = torch.cat(all_generated_ids)

            # tgt_tokens = batch['labels'][:, cfg.task.max_input_length:].contiguous()
            if cfg.task.decoder_only:
                # OPT: Labels are Input + Output, need to cut off Input part
                tgt_tokens = batch['labels'][:, cfg.task.max_input_length:].contiguous()
            else:
                # T5: Labels are Output themselves, no slicing needed
                tgt_tokens = batch['labels']
            all_tgt_tokens = [torch.zeros_like(tgt_tokens) for _ in range(dist.get_world_size())]
            dist.all_gather(all_tgt_tokens, tgt_tokens)
            all_tgt_tokens[dist.get_rank()] = tgt_tokens
            all_tgt_tokens = torch.cat(all_tgt_tokens)


            all_tgt_tokens[all_tgt_tokens == -100] = tokenizer.pad_token_id
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gt_captions = tokenizer.batch_decode(all_tgt_tokens, skip_special_tokens=True)

            for cap_i in range(len(generated_captions)):
                all_generated_captions.append(generated_captions[cap_i])
                all_gt_captions.append([gt_captions[cap_i]])

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if gpu % world_size == 0:
                 batch_iter.set_postfix({"Loss": f"{losses.avg:.4e}", "Time": f"{batch_time.avg:.3f}s"})

            #if i == args.val_steps_per_epoch - 1:
            #    break

        if gpu % world_size == 0:
            print('=' * 30)
            print(f'Computing BLEU with {len(all_generated_captions)} generated captions and {len(all_gt_captions)} groundtruth captions.')
            for cap_i, cap in enumerate(all_generated_captions[:max_to_display]):
                print(f'{cap_i}) {cap}')
            print('=' * 30)
            print('Real samples:')
            for cap_i, cap in enumerate(all_gt_captions[:max_to_display]):
                print(f'{cap_i}) {cap}')
            print('=' * 30)

        bleu1_score = bleu_scorers[0](all_generated_captions, all_gt_captions)
        bleu1.update(bleu1_score, 1)
        bleu2_score = bleu_scorers[1](all_generated_captions, all_gt_captions)
        bleu2.update(bleu2_score, 1)
        bleu3_score = bleu_scorers[2](all_generated_captions, all_gt_captions)
        bleu3.update(bleu3_score, 1)
        bleu4_score = bleu_scorers[3](all_generated_captions, all_gt_captions)
        bleu4.update(bleu4_score, 1)

        rouge_scores = rouge_scorer(all_generated_captions, all_gt_captions)
        rouge1.update(rouge_scores['rouge1_fmeasure'], 1)
        rouge2.update(rouge_scores['rouge2_fmeasure'], 1)
        rougeL.update(rouge_scores['rougeL_fmeasure'], 1)
        rougeLsum.update(rouge_scores['rougeLsum_fmeasure'], 1)

        cands = {idx: [pred] for idx, pred in enumerate(all_generated_captions)}
        refs = {idx: [label] for idx, label in enumerate(all_gt_captions)}
        cider_scores, _ = cider_scorer.compute_score(refs, cands)
        cider.update(cider_scores, 1)

    batch_time.all_reduce()
    losses.all_reduce()
    bleu1.all_reduce()
    bleu2.all_reduce()
    bleu3.all_reduce()
    bleu4.all_reduce()
    rouge1.all_reduce()
    rouge2.all_reduce()
    rougeL.all_reduce()
    rougeLsum.all_reduce()
    cider.all_reduce()

    if gpu % world_size == 0:
        print("BLEU", bleu1.avg, bleu2.avg, bleu3.avg, bleu4.avg)
        print("ROUGE", rouge1.avg, rouge2.avg, rougeL.avg, rougeLsum.avg)
        print("CIDER", cider.avg)

        run.log({f"{prefix}/total_secs_per_batch": batch_time.avg}, step=actual_step)
        run.log({f"{prefix}/loss": losses.avg}, step=actual_step)
        run.log({f"{prefix}/bleu1": bleu1.avg}, step=actual_step)
        run.log({f"{prefix}/bleu2": bleu2.avg}, step=actual_step)
        run.log({f"{prefix}/bleu3": bleu3.avg}, step=actual_step)
        run.log({f"{prefix}/bleu4": bleu4.avg}, step=actual_step)
        run.log({f"{prefix}/rouge1": rouge1.avg}, step=actual_step)
        run.log({f"{prefix}/rouge2": rouge2.avg}, step=actual_step)
        run.log({f"{prefix}/rougeL": rougeL.avg}, step=actual_step)
        run.log({f"{prefix}/rougeLsum": rougeLsum.avg}, step=actual_step)
        run.log({f"{prefix}/cider": cider.avg}, step=actual_step)

    return bleu4.avg