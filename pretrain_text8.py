import os
import math
import gc
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import noise_lib
from itertools import chain
import data
import losses
from model.transformer import RADD 
from sampling import DiffusionSampler
from model.ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast
import numpy as np
from datetime import datetime
import os

torch.backends.cudnn.benchmark = True

def get_timestamped_log_name(prefix="run_sequence"):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    return f"{prefix}_{timestamp}"


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

def main(cfg):    
    sample_dir = './samples'
    os.makedirs(sample_dir, exist_ok=True)
    
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_meta_dir = os.path.join(checkpoint_dir, 'checkpoint_meta', "checkpoint.pth")
    

    logpath = utils.generate_logpath()
    logger = utils.get_logger(logpath)
    run_name = get_timestamped_log_name()
        
    def mprint(msg):
        logger.info(msg)
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")
    
    radd_model = RADD(cfg).to(device)
    ema = ExponentialMovingAverage(radd_model.parameters(), decay=cfg.training.ema)

    mprint(radd_model)
    mprint(f"EMA: {ema}")
    token_dim = cfg.tokens + 1

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt_dir) 
    noise = noise_lib.get_noise(cfg).to(device)
    optimizer = losses.get_optimizer(cfg, chain(radd_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=radd_model, noise=noise, ema=ema, step=0) 
    
    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])
    
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, token_dim, True, optimize_fn, cfg.training.accum,cfg.training.loss_type)
    
    train_ds, eval_ds = data.get_dataloaders(cfg, distributed=False)
    train_iter = iter(train_ds)

    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampler = DiffusionSampler(cfg.sampling.predictor,radd_model,noise,sampling_shape,token_dim, strategy = 'direct', device = device)
    
    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")
    
    while state['step'] < num_train_steps + 1:
        step = state['step']
        batch = next(train_iter)['input_ids'].to(device)
        
        loss = train_step_fn(state, batch)
        
        if step != state['step']:
            
            if step % cfg.training.log_freq == 0:
                metric = loss
                mprint(f"Step {step}: train_loss {metric.item():.3f}")
            
            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                dir_name = os.path.join(checkpoint_dir, 'text8')
                os.makedirs(dir_name, exist_ok=True)
                utils.save_checkpoint(os.path.join(
                        dir_name, f'checkpoint_{step}.pth'), state)
                mprint(f"Checkpoint saved at step {step}.")
                
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    ema.store(radd_model.parameters())
                    ema.copy_to(radd_model.parameters())
                    sample = sampler.sample(cfg.sampling.steps)
                    ema.restore(radd_model.parameters())

                    sentences = tokenizer.batch_decode(sample)

                    file_name = os.path.join( sample_dir, f"sample_text8_{step}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("="*200 +"\n")

if __name__ == '__main__':
    cfg_path = './config/config_text8.yaml'
    cfg = OmegaConf.load(cfg_path)
    main(cfg)