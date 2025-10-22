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
# from lion_pytorch import Lion  # pip install lion-pytorch
from model.transformer import RADD  # assumes your RADD class exists
from sampling import DiffusionSampler
from model.ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast
import numpy as np
from datetime import datetime

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

class DNATokenizer:
    def __init__(self):
        self.vocab = ['A', 'T', 'G', 'C']
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, sequence):
        """Convert DNA string (e.g. 'ATGC') to list of indices."""
        return [self.stoi.get(ch, -1) for ch in sequence]

    def decode(self, indices):
        """Convert list of indices back to DNA string."""
        return ''.join([self.itos.get(i, '?') for i in indices])

    def batch_decode(self, tensor: torch.Tensor):
        """
        Decodes a 2D tensor into a list of strings.
        Accepts [seq_len, batch_size] or [batch_size, seq_len].
        """
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")

        # Auto-handle shape [batch_size, seq_len] by transposing
        if tensor.shape[0] < tensor.shape[1]:  # [seq_len, batch_size] is preferred
            tensor = tensor  # already correct
        else:
            tensor = tensor.T  # transpose if input was [batch_size, seq_len]

        seq_len, batch_size = tensor.shape
        return [self.decode(tensor[:, i].tolist()) for i in range(batch_size)]

    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, sequence):
        """Mimic Hugging Face interface: return dict with input_ids and attention_mask."""
        input_ids = self.encode(sequence)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask
        }


class DNADataset(Dataset):
    def __init__(self, tokenizer, block_size=8):
        self.tokenizer = tokenizer
        self.block_size = block_size

        file_path = './data/sequence_data.txt'
        with open(file_path, "r") as f:
            lines = f.readlines()
        sequence = ''.join([line.strip().upper() for line in lines])
        sequence = ''.join([c for c in sequence if c in {'A', 'T', 'G', 'C'}])  # filter invalid

        # Tokenize the entire sequence
        token_ids = tokenizer.encode(sequence)
        print(f"Tokenized sequence length: {len(token_ids) // block_size}")
        # Drop remainder
        n_tokens = (len(token_ids) // block_size) * block_size
        token_ids = token_ids[:n_tokens]

        # Reshape into [num_chunks, block_size]
        self.chunks = torch.tensor(token_ids, dtype=torch.long).view(-1, block_size)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask
        }



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
    ema = ExponentialMovingAverage(
        radd_model.parameters(), decay=cfg.training.ema)

    mprint(radd_model)
    mprint(f"EMA: {ema}")
    token_dim = cfg.tokens + 1

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
    
    tokenizer = DNATokenizer()
    train_set = DNADataset(tokenizer=tokenizer)
    train_sampler = None
    train_ds = cycle_loader(DataLoader(
        train_set,
        batch_size=cfg.training.batch_size // (cfg.ngpus * cfg.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    
    train_iter = iter(train_ds)
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, token_dim, True, optimize_fn, cfg.training.accum, cfg.training.loss_type)

    if cfg.training.snapshot_sampling:
        sampling_shape = (32, cfg.model.length)
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
                dir_name = os.path.join(checkpoint_dir, 'sequence')
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

                    file_name = os.path.join( sample_dir, f"sample_sequence_{step}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("="*200 +"\n")

if __name__ == '__main__':
    cfg_path = './config/config_sequence.yaml'
    cfg = OmegaConf.load(cfg_path)
    main(cfg)