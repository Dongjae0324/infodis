import os
import torch
from model import RADD
import utils
from model.ema import ExponentialMovingAverage
import noise_lib
from transformers import GPT2TokenizerFast
from omegaconf import OmegaConf
from sampling import DiffusionSampler
from typing import List, Any
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde


def harmonic_number(n):
    return sum(1.0 / i for i in range(1, n + 1))

def sample_masked_indices(L):
    u_values = list(range(1, L))
    probs = [1 / (L - u) for u in u_values]
    total = sum(probs)
    probs = [p / total for p in probs]
    u = random.choices(u_values, weights=probs, k=1)[0]
    return sorted(random.sample(range(L), L-u))

def sample_batch_mask(batch_size, L):
    indice_lst = []
    mask = torch.zeros(batch_size, L, dtype=torch.bool)
    for i in range(batch_size):
        u_values = list(range(1, L))
        probs = [1 / (L - u) for u in u_values]
        total = sum(probs)
        probs = [p / total for p in probs]
        u = random.choices(u_values, weights=probs, k=1)[0]
        mask[i, sorted(random.sample(range(L), L-u))] = True
        indice_lst.append(len(sorted(random.sample(range(L), L-u))))
    return mask, indice_lst

def group_means_cuda(arr: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    if arr.dim() != 1 or not arr.is_cuda:
        raise ValueError("arr must be a 1D CUDA tensor")
    if sum(sizes) != arr.numel():
        raise ValueError("sum(sizes) must equal length of arr")

    arr = arr.to(torch.float64)
    device = arr.device
    cumsum = torch.cumsum(arr, dim=0)
    sizes_t = torch.tensor(sizes, dtype=torch.long, device=device)
    ends = torch.cumsum(sizes_t, dim=0) - 1
    starts = ends - (sizes_t - 1)

    sum_ends = cumsum[ends]
    sum_starts_minus1 = torch.zeros_like(sum_ends)
    mask = starts > 0
    sum_starts_minus1[mask] = cumsum[starts[mask] - 1]

    chunk_sums = sum_ends - sum_starts_minus1
    means = chunk_sums / sizes_t.to(torch.float64)
    return means


@torch.no_grad()
def main(ckpt_dir):
    fig_path = './figures'
    os.makedirs(fig_path, exist_ok=True)
    cfg_path = './config/config_text8.yaml'
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #test_cfg
    x_len = cfg.model.length//2
    y_len = cfg.model.length - x_len
    
    #load model
    model = RADD(cfg).to(device)
    loaded_state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(loaded_state['model'])
    model.eval()
    
    #tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt_dir)
    token_dim = model.config.tokens + 1
    
    #selected 50,000 data from text8
    with open("./data/text8_eval_data.txt", "r", encoding="utf-8") as f:
        line = f.readlines() 
        train_data = line[0]
    
    with open("./data/text8_gpt_eval_data.txt", "r", encoding="utf-8") as f:
        line = f.readlines()
        gpt_data = line[0]
    
    text8_ids = tokenizer(train_data).input_ids 
    gpt_ids = tokenizer(gpt_data).input_ids
    
    #compute -log p(y|x)
    subset_len = (x_len + y_len)
    Hl = harmonic_number(y_len)
    msk_tok_id = token_dim - 1
    mc_num = 100
    
    num_samples = 500
    lst_text8 = torch.zeros(num_samples, device=device)
    lst_gpt = torch.zeros(num_samples, device=device)
    
    for i in tqdm(range(num_samples)):
        std_idx = np.random.randint(0, len(text8_ids) - subset_len)
        subset_text8_batch = torch.tensor(text8_ids[std_idx:std_idx+subset_len])\
                        .to(device).repeat(mc_num, 1)
        
        std_gpt_idx = np.random.randint(0, len(gpt_ids) - y_len)
        subset_gpt_batch = torch.tensor((text8_ids[std_idx:std_idx+x_len], gpt_ids[std_gpt_idx:std_gpt_idx+y_len])).flatten()\
                        .to(device).repeat(mc_num, 1)
        
        masked_text8_batch, masked_gpt_batch = subset_text8_batch.clone(), subset_gpt_batch.clone()
        y_masks, mask_indices = sample_batch_mask(mc_num, y_len)   
        target_text8, target_gpt = subset_text8_batch[:, x_len:][y_masks], subset_gpt_batch[:, x_len:][y_masks]
        masked_text8_batch[:, x_len:][y_masks] = msk_tok_id
        masked_gpt_batch[:, x_len:][y_masks] = msk_tok_id
        
        masked_batch = torch.cat([masked_text8_batch, masked_gpt_batch], dim=0)
        y_masks = torch.cat([y_masks, y_masks], dim=0)
        target = torch.cat([target_text8, target_gpt], dim=0)
        mask_indices = mask_indices + mask_indices


        logp = model(masked_batch)
        sel = logp[:, x_len:, :][y_masks, target]
        nll = -group_means_cuda(sel, mask_indices)

        lst_text8[i] = (Hl * nll[:mc_num].mean())
        lst_gpt[i] = (Hl * nll[mc_num:].mean())
        

    lst_text8 = lst_text8.cpu().numpy()
    lst_gpt = lst_gpt.cpu().numpy()
    
    all_data = np.hstack([lst_text8, lst_gpt])
    x_min, x_max = all_data.min(), all_data.max()
    x_grid = np.linspace(x_min, x_max, 500)

    # Build KDEs
    kde_text8 = gaussian_kde(lst_text8)
    kde_gpt = gaussian_kde(lst_gpt)

    # Evaluate densities
    dens_text8 = kde_text8(x_grid)
    dens_gpt = kde_gpt(x_grid)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.minorticks_off()
    # plt.xlabel("Negative Log likelihood")
    # plt.xlabel(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_grid, dens_text8, color='#384786', linewidth=2)
    plt.fill_between(x_grid, dens_text8, color='#384786', alpha=0.3, label='text8')

    plt.plot(x_grid, dens_gpt, color='#EA5455', linewidth=2)
    plt.fill_between(x_grid, dens_gpt, color='#EA5455', alpha=0.3, label='GPT generated')
    plt.tick_params(top=False)
    plt.tick_params(bottom=False)
    plt.legend(fontsize=16,
            fancybox=False,    # rounded box
            shadow=True,      # drop-shadow
            framealpha=0.8, 
            loc='upper left',
    )
    plt.tight_layout()
    fig_path = os.path.join(fig_path, 'text8_plot.png')
    plt.savefig(fig_path, dpi=300)



if __name__ == "__main__":
    ##### Set the checkpoint directory here #####
    # ckpt_dir = 'checkpoints/text8/checkpoint_7501.pth'
    ckpt_dir = None
    #############################################
    assert ckpt_dir is not None, "Please provide checkpoint directory manually like ./checkpoints/text8/checkpoint_4000.pth. Remove 'None'"
    main(ckpt_dir)