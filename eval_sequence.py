import os
import torch
from model import RADD
import utils
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast
from omegaconf import OmegaConf
from sampling import DiffusionSampler
from typing import List, Any
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import beta
from math import comb
import itertools
import json
from tqdm import tqdm
import math

np.random.seed(42)
torch.manual_seed(42)

# === UTILITY FUNCTIONS ===
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

def all_subset_indicators_and_weights(L, device="cuda"):
    # 1) compute β-weight numerator for each subset size k
    num_beta = [beta(L - k, k + 1) for k in range(L)]
    # 2) normalizing constant H_L
    H_L = sum(comb(L, k) * num_beta[k] for k in range(L))

    indicators = []
    weights    = []

    # 3) enumerate all subsets of size k
    for k in range(L):
        w_k = num_beta[k] / H_L
        for I in itertools.combinations(range(L), k):
            # build 0/1 indicator vector
            row = np.ones(L, dtype=np.int8)
            row[list(I)] = 0
            indicators.append(row)
            weights.append(w_k)

    # stack into array / tensor
    I_np = np.stack(indicators, axis=0)            # shape: [num_subsets, L]
    w_np = np.array(weights, dtype=np.float32)      # shape: [num_subsets]
    
    I_torch = torch.tensor(I_np, device=device)          # shape: [num_subsets, L]
    w_torch = torch.tensor(w_np, device=device)          # shape: [num_subsets]
    
    return I_torch, w_torch, H_L


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

@torch.no_grad()
def main(ckpt_dir):
    fig_path = './figures'
    os.makedirs(fig_path, exist_ok=True)
    # cfg
    cfg_path = './config/config_sequence.yaml'
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MC_NUM = 2**15
    batch_num = 32768
    
    #model
    model = RADD(cfg).to(device)
    loaded_state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(loaded_state['model'])
    model.eval()
    
    #tokenizer
    tokenizer = DNATokenizer()
    token_dim = model.config.tokens + 1
    msk_tok_id = token_dim - 1
    
    #load all sets
    all_subset, all_prob, HL = all_subset_indicators_and_weights(cfg.model.length, device='cpu')
    
    print("Loading ground truth distribution...")
    with open('./data/sequence_eval.json') as f:
        true_dist = json.load(f)
    
    @torch.no_grad()
    def estimate_neg_log_mc(model, seq_tensor):
        """
        Monte Carlo estimate of negative log likelihood for one sequence.
        """
        model.eval()
        seq_tensor = seq_tensor.to(device)

        total_loss = 0.0
        num_blocks = math.ceil(MC_NUM / batch_num)

        for b in range(num_blocks):
            size = min(batch_num, MC_NUM - b * batch_num)
            idxs = np.random.choice(len(all_subset), size=size, p=np.array(all_prob))
            I = all_subset[idxs].to(device=device, dtype=torch.bool)   # shape (size, L)

            inputs = seq_tensor.expand(size, -1).clone()               # (size, L)
            inputs = inputs.masked_fill(I, msk_tok_id)
            log_probs = model(inputs)                                  # already log-softmax

            target = seq_tensor.unsqueeze(0).expand(size, -1)          # (size, L)
            chosen = log_probs.gather(dim=2, index=target.unsqueeze(-1)).squeeze(-1)
            loss = -chosen.masked_select(I).sum(dtype=torch.float64)  # scalar 64-bit
            total_loss += loss

        # average over Monte Carlo draws and rescale
        return (total_loss / MC_NUM * HL).item()

          
    tot_lst = list(true_dist.keys())
    seqs = tot_lst
    seq_lst = [] 
    
    log_p_true = []
    log_p_model = []  
    
    print("Estimating -log p for 128 sequences via MC...")
    for seq in tqdm(seqs):
        st = tokenizer.encode(seq)
        st = torch.tensor(st, device=device)
        log_p_true.append(-math.log(true_dist[seq]))
        log_p_model.append(estimate_neg_log_mc(model, st))
        seq_lst.append(seq[:])
    log_p_true = torch.tensor(log_p_true)
    log_p_model = torch.tensor(log_p_model)
    
    
    plt.figure(figsize=(17,4))
    plt.plot(log_p_true.cpu().numpy(), color='black', linewidth=1.8, label='True NLL')
    plt.plot(log_p_model.cpu().numpy(), color='#EA5455', linewidth=1.8, label=r'Estimated NLL')
    x = range(len(log_p_true))
    plt.yticks(fontsize=14)
    plt.minorticks_off()
    plt.tick_params(top=False)
    plt.tick_params(bottom=False)
    plt.vlines(x, ymin=3, ymax=log_p_true.cpu().numpy(), color='#B6B09F', linestyle='--', linewidth=0.9, alpha=0.6) 
    plt.ylim(3, 7.5)
    
    k = 1  # show real sequence every k positions
    x = range(len(log_p_true))
    
    plt.xticks(x, seq_lst, fontsize=8, rotation=80)
    plt.xlim(-1, len(log_p_true)) 
    plt.legend(fontsize=16,
                fancybox=False,    # rounded box
                shadow=True,      # drop-shadow
                framealpha=0.8, 
                loc='upper left',
                )
    
    plt.tight_layout()
    fig_path = os.path.join(fig_path, 'sequence_plot.png')
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')        
 
if __name__ == "__main__":
    ##### Set the checkpoint directory here #####
    # ckpt_dir = 'checkpoints/sequence/checkpoint_40000.pth'
    ckpt_dir = None
    #############################################
    assert ckpt_dir is not None, "Please provide checkpoint directory manually like ./checkpoints/sequence/checkpoint_4000.pth. Remove 'None'"
    main(ckpt_dir)