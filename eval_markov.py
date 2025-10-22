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


def markov_true_logp(markov_table, x_str, y_str):
    logp = 0.0
    context = x_str[-4:]  # initial context
    for ch in y_str:
        prob = markov_table[context][ch]
        logp -= math.log(prob + 1e-12)
        context = context[1:] + ch
    return logp

def chunked_sums(x, num_chunks):
    assert len(x) % num_chunks == 0, "Tensor size must be divisible by num_chunks"
    chunk_size = len(x) // num_chunks
    sums = [x[i * chunk_size:(i + 1) * chunk_size].sum() for i in range(num_chunks)]
    return torch.tensor(sums)

#marginal liklihood can be also estimated through this function.
def batch_estimate_log_py_given_x(model, x_ids, y_ids, mask_token_id, MC_NUM=1000, mc_batch_size=128, device='cuda', prefix='y_given_x'):
    '''
    x_ids = [b, PROMPT_LENGTH]
    y_ids = [b, SEQUENCE_LENGTH]
    MC_NUM -> # of monte-carlo
    
    estimate batch of sequences within batch of "monte carlo" samples
    '''
    model.eval()
    assert x_ids.dim() == 2 and y_ids.dim() == 2, "only matrix format of x_ids, y_ids are allowed"
    seq_batch = torch.cat([x_ids, y_ids], dim=1).to(device)
    seq_batch_size = seq_batch.shape[0]
    
    seq_nll = torch.zeros(seq_batch_size, device='cpu')
    
    seq_len = y_ids.shape[1]
    all_subset, all_prob, HL = all_subset_indicators_and_weights(seq_len, device='cpu')
    
    num_blocks = math.ceil(MC_NUM / mc_batch_size)
    for b in range(num_blocks):
        size = min(mc_batch_size, MC_NUM - b * mc_batch_size)
        idxs = np.random.choice(len(all_subset), size=size, p=np.array(all_prob))
        I = all_subset[idxs].to(device=device, dtype=torch.bool)
        I = I.repeat(seq_batch_size, 1)
        
        I = torch.cat([torch.zeros(I.shape[0], x_ids.shape[1], device=device, dtype=bool), I], dim=1) \
            if prefix == 'y_given_x' else torch.cat([I, torch.zeros(I.shape[0], x_ids.shape[1], device=device, dtype=bool)], dim=1)
        
        #concat mask of prompt
        inputs = seq_batch.repeat_interleave(size, dim=0)
        inputs = inputs.masked_fill(I, mask_token_id)
        inputs = inputs.to(torch.long)
        
        log_probs = model(inputs)  #be aware that input size dim == batch_size * block_size (named as 'size')
        
        target = seq_batch.repeat_interleave(size, dim=0)
        chosen = log_probs.gather(dim=2, index=target.unsqueeze(-1)).squeeze(-1)
        tot_loss = -chosen.masked_select(I).to(torch.float64)
        tot_loss = chunked_sums(tot_loss, seq_batch_size)
        seq_nll += tot_loss
        
    return (seq_nll / MC_NUM * HL).to(device='cpu')


@torch.no_grad()
def main(ckpt_dir):
    PROMPT_LEN, RESP_LENGTH = 16, 16
    
    cfg_path = './config/config_markov.yaml'
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model
    model = RADD(cfg).to(device)
    loaded_state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(loaded_state['model'])
    model.eval()
    
    #tokenizer
    tokenizer = DNATokenizer()
    token_dim = model.config.tokens + 1 
    msk_tok_id = token_dim - 1


    print("Loading ground truth distribution...")
    with open('./data/markov_data_table.json') as f:
        markov_table = json.load(f)
    
    #precomputing data
    file_path = './data/markov_eval_data.txt'
    with open(file_path, "r") as f:
        lines = f.readlines()
    sequence = ''.join([line.strip().upper() for line in lines])
    sequence = ''.join([c for c in sequence if c in {'A', 'T', 'G', 'C'}])  # filter invalid

    VOCAB = ['A', 'T', 'G', 'C']
    data_lst = []
    for _ in range(1000):
        std_idx = np.random.randint(0, len(sequence) - PROMPT_LEN)
        x_str = sequence[std_idx:std_idx + PROMPT_LEN]
        context = x_str[-4:]  # initial context
        y_str = ''

        for _ in range(RESP_LENGTH):
            next_ch = random.choices(VOCAB, weights=[markov_table[context][c] for c in VOCAB])[0]
            y_str += next_ch
            context = context[1:] + next_ch


        tot_ids = tokenizer.encode(x_str + y_str) 
        x_ids, y_ids = tot_ids[:PROMPT_LEN], tot_ids[PROMPT_LEN:]
        true_logp = markov_true_logp(markov_table, x_str, y_str)
        
        data = {
            'x_ids': x_ids,
            'y_ids': y_ids,
            'true_logp': true_logp,
            'x_str': x_str,
            'y_str': y_str,
        }
        
        data_lst.append(data)
    

    markov_data_lst = data_lst[:64]
    data_len = len(markov_data_lst) #total data to evaluate
    
    x_ids_batch = torch.zeros(data_len, PROMPT_LEN, dtype=torch.long)
    y_ids_batch = torch.zeros(data_len, RESP_LENGTH, dtype=torch.long) 
    true_logp_lst = torch.zeros(data_len)
    model_logp_lst = torch.zeros(data_len)
    
    
    seq_lst = []
    for i, data in enumerate(markov_data_lst):
        x_ids_batch[i] = torch.tensor(data['x_ids'], dtype=torch.long)
        y_ids_batch[i] = torch.tensor(data['y_ids'], dtype=torch.long)
        true_logp_lst[i] = torch.tensor(data['true_logp'], dtype=torch.float64)
        seq_lst.append(data['x_str'][:3]+' ...')


    batch_quota = 2 ** 13
    model_batch_size = 2 ** 5
    mc_batch_size = batch_quota // model_batch_size
    MC_NUM = 2**13

    model_logp_lst = torch.zeros(data_len)
    data_blocks = math.ceil(data_len / model_batch_size)
    for i in tqdm(range(data_blocks)):
        st = i * model_batch_size
        ed = min((i + 1) * model_batch_size, data_len)
        
        x_ids, y_ids = x_ids_batch[st:ed], y_ids_batch[st:ed]
        model_cond_nll = batch_estimate_log_py_given_x(model, x_ids, y_ids, msk_tok_id, MC_NUM=MC_NUM, mc_batch_size = mc_batch_size)
        model_logp_lst[st:ed] = model_cond_nll
        
    true_p = true_logp_lst
    model_direct = model_logp_lst
    
    #generating figs
    plt.figure(figsize=(6,4))
    plt.plot(true_p.cpu().numpy(), color='black', linewidth=1.6, label='True NLL')
    plt.plot(model_direct.cpu().numpy(),  color='#EA5455', linewidth=1.6, label='Estimated NLL')
    plt.yticks(fontsize=14)
    plt.minorticks_off()
    plt.tick_params(top=False)
    plt.tick_params(bottom=False)
    k = 4  # show real sequence every k positions
    x = range(len(true_p))
    
    plt.xticks(ticks=x)
    ax = plt.gca()

    ax.set_xticklabels([])
    for i in x:
        label = seq_lst[i] if i % k == 0 or i == x[-1] else "."
        rotation = 85 if label != ".,," else 0
        fontsize = 12 if label != "." else 8
        ax.text(
            i, -0.3, label,  # position slightly below x-axis (tweak -1.5 if needed)
            ha='right', va='top',
            rotation=rotation,
            fontsize=fontsize,
            clip_on=False,
        )

    plt.vlines(x, ymin=0, ymax=true_p.cpu().numpy(), color='#B6B09F', linestyle='--', linewidth=0.8, alpha=0.6)   
    plt.ylim(0, 30)
    plt.legend(fontsize=14,
                fancybox=False,    # rounded box
                shadow=True,      # drop-shadow
                framealpha=0.8, 
                loc='upper left',
                )
    
    plt.savefig(f'./figures/markov_plot.png', dpi=300) 
    
    
        
if __name__ == "__main__":
    #### Set the checkpoint directory here #####
    # ckpt_dir = 'checkpoints/markov/checkpoint_4th_exp_30000.pth'
    ckpt_dir = None
    #############################################
    assert ckpt_dir is not None, "Please provide checkpoint directory manually e.g) ./checkpoints/markov/checkpoint_4th_exp_30000. Remove 'None'"
    main(ckpt_dir)