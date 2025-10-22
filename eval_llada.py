import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
import numpy as np
from scipy.stats import entropy
from scipy.special import beta
from math import comb
import itertools
import math
from scipy.stats import gaussian_kde
import os


def harmonic_number(n):
    return sum(1.0 / i for i in range(1, n + 1))

def sample_mask_indices(length):
    u_values = list(range(1, length))
    probs = [1 / (length - u) for u in u_values]
    total = sum(probs)
    probs = [p / total for p in probs]
    u = random.choices(u_values, weights=probs, k=1)[0]
    return sorted(random.sample(range(length), length - u))

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

def chunked_sums(x, num_chunks):
    assert len(x) % num_chunks == 0, "Tensor size must be divisible by num_chunks"
    chunk_size = len(x) // num_chunks
    sums = [x[i * chunk_size:(i + 1) * chunk_size].sum() for i in range(num_chunks)]
    return torch.tensor(sums)

@torch.no_grad()
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
    HL = harmonic_number(seq_len)
    
    num_blocks = math.ceil(MC_NUM / mc_batch_size)
    for b in range(num_blocks):
        size = min(mc_batch_size, MC_NUM - b * mc_batch_size)
        I = sample_batch_mask(size, seq_len)[0]
        I = I.repeat(seq_batch_size, 1).to(device)
        I = torch.cat([torch.zeros(I.shape[0], x_ids.shape[1], device=device, dtype=bool), I], dim=1) \
            if prefix == 'y_given_x' else torch.cat([I, torch.zeros(I.shape[0], x_ids.shape[1], device=device, dtype=bool)], dim=1)

        #concat mask of prompt
        inputs = seq_batch.repeat_interleave(size, dim=0)
        inputs = inputs.masked_fill(I, mask_token_id)
        inputs = inputs.to(torch.long)
        logits = model(inputs).logits
        logp = F.log_softmax(logits, dim=-1)

        target = seq_batch.repeat_interleave(size, dim=0)
        chosen = logp.gather(dim=2, index=target.unsqueeze(-1)).squeeze(-1)
        tot_loss = -chosen.masked_select(I).to(torch.float64)
        tot_loss = chunked_sums(tot_loss, seq_batch_size)
        seq_nll += tot_loss
        
    return (seq_nll / MC_NUM * HL).to(device='cpu')



@torch.no_grad()
def compute_log_likelihood(model, x, y, mask_token_id, mc_samples=100):
    device = next(model.parameters()).device
    L = y.size(0)
    Hl = harmonic_number(L)
    x = x.to(device)
    y = y.to(device)

    losses = []
    for _ in range(mc_samples):
        masks = sample_mask_indices(L)
        y_masked = y.clone()
        y_masked[masks] = mask_token_id

        input_ids = torch.cat([x, y_masked], 0).unsqueeze(0).to(device)
        logits = model(input_ids).logits
        logp = F.log_softmax(logits, dim=-1)

        mask_tensor = torch.tensor([i in masks for i in range(L)], device=device)
        sel = logp[0, len(x):, :][mask_tensor, y[mask_tensor]]
        loss = -sel.sum()
        losses.append(loss.item())

    return Hl * sum(losses)/mc_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["zh", "wikitext"])
    parser.add_argument("--json_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    json_path = args.json_path or f"./data/{dataset}_llada.json"
    
    fig_path = './figures'
    os.makedirs(fig_path, exist_ok=True)
    
    # Load records
    with open(json_path, "r") as f:
        records = json.load(f)

    # Load tokenizer and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True
    )
    mask_id = 126336  # '<mask>' token ID

    y_scores, r_scores, z_scores = [], [], []

    batch_quota = 250
    model_batch_size = 25
    mc_batch_size = batch_quota // model_batch_size
    MC_NUM = 100
    SEQ_LEN = 128
    
    x_ids_batch, y_ids_batch, r_ids_batch, z_ids_batch =torch.zeros((len(records), SEQ_LEN//2), device=device, dtype=torch.long), \
                                                        torch.zeros((len(records), SEQ_LEN//2), device=device, dtype=torch.long), \
                                                        torch.zeros((len(records), SEQ_LEN//2), device=device, dtype=torch.long), \
                                                        torch.zeros((len(records), SEQ_LEN//2), device=device, dtype=torch.long)
    for i, record in enumerate(tqdm(records)):
        rec_x = record['x']
        rec_x_token = tokenizer(rec_x, return_tensors="pt", padding=True).input_ids
        rec_x_token = rec_x_token[-SEQ_LEN//2:]
        if len(rec_x_token[0]) < SEQ_LEN//2:
            print(f"Warning: {rec_x} is too short, padding with {rec_x_token[0][0].item()}")
            rec_x_token = torch.cat([torch.ones(SEQ_LEN//2-len(rec_x_token[0]), dtype=torch.long)*rec_x_token[0][0].item(), rec_x_token[0]])

        rec_yrx = [record['y'], record['r'], record['z']]
        rec_token = tokenizer(rec_yrx, return_tensors="pt", padding=True).input_ids
        rec_token = rec_token[:,:SEQ_LEN//2]
        
        x_ids_batch[i] = rec_x_token.squeeze(0)
        y_ids_batch[i] = rec_token[0]
        r_ids_batch[i] = rec_token[1]
        z_ids_batch[i] = rec_token[2]
        
    y_scores = torch.zeros((len(records),))
    r_scores = torch.zeros((len(records),))
    z_scores = torch.zeros((len(records),))
    
    record_blocks = math.ceil(len(records) / model_batch_size)
    for i in tqdm(range(record_blocks)):
        st = i * model_batch_size
        ed = min((i + 1) * model_batch_size, len(records))
        
        x_ids = x_ids_batch[st:ed].to(device)
        y_ids = y_ids_batch[st:ed].to(device)
        r_ids = y_ids_batch[st:ed].to(device)
        z_ids = z_ids_batch[st:ed].to(device)         
        
        y_ll = batch_estimate_log_py_given_x(model, x_ids, y_ids, mask_id, MC_NUM=MC_NUM, mc_batch_size = mc_batch_size)
        # r_ll = batch_estimate_log_py_given_x(model, x_ids, r_ids, mask_id, MC_NUM=MC_NUM, mc_batch_size = mc_batch_size) #not used in paper
        z_ll = batch_estimate_log_py_given_x(model, x_ids, z_ids, mask_id, MC_NUM=MC_NUM, mc_batch_size = mc_batch_size)
        
        y_scores[st:ed] = y_ll
        # r_scores[st:ed] = r_ll 
        z_scores[st:ed] = z_ll
        

    lst_origin = y_scores.cpu().numpy()
    lst_llama = z_scores.cpu().numpy()

    all_data = np.hstack([lst_origin, lst_llama])
    x_min, x_max = all_data.min(), all_data.max()
    x_grid = np.linspace(x_min, x_max, 500)

    # Build KDEs
    kde_origin = gaussian_kde(lst_origin)
    kde_llama = gaussian_kde(lst_llama)

    # Evaluate densities
    dens_text8 = kde_origin(x_grid)
    dense_llama = kde_llama(x_grid)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.minorticks_off()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_grid, dens_text8, color='#384786', linewidth=2)
    plt.fill_between(x_grid, dens_text8, color='#384786', alpha=0.3, label=f'{dataset}')

    plt.plot(x_grid, dense_llama, color='#FFD460', linewidth=2)
    plt.fill_between(x_grid, dense_llama, color='#FFD460', alpha=0.3, label='LLaMA generated')
    plt.tick_params(top=False)
    plt.tick_params(bottom=False)
    plt.legend(fontsize=16,
            fancybox=False,    # rounded box
            shadow=True,      # drop-shadow
            framealpha=0.8, 
            loc='upper right',
    )
    plt.tight_layout()
    fig_path = os.path.join(fig_path, f'llada_{dataset}_plot.png')
    plt.savefig(fig_path, dpi=500)


if __name__ == "__main__":
    main()
