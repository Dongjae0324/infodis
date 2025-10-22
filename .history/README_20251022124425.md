# Information Theoretic Discrete Diffusion

This repository contains the official implementation of our paper:  
**Information Theoretic Discrete Diffusion**.

We provide code for four experiments introduced in the paper.  
All experiments can be run on a **single NVIDIA L40S GPU (48GB VRAM)**.

---

## 1. Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml
conda activate infodis
```

---

## 2. Running Experiments

### [1] Detecting Out-of-Distribution Inputs

Refer to **Section 4.2 (Detecting Out-of-Distribution Inputs)** and **Figure 3** in the paper.

- The model (**RADD**) is trained on **`text8`**.
- Conditional NLL is evaluated on both **`text8`** and **GPT-generated text**.

#### Pre-training

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_text8.py
```

This will create checkpoints like:

```
checkpoints/text8/checkpoint_7501.pth
```

which indicates the model was trained for 7500 steps.

#### Evaluation

Edit the `ckpt_dir` in `eval_text8.py` to point to the desired checkpoint from "None", for instance:

```python
ckpt_dir = "checkpoints/text8/checkpoint_7501.pth"
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_text8.py
```

This will automatically produce figures in the `./figures` directory.

> The following experiments follow a similar procedure.

---

### [2] Toy Experiments

Refer to **Section 4.1 (Evaluating reliability of I-MDCE on toy dataset)** and **Figures 1 and 2**.

These experiments evaluate NLL on synthetic DNA sequence datasets.

#### Pre-training and Evaluation

- **128-sequence experiment:**

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_sequence.py
CUDA_VISIBLE_DEVICES=0 python eval_sequence.py
```

- **4th-order Markov experiment:**

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_markov.py
CUDA_VISIBLE_DEVICES=0 python eval_markov.py
```

Note. Manually update `ckpt_dir` in the evaluation scripts before running.

---

### [3] LLaDA Experiments

Refer to **Section 4.2 (Application to a Large-Scale Open-Source Model)** and **Figures 4 and 8**.

This experiment evaluates NLL using **LLaDA**, a recent open-source model, on the following datasets:
- `wikitext`
- `pretrain_zh`
- LLaMA-generated responses

#### Evaluation Example

```bash
CUDA_VISIBLE_DEVICES=0 python eval_llada.py --dataset wikitext
```
you can reduce the dataset size manually as our default evaluation file contains lots of data.
However, if you reduce the number of Monte Carlo estimation dramatically, the result won't be accurate.