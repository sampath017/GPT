import os
from pathlib import Path
import torch
import tiktoken

# Paths
data_path = Path("../data/shakespear.txt")
tokens_path = data_path.parent / "tokens.pt"

logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)
model_checkpoint_path = logs_path / 'model_checkpoint.pth'

enc = tiktoken.get_encoding("gpt2")
torch.set_float32_matmul_precision("high")

if (enc.n_vocab + 47) % 64 == 0:
    vocab_size = enc.n_vocab + 47
else:
    raise ValueError("Vocabulary size must be divisible by 64.")

# DDP
is_ddp = int(os.environ.get('RANK', -1)) != -1
if is_ddp:
    assert torch.cuda.is_available(), "we need CUDA for DDP"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Config
wandb_mode = "offline"

config = {
    "project_name": "GPT3-small",
    "device": device,
    "model": {
        "name": "GPT",
        "num_embds": 768,
        "num_heads": 12,
        "head_size": 64,
        "num_blocks": 12,
        "dropout": 0.2
    },
    "training": {
        "max_steps": 50,
        "max_grad_norm": 1.0
    },
    "dataset": {
        "name": "tiny_shakespear",
        "vocab_size": vocab_size,
        "block_size": 1024,
        "total_batch_size": 2**19,  # In Tokens
        "batch_size": 64,
        "train_split": 0.7,
        "val_split": 0.3
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01
    }
}

assert (config["dataset"]["total_batch_size"] % (
    config["dataset"]["batch_size"]*config["dataset"]["block_size"]) == 0), "make sure total_batch_size is divisible by batch_size * block_size"
