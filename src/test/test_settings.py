import os
from pathlib import Path
import torch
import tiktoken
import torch.distributed as dist

# Seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Paths
project_root_path = Path(__file__).parent.parent.parent
sample10B_data_path = Path("/home/jl_fs/sample10B_data")

logs_path = project_root_path / "logs"
logs_path.mkdir(exist_ok=True)
model_checkpoint_path = logs_path / 'model_checkpoint.pth'

enc = tiktoken.get_encoding("gpt2")
torch.set_float32_matmul_precision("high")

if (enc.n_vocab + 47) % 64 == 0:
    vocab_size = enc.n_vocab + 47
else:
    raise ValueError("Vocabulary size must be divisible by 64.")


# DDP
device = "cpu"
is_ddp_available = dist.is_available() and not dist.is_initialized()
try:
    # Attempt to init process group (works only under torchrun)
    if is_ddp_available:
        dist.init_process_group(backend='gloo')  # this will fail in notebook
        ddp_global_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
        ddp_local_rank = ddp_global_rank % ddp_world_size
        ddp_master_process = ddp_global_rank == 0
    else:
        raise RuntimeError("Dist not available or already initialized")
except Exception:
    # Fall back to CPU / non-DDP mode
    ddp_global_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    ddp_master_process = True

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
        "batch_size": 4,
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
    config["dataset"]["batch_size"]*config["dataset"]["block_size"]*ddp_world_size) == 0), "make sure total_batch_size is divisible by batch_size * block_size"
