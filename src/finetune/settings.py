from pathlib import Path
import torch
import tiktoken
import torch.distributed as dist


# Paths
project_root_path = Path(__file__).parent.parent.parent.resolve()
data_root_path = project_root_path / "data"
ultrachat_200k_data_path = data_root_path / "ultrachat_200k"
logs_root_path = project_root_path / "logs"
logs_root_path.mkdir(exist_ok=True)
models_root_path = project_root_path / "models"
models_root_path.mkdir(exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

if (enc.n_vocab + 47) % 64 == 0:
    vocab_size = enc.n_vocab + 47
else:
    raise ValueError("Vocabulary size must be divisible by 64.")

# DDP
is_ddp_available = dist.is_available() and not dist.is_initialized()
try:
    if is_ddp_available:
        assert torch.cuda.is_available(), "we need CUDA for DDP so falling back to CPU"
        # this will fail in notebook
        # torchrun sets these env vars automatically
        dist.init_process_group(backend='nccl')

        ddp_global_rank = dist.get_rank()
        ddp_world_size = dist.get_world_size()
        ddp_local_rank = ddp_global_rank % ddp_world_size
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        ddp_master_process = (ddp_global_rank == 0)
    else:
        raise RuntimeError("Dist not available or already initialized")
except Exception as e:
    # vanilla, non-DDP run
    print(e)
    ddp_global_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    ddp_master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_ddp_available = False

if device == "cuda":
    torch.set_float32_matmul_precision("high")

# Config
wandb_mode = "online"  # or "offline"

config = {
    "project_name": "GPT3_124M",
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
        "max_steps": 3142,  # 1 epoch
        "val_interval": 200,  # TEMP
        "val_steps": 20,
        "max_grad_norm": 1.0
    },
    "dataset": {
        "name": "finewebedu_sample10B",
        "vocab_size": vocab_size,
        "block_size": 1024,
        "total_batch_size": 1024*32,  # In Tokens
        "batch_size": 8,
        "train_split": 0.7,
        "val_split": 0.3
    },
    "optimizer": {
        "name": "AdamW",
        "max_lr": 1e-3,
        "min_lr": 1e-3 * 0.5,
        "warmup_steps": 715,
        "betas": (0.9, 0.999),
        "weight_decay": 0.1,
        "eps": 1e-8
    }
}

assert (config["dataset"]["total_batch_size"] % (
    config["dataset"]["batch_size"]*config["dataset"]["block_size"]*ddp_world_size) == 0), "make sure total_batch_size is divisible by batch_size * block_size"
