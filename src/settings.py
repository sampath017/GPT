from pathlib import Path
import torch

# data_path = Path("/home/jl_fs/shards")
data_path = Path("../data/shakespear.txt")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "project_name": "GPT-mini",
    "device": device,
    "model": {
        "name": "GPT",
        "num_embds": 384,
        "num_heads": 8,
        "head_size": 48,
        "num_blocks": 12,
        "dropout": 0.2
    },
    "training": {
        "max_steps": 5000,
        "eval_interval": 100,
        "eval_steps": 200
    },
    "dataset": {
        "name": "tiny_shakespear",
        "vocab_size": 65,
        "block_size": 256,
        "batch_size": 128,
        "train_split": 0.7,
        "val_split": 0.3
    }
}
