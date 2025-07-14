from pathlib import Path
import torch
import tiktoken

data_path = Path("../data/shakespear.txt")
tokens_path = data_path.parent / "tokens.pt"

logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")

config = {
    "project_name": "GPT-mini",
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
        "max_steps": 5000,
    },
    "dataset": {
        "name": "tiny_shakespear",
        "vocab_size": enc.n_vocab,
        "block_size": 1024,
        "batch_size": 128,
        "train_split": 0.7,
        "val_split": 0.3
    }
}
