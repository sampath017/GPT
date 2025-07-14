from pathlib import Path
import torch
import tiktoken

data_path = Path("../data/shakespear.txt")
tokens_path = data_path.parent / "tokens.pt"

logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
torch.set_float32_matmul_precision("high")

if (enc.n_vocab + 47) % 64 == 0:
    vocab_size = enc.n_vocab + 47
else:
    raise ValueError("Vocabulary size plus 47 must be divisible by 64.")


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
        "max_steps": 50,
    },
    "dataset": {
        "name": "tiny_shakespear",
        "vocab_size": vocab_size,
        "block_size": 1024,
        "batch_size": 64,
        "train_split": 0.7,
        "val_split": 0.3
    }
}
