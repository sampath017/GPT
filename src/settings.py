from pathlib import Path

data_path = Path("/home/jl_fs/shards")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)

project_name = "GPT"

model = {
    "name": "GPT",
    "num_embds": 768,
    "num_heads": 12,
    "num_blocks": 12,
}

dataset = {
    "name": "Fineweb1B",
    "vocab_size": 50304,
    "context_size": 1024,
    "batch_size": 32,
    "total_batch_size": 524288,  # 2**19, ~0.5M, in number of tokens
    "train_split": 0.7,
    "val_split": 0.3
}

wandb_offline = False
