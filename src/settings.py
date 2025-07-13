from pathlib import Path

# data_path = Path("/home/jl_fs/shards")
data_path = Path("../data/shakespear.txt")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)

config = {
    "project_name": "GPT-mini",
    "model": {
        "name": "GPT",
        "num_embds": 32,
        "num_heads": 4,
        "head_size": 8,
        "num_blocks": 6
    },
    "training": {
        "max_steps": 5000,
        "eval_interval": 100,
        "eval_steps": 200
    },
    "dataset": {
        "name": "tiny_shakespear",
        "vocab_size": 65,
        "block_size": 8,
        "batch_size": 64,
        "train_split": 0.7,
        "val_split": 0.3
    }
}
