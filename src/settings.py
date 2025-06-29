from pathlib import Path

# data_path = Path("/home/jl_fs/shards")
data_path = Path("../data/edu_fineweb1B/shards")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)

project_name = "GPT"

model = {
    "name": "GPT",
    "num_embds": 96,
    "num_heads": 4,
    "num_blocks": 4,
}


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
# max_steps = 19073
max_steps = 2
max_epochs = 10

dataset = {
    "name": "Fineweb1B",
    # "vocab_size": 50304, # optimized vocabsize
    "vocab_size": 50257,
    "context_size": 1024,
    "batch_size": 32,
    "total_batch_size": 524288,  # 2**19, ~0.5M, in number of tokens
    "train_split": 0.7,
    "val_split": 0.3
}

wandb_offline = False
