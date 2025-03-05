project_name = "GPT"

model = {
    "name": "GPT",
    "num_embds": 768,
    "num_heads": 12,
    "num_blocks": 12,
}

dataset = {
    "name": "ShakespearDataset",
    "vocab_size": 50304,
    "context_size": 1024,
    "batch_size": 16,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 1

wandb_offline = False
