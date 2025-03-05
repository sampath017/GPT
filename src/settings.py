project_name = "GPT"

model = {
    "name": "GPT",
    "num_embds": 768,
    "num_heads": 12,
    "num_blocks": 12,
}

dataset = {
    "name": "ShakespearDataset",
    "context_size": 1024,
    "batch_size": 16,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 6

wandb_offline = False
