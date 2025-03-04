project_name = "GPT"

model = {
    "name": "GPT",
    "num_embds": 768,
    "head_size": 64,
    "num_heads": 12,
    "num_blocks": 12,
    "dropout": 0.0
}

dataset = {
    "name": "ShakespearDataset",
    "context_size": 1024,
    "batch_size": 4,
    "train_split": 0.7,
    "val_split": 0.3
}

max_epochs = 6

optimizer = {
    "name": "AdamW",
    "weight_decay": 0.01
}

lr_scheduler = None
# lr_scheduler = {
#     "name": "OneCycleLR",
#     "max_lr": 0.01,
# }

transfer_learning = None
# transfer_learning = {
#     "freeze_fe": True,
#     "change_fc": True
# }

limit_batches = None
wandb_offline = False
test_run = False
fast_dev_run = False
