project_name = "GPT"

model = {
    "name": "BigramLanguageModel",
    "num_layers": 1,
}

dataset = {
    "name": "ShakespearDataset",
    "batch_size": 64,
    "train_split": 0.7,
    "val_split": 0.3,
    "context_size": 8
}

max_epochs = 50

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
test_run = True
wandb_offline = True
fast_dev_run = True
