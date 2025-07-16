import settings as s
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import torch.nn.functional as F
import torch
from utils import load_from_checkpoint, Trainer, ModelSummary
from dataset import DatasetLite, DataLoaderLite
from models import GPT
import os
import sys
sys.path.append("../src")

print(s.is_ddp)

if s.is_ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

# model = GPT().to(s.config["device"])
# model = torch.compile(model)
# model_summary = ModelSummary(model)
# model_summary.summary()

# dataset = DatasetLite()
# train_dataloader = DataLoaderLite(dataset, split="train")
# val_dataloader = DataLoaderLite(dataset, split="val")

# optimizer = model.configure_optimizers(
#     lr=s.config["optimizer"]["lr"],
#     weight_decay=s.config["optimizer"]["weight_decay"],
#     betas=s.config["optimizer"]["betas"]
# )

# trainer = Trainer(model, optimizer, {"train_dataloader":train_dataloader, "val_dataloader":val_dataloader})

# # 🧪 Training Loop with WandB
# try:
#     wandb.init(project="GPT-mini", config=s.config, dir=s.logs_path, mode=s.wandb_mode)

#     for step in range(s.config["training"]["max_steps"]):
#         train_loss, elapsed_time = trainer.train_step()
#         val_loss = trainer.val_step()

#         print(f"step {step:<3} | train_loss {train_loss:<5.2f} | val_loss {val_loss:<5.2f} | time {elapsed_time * 1000:<4.2f} ms")
#         wandb.log({"train_loss": train_loss, "val_loss": val_loss})

#         if step % 10 == 0:
#             # Save for every 10 steps
#             checkpoint = {
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'val_loss': val_loss
#             }

#             torch.save(checkpoint, s.model_checkpoint_path)
# except KeyboardInterrupt:
#     print("Stopping Run!")
# finally:
#     wandb.log_model(s.model_checkpoint_path)
#     wandb.finish()
