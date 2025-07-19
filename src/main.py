import settings as s
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import wandb
import torch.nn.functional as F
import torch
from utils import load_from_checkpoint, Trainer, ModelSummary
from dataset import DataLoaderLite
from models import GPT
import os
import sys
sys.path.append("../src")


# if ddp_master_process:
print(s.ddp_global_rank, s.ddp_local_rank, s.ddp_world_size, s.device)

model = GPT().to(s.config["device"])
model = torch.compile(model)
if s.is_ddp_available:
    model = DDP(model, device_ids=[s.ddp_local_rank])
    raw_model = model.module if s.is_ddp_available else model

if s.ddp_master_process:
    model_summary = ModelSummary(model)
    model_summary.summary()


train_dataloader = DataLoaderLite(split="train")
val_dataloader = DataLoaderLite(split="val")

optimizer = raw_model.configure_optimizers(
    lr=s.config["optimizer"]["lr"],
    weight_decay=s.config["optimizer"]["weight_decay"],
    betas=s.config["optimizer"]["betas"]
)  # type: ignore

trainer = Trainer(model, optimizer, {
                  "train_dataloader": train_dataloader, "val_dataloader": val_dataloader})

# 🧪 Training Loop with WandB
try:
    if s.ddp_master_process:
        wandb.init(project="GPT-mini", config=s.config,
                   dir=s.logs_path, mode=s.wandb_mode)

    for step in range(s.config["training"]["max_steps"]):
        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Train
        train_loss, norm = trainer.train_step()
        val_loss = trainer.val_step()

        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time = end_time - start_time # in seconds
        tokens_processed = train_dataloader.B * train_dataloader.T * s.grad_accum_steps * s.ddp_world_size
        tokens_per_sec = tokens_processed / elapsed_time

        if s.ddp_master_process:
            print(
                f"step {step:<3} | train_loss {train_loss:<5.2f} | val_loss {val_loss:<5.2f} | norm {norm:<5.2f} | time {elapsed_time * 1000:<4.2f} ms | tok/sec {tokens_per_sec:<5.2f}")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            if step % 10 == 0:
                # Save for every 10 steps
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

                torch.save(checkpoint, s.model_checkpoint_path)
except KeyboardInterrupt:
    print("Stopping Run!")
finally:
    if s.ddp_master_process:
        wandb.log_model(s.model_checkpoint_path)
        wandb.finish()

if s.is_ddp_available:
    destroy_process_group()
