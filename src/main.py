import sys
import settings as s
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import wandb
import torch.nn.functional as F
import torch
from utils import Trainer, ModelSummary, generate
from dataset import DataLoaderLite
from models import GPT
import time
from hellaswag import evaluate
from transformers import GPT2LMHeadModel

# if ddp_master_process:
print(s.ddp_global_rank, s.ddp_local_rank, s.ddp_world_size, s.device)


gpt2_xl_hellaswag_acc_path = s.logs_path / "gpt2_xl_hellaswag_acc.pt"
gpt2_hellaswag_acc_path = s.logs_path / "gpt2_hellaswag_acc.pt"

# Check if both values are cached
if gpt2_xl_hellaswag_acc_path.exists() and gpt2_hellaswag_acc_path.exists():
    gpt2_xl_hellaswag_acc = torch.load(gpt2_xl_hellaswag_acc_path)
    gpt2_hellaswag_acc = torch.load(gpt2_hellaswag_acc_path)
else:
    # gpt2 models
    gpt2_xl_model = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl", cache_dir=s.logs_path).to(s.device)
    gpt2_model = GPT2LMHeadModel.from_pretrained(
        "gpt2", cache_dir=s.logs_path).to(s.device)
    # Evaluate only if not cached
    gpt2_xl_hellaswag_acc = evaluate(gpt2_xl_model)
    gpt2_hellaswag_acc = evaluate(gpt2_model)

    if s.ddp_master_process:
        torch.save(gpt2_xl_hellaswag_acc, gpt2_xl_hellaswag_acc_path)
        torch.save(gpt2_hellaswag_acc, gpt2_hellaswag_acc_path)

if s.ddp_master_process:
    print(f"gpt2_xl_hellaswag_acc: {gpt2_xl_hellaswag_acc:.2f}")
    print(f"gpt2_hellaswag_acc: {gpt2_hellaswag_acc:.2f}")
    wandb.init(project="GPT-mini", config=s.config,
               dir=s.logs_path, mode=s.wandb_mode)
    wandb.log({"gpt2_xl_hellaswag_acc": gpt2_xl_hellaswag_acc})
    wandb.log({"gpt2_hellaswag_acc": gpt2_hellaswag_acc})

model = GPT().to(s.device)
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

try:
    # 🧪 Training Loop with WandB
    if s.ddp_master_process:
        print("Started Training!")
    for train_step in range(s.config["training"]["max_steps"]):
        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Training
        train_loss, gradient_norm = trainer.train_step()

        # Validation
        val_loss_accum = 0.0
        if train_step != 0 and train_step % s.config["training"]["val_interval"] == 0:
            val_steps = s.config["training"]["val_steps"]
            for _ in range(val_steps):
                val_loss = trainer.val_step()
                val_loss_accum += val_loss

            val_loss_accum = val_loss_accum / val_steps
            if s.ddp_master_process:
                print(f"val loss {val_loss_accum:.4f}")
                wandb.log({"val_loss": val_loss_accum,
                          "train_step": train_step})

            # Generation
            generations = generate(model)
            if s.ddp_master_process:
                wandb.log({"generations": generations,
                          "train_step": train_step})

            # Evaluate on hellaswag
            hellaswag_acc = evaluate(model)
            if s.ddp_master_process:
                wandb.log({"hellaswag_accuracy": hellaswag_acc,
                          "train_step": train_step})

        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        # Logging
        elapsed_time = end_time - start_time  # in seconds
        tokens_processed = train_dataloader.B * \
            train_dataloader.T * trainer.grad_accum_steps * s.ddp_world_size
        tokens_per_sec = tokens_processed / elapsed_time
        tokens_per_sec_str = ModelSummary.format_number(tokens_per_sec)

        if s.ddp_master_process:
            print(
                f"step {train_step:<3} | train_loss {train_loss:<5.2f} | norm {gradient_norm:<5.2f} | time {elapsed_time * 1000:<4.2f} ms | tok/sec {tokens_per_sec_str}")
            wandb.log({"train_loss": train_loss, "tok/sec": tokens_per_sec,
                      "gradient_norm": gradient_norm, "train_step": train_step})

            if train_step % 10 == 0:
                # Save for every 10 steps
                checkpoint = {
                    "train_step": train_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss_accum
                }

                torch.save(checkpoint, s.model_checkpoint_path)

except KeyboardInterrupt:
    if s.ddp_master_process:
        print("Stopping Run!")
finally:
    if s.ddp_master_process:
        wandb.log_model(s.model_checkpoint_path)
        wandb.finish()

    if s.is_ddp_available:
        dist.barrier()
        destroy_process_group()

    # Ensure all processes exit
    sys.exit(0)
