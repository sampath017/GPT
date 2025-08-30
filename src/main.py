import time
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import torch
import finetune.settings as s
from finetune.utils import Trainer, ModelSummary, instruct_generate, ModelCheckpointManager
from finetune.models import GPT
from finetune.LoRA import LoRALinear
from finetune.dataloader import UltraChat200kDataLoaderLite


def main():
    if s.is_ddp_available:
        print(s.ddp_global_rank, s.ddp_local_rank, s.ddp_world_size, s.device)

    train_dataloader = UltraChat200kDataLoaderLite(split="train")
    val_dataloader = UltraChat200kDataLoaderLite(split="val")

    # apply LoRA
    model, _ = ModelCheckpointManager.get_checkpoint_from_wandb(
        model=GPT(), model_type="pretrained")
    model = LoRALinear.apply_lora(model, r=1024*2, scale=1024*4, dropout=0.05,
                                  target_modules=("attn", "proj"))

    if s.ddp_master_process:
        # check trainable params
        trainable = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)")

    if not s.device == "cpu":
        model = torch.compile(model)
    if s.is_ddp_available:
        model = DDP(model, device_ids=[s.ddp_local_rank],
                    output_device=s.ddp_local_rank)
        raw_model = model.module
    else:
        raw_model = model

    optimizer = raw_model.configure_optimizers(
        weight_decay=s.config["optimizer"]["weight_decay"],
        lr=s.config["optimizer"]["max_lr"],
        betas=s.config["optimizer"]["betas"],
        eps=s.config["optimizer"]["eps"]
    )  # type: ignore

    trainer = Trainer(
        model,
        optimizer,
        dataloaders={
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader
        }
    )
    model_checkpoint_manager = ModelCheckpointManager()

    # 🧪 Training Loop with WandB
    if s.ddp_master_process:
        wandb_run = wandb.init(project="GPT3_124M_INSTRUCT", config=s.config,
                               dir=s.logs_root_path, mode=s.wandb_mode)  # type: ignore
        print("✅ Started Training!")

    for train_step in range(s.config["training"]["max_steps"]):
        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Training
        train_loss, gradient_norm = trainer.train_step(train_step)

        # Validation, Generation and checkpoint
        if train_step != 0 and train_step % s.config["training"]["val_interval"] == 0:
            val_loss = 0.0
            val_steps = s.config["training"]["val_steps"]
            for _ in range(val_steps):
                step_val_loss = trainer.val_step()
                val_loss += step_val_loss

            val_loss = val_loss / val_steps

            if s.ddp_master_process:
                print(f"val loss {val_loss:.4f}")
                generations = instruct_generate(model)
                print(generations)
                wandb.log({"generations": generations,
                           "train_step": train_step})
                wandb.log({"val_loss": val_loss,
                           "train_step": train_step})

                model_checkpoint_manager.save_checkpoint_to_wandb(
                    model, optimizer, train_step, train_loss, val_loss, wandb_run, model_type="finetuned")

        # Ensure previous CUDA ops are done
        if s.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        # Logging
        elapsed_time = end_time - start_time  # in seconds
        tokens_processed = train_dataloader.B * \
            train_dataloader.T * trainer.grad_accum_steps * s.ddp_world_size
        tokens_per_sec_number = tokens_processed / elapsed_time
        tokens_per_sec = ModelSummary.format_number(tokens_per_sec_number)

        if s.ddp_master_process:
            print(
                f"step {train_step:<3} | train_loss {train_loss:<5.2f} | norm {gradient_norm:<5.2f} | time {elapsed_time * 1000:<4.2f} ms | tok/sec {tokens_per_sec}")
            wandb.log({"train_loss": train_loss, "tok/sec": tokens_per_sec_number,
                       "gradient_norm": gradient_norm, "train_step": train_step})


if __name__ == "__main__":
    main()
