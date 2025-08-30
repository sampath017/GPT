from pathlib import Path
from wandb import Api
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from finetune import LoRA
import finetune.settings as s
import math


class Trainer:
    def __init__(self, model, optimizer, dataloaders):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = dataloaders["train_dataloader"]
        self.val_dataloader = dataloaders["val_dataloader"]
        self.grad_accum_steps = int(s.config["dataset"]["total_batch_size"] / (
            s.config["dataset"]["batch_size"]*s.config["dataset"]["block_size"]*s.ddp_world_size))

        if s.ddp_master_process:
            print(
                f"total desired batch size: {s.config["dataset"]["total_batch_size"]} tokens.")
            print(
                f"calculated gradient accumulation steps: {self.grad_accum_steps}")

    def train_step(self, train_step):
        self.model.train()

        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):
            xb, yb = self.train_dataloader.next_batch()
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                _, loss = self.model(xb, yb)

            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()

            # ddp grads sync only for last micro step
            should_sync = (micro_step == self.grad_accum_steps -
                           1) or not s.is_ddp_available

            if should_sync:
                loss.backward()
            else:
                with self.model.no_sync():
                    loss.backward()

        if s.is_ddp_available:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=s.config["training"]["max_grad_norm"])

        # Perform a Optimization Step
        lr = Trainer.get_lr(train_step)
        if s.ddp_master_process:
            wandb.log({"train_step": train_step, "lr": lr})
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.step()

        return loss_accum.item(), norm  # type: ignore

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            xb, yb = self.val_dataloader.next_batch()
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                _, loss = self.model(xb, yb)

        if s.is_ddp_available:
            dist.all_reduce(loss.detach(), op=dist.ReduceOp.AVG)

        return loss.item()

    @staticmethod
    def get_lr(train_step):
        # 1) linear warmup for warmup_iters steps
        if train_step < s.config["optimizer"]["warmup_steps"]:
            return s.config["optimizer"]["max_lr"] * (train_step+1) / s.config["optimizer"]["warmup_steps"]

        # 2) in between, use cosine decay down to min learning rate
        decay_ratio = (train_step - s.config["optimizer"]["warmup_steps"]) / (
            s.config["training"]["max_steps"] - s.config["optimizer"]["warmup_steps"])
        assert 0 <= decay_ratio <= 1, "train_step should be greater than warmup steps"

        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        # 3) if train_step > lr_decay_iters, return min learning rate
        if train_step > s.config["training"]["max_steps"]:
            return s.config["optimizer"]["min_lr"]

        return s.config["optimizer"]["min_lr"] + coeff * (s.config["optimizer"]["max_lr"] - s.config["optimizer"]["min_lr"])


class ModelSummary:
    @staticmethod
    def format_number(num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)

    @staticmethod
    def count_parameters(model):
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel()
                                   for p in model.parameters() if not p.requires_grad)

        print(
            f"Trainable parameters: {ModelSummary.format_number(trainable_params)}")
        print(
            f"Non-trainable parameters: {ModelSummary.format_number(non_trainable_params)}")

    @staticmethod
    def model_size(model):
        param_size = sum(param.nelement() * param.element_size()
                         for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size()
                          for buffer in model.buffers())
        total_size = param_size + buffer_size

        def format_size(size_bytes):
            if size_bytes >= 1_073_741_824:  # 1024^3
                return f"{size_bytes / 1_073_741_824:.2f} GB"
            elif size_bytes >= 1_048_576:  # 1024^2
                return f"{size_bytes / 1_048_576:.2f} MB"
            elif size_bytes >= 1024:  # 1024^1
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes} bytes"

        size_all = format_size(total_size)
        print(f"Model size: {size_all}")

    @staticmethod
    def summary(model):
        ModelSummary.model_size(model)
        ModelSummary.count_parameters(model)


@torch.no_grad()
def accuracy(logits, y):
    probs = F.softmax(logits, dim=-1)
    y_pred = probs.argmax(dim=-1)
    accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

    return accuracy


class ModelCheckpointManager:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.best_models = []  # list of (val_loss, path)

    def save_checkpoint_to_wandb(self, model, optimizer, train_step, train_loss, val_loss, wandb_run, model_type="pretrained"):
        self.run = wandb_run
        assert model_type in ["pretrained", "finetuned"]
        checkpoint_path = s.models_root_path / \
            f"{model_type}/model_checkpoint_train_step_{train_step}_val_loss_{val_loss:.2f}.pt"

        model_state_dict = model.state_dict(
        ) if model_type == "pretrained" else LoRA.get_state_dict()  # type: ignore # nopep8
        checkpoint = {
            "train_step": train_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)

        # Log to wandb
        artifact = wandb.Artifact(
            f"model_checkpoint_train_step_{train_step}_val_loss_{val_loss:.2f}", type="model",
            metadata={"train_step": train_step}
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)

        # Track top-K
        self.best_models.append((train_step, val_loss, checkpoint_path))
        self.best_models.sort(key=lambda x: x[0])
        self.best_models = self.best_models[:self.top_k]

        # Delete local files not in top-K
        current_paths = [p for _, _, p in self.best_models]
        for f in s.models_root_path.iterdir():
            if f.is_file() and (f not in current_paths):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass

        self._cleanup_wandb_artifacts()

    def _cleanup_wandb_artifacts(self):
        # Get all model artifacts
        api = wandb.Api()
        artifacts = api.run(self.run.path).logged_artifacts()

        top_k_train_steps = [
            train_step for train_step, _, _ in self.best_models]

        for artifact in artifacts:
            if artifact.metadata["train_step"] not in top_k_train_steps:
                try:
                    # remove all aliases first
                    for alias in list(artifact.aliases):
                        artifact.aliases.remove(alias)

                    artifact.save()
                    artifact.delete()
                except Exception as e:
                    print(f"Failed to delete artifact {artifact.name}: {e}")

    @staticmethod
    def load_checkpoint_from_local(path, model, optimizer=None, model_type="pretrained"):
        """Load checkpoint into model and optimizer."""
        checkpoint = torch.load(path, map_location=s.device)
        state_dict = checkpoint["model_state_dict"]

        if model_type == "pretrained":
            # 🔑 Fix DDP/FSDP prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                # remove "module._orig_mod." or "module." prefix if present
                if k.startswith("module._orig_mod."):
                    new_state_dict[k.replace("module._orig_mod.", "", 1)] = v
                elif k.startswith("module."):
                    new_state_dict[k.replace("module.", "", 1)] = v
                else:
                    new_state_dict[k] = v
        elif model_type == "finetuned":
            new_state_dict = state_dict

        # now load
        model.load_state_dict(new_state_dict)
        model = model.to(s.device)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer

    @staticmethod
    def get_checkpoint_from_wandb(model, wandb_path='sampath017/GPT3_124M/model_checkpoint_train_step_17000_val_loss_3.08:v0', cache_dir=s.models_root_path/"pretrained_models", model_type="pretrained"):
        if s.ddp_master_process:  # Only master downloads
            api = Api()
            artifact = api.artifact(
                wandb_path,
                type='model'
            )
            artifact.download(cache_dir)

        # 🔑 Wait for rank 0 to finish downloading
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        checkpoint_files = list(Path(cache_dir).glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No .pt files found in {cache_dir}"
            )

        # Pick most recent checkpoint
        checkpoint_path = max(
            checkpoint_files, key=lambda p: p.stat().st_mtime)
        if s.ddp_master_process:
            print(f"Using checkpoint: {checkpoint_path}")

        # Load checkpoint on each rank
        model, optimizer = ModelCheckpointManager.load_checkpoint_from_local(
            path=checkpoint_path, model=model, model_type=model_type
        )

        return model, optimizer


def instruct_generate(model, prompt, max_length=256):
    model.eval()
    # always prepend <EOD> for fresh conversation
    prompt = f"User: {prompt}\nAssistant:"

    tokens = s.enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(s.device)

    xgen = tokens
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                logits, _ = model(xgen)

        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)

        # optional EOS stop
        if xcol.item() == s.enc._special_tokens['<|endoftext|>']:
            break

    decoded = s.enc.decode(xgen[0].tolist())
    # only keep the assistant’s response
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1].strip()

    return decoded
