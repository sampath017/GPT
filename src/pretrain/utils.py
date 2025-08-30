from pathlib import Path
import wandb
import torch
import torch.nn.functional as F
import pretrain.settings as s
import math
import torch.distributed as dist


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
            if micro_step == (self.grad_accum_steps - 1):
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

    def save_checkpoint_to_wandb(self, model, optimizer, train_step, train_loss, val_loss):
        checkpoint_path = s.models_root_path / \
            f"model_checkpoint_train_step_{train_step}_val_loss_{val_loss:.2f}.pt"

        checkpoint = {
            "train_step": train_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
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
        current_paths = [p for _, p in self.best_models]
        for f in s.models_root_path.iterdir():
            if f not in current_paths:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass

    def cleanup_wandb_artifacts(self, run):
        # Get all model artifacts
        api = wandb.Api()
        artifacts = api.run(run.path).logged_artifacts()

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
    def load_checkpoint(path, model, optimizer=None):
        """Load checkpoint into model and optimizer."""
        checkpoint = torch.load(path, map_location=s.device)
        state_dict = checkpoint["model_state_dict"]

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

        # now load
        model.load_state_dict(new_state_dict)
        model = model.to(s.device)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer

    @staticmethod
    def get_model_from_wandb(model):
        run = wandb.init(dir=s.logs_root_path)
        # artifact = run.use_artifact(
        #     'sampath017/GPT3_124M/model_checkpoint_train_step_17000_val_loss_3.08:v0', type='model')
        # artifact_dir = artifact.download(s.models_root_path)
        artifact_dir = s.models_root_path
        checkpoint_files = list(Path(artifact_dir).glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pt files found in {artifact_dir}")

        checkpoint_path = checkpoint_files[0]
        print("Using checkpoint:", checkpoint_path)

        model, _ = ModelCheckpointManager.load_checkpoint(
            checkpoint_path, model)

        return model


def pretrain_generate(model):
    model.eval()
    num_return_sequences = 2
    max_length = 32
    tokens = s.enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(s.device)
    sample_rng = torch.Generator(device=s.device)
    sample_rng.manual_seed(42 + s.ddp_global_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                logits, loss = model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(
                topk_probs, 1, generator=sample_rng)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    generations = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = s.enc.decode(tokens)
        if s.ddp_world_size == 1:
            generation = f"sample {i}: {decoded}"
        else:
            generation = f"rank {s.ddp_global_rank} sample {i}: {decoded}"
        generations.append(generation)
        print(generation)

    return generations


def instruct_generate(model, max_length=256, temperature=0.7, top_p=0.9):
    model.eval()
    # always prepend <EOD> for fresh conversation
    prompt = "<EOD>User: Tell me a joke\nAssistant:"

    tokens = s.enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(s.device)

    xgen = tokens
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                logits, _ = model(xgen)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        # nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        sorted_probs[cutoff] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(sorted_probs, 1)
        next_token = torch.gather(sorted_indices, -1, next_token)

        xgen = torch.cat([xgen, next_token], dim=1)

        # optional EOS stop
        if next_token.item() == s.enc._special_tokens['<|endoftext|>']:
            break

    decoded = s.enc.decode(xgen[0].tolist())
    # only keep the assistant’s response
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1].strip()

    return decoded
