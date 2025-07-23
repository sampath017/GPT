import torch
import torch.nn.functional as F
import test_settings as s
import wandb
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

    def train_step(self):
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

        if s.ddp_master_process:
            wandb.log({"gradient_norm": norm})
        self.optimizer.step()

        return loss_accum.item(), norm

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            xb, yb = self.val_dataloader.next_batch()
            with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
                _, loss = self.model(xb, yb)

        if s.is_ddp_available:
            dist.all_reduce(loss.detach(), op=dist.ReduceOp.AVG)

        return loss.item()


class ModelSummary:
    def __init__(self, model):
        self.model = model

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

    def count_parameters(self):
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel()
                                   for p in self.model.parameters() if not p.requires_grad)

        print(
            f"Trainable parameters: {ModelSummary.format_number(trainable_params)}")
        print(
            f"Non-trainable parameters: {ModelSummary.format_number(non_trainable_params)}")

    def model_size(self):
        param_size = sum(param.nelement() * param.element_size()
                         for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size()
                          for buffer in self.model.buffers())
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

    def summary(self):
        self.model_size()
        self.count_parameters()


@torch.no_grad()
def accuracy(logits, y):
    probs = F.softmax(logits, dim=-1)
    y_pred = probs.argmax(dim=-1)
    accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

    return accuracy


def load_from_checkpoint(path, model, optimizer=None, lr_scheduler=None, device="cpu"):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler:
        lr_scheduler = lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return model, optimizer, lr_scheduler


def generate(model):
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
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = s.enc.decode(tokens)
        print(f"rank {s.ddp_global_rank} sample {i}: {decoded}")
