import torch
import torch.nn.functional as F
import time
import settings as s


class Trainer:
    def __init__(self, model, optimizer, dataloaders):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = dataloaders["train_dataloader"]
        self.val_dataloader = dataloaders["val_dataloader"]

    def train_step(self):
        # Ensure previous CUDA ops are done
        torch.cuda.synchronize()
        start_time = time.time()

        self.model.train()
        xb, yb = self.train_dataloader.next_batch()

        with torch.autocast(device_type=s.device, dtype=torch.bfloat16):
            _, loss = self.model(xb, yb)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync again to wait for CUDA ops to finish
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time

        return loss.item(), elapsed_time

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            xb, yb = self.val_dataloader.next_batch()
            _, loss = self.model(xb, yb)

        return loss.item()


class ModelSummary:
    def __init__(self, model):
        self.model = model

    def count_parameters(self):
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel()
                                   for p in self.model.parameters() if not p.requires_grad)

        def format_number(num):
            if num >= 1_000_000_000:
                return f"{num / 1_000_000_000:.2f}B"
            elif num >= 1_000_000:
                return f"{num / 1_000_000:.2f}M"
            elif num >= 1_000:
                return f"{num / 1_000:.2f}K"
            else:
                return str(num)

        print(f"Trainable parameters: {format_number(trainable_params)}")
        print(
            f"Non-trainable parameters: {format_number(non_trainable_params)}")

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
