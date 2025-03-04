import torch
import torch.nn.functional as F


def count_parameters(model):
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel()
                               for p in model.parameters() if not p.requires_grad)

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
    print(f"Non-trainable parameters: {format_number(non_trainable_params)}")


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


@torch.no_grad()
def accuracy(logits, y):
    probs = F.softmax(logits, dim=-1)
    y_pred = probs.argmax(dim=-1)
    accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

    return accuracy


def load_from_checkpoint(path, model, optimizer=None, lr_scheduler=None, device="cpu"):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if lr_scheduler:
        lr_scheduler = lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return model, optimizer, lr_scheduler
