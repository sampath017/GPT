import settings as s
from torch.nn import functional as F
from models import GPT
from utils import accuracy, count_parameters, model_size
from torch.utils.data import DataLoader, random_split
from dataset import ShakespearDataset
import torch
from pathlib import Path
import sys
import os
import time
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

data_path = Path("../data")
logs_path = Path("../logs")
logs_path.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# cpu_count = os.cpu_count()
cpu_count = 7

dataset = ShakespearDataset(data_path/"shakespear.txt")

train_dataset, val_dataset = random_split(
    dataset, [s.dataset["train_split"], s.dataset["val_split"]]
)

train_dataloader = DataLoader(
    train_dataset, batch_size=s.dataset["batch_size"], shuffle=True, num_workers=cpu_count)
val_dataloader = DataLoader(
    val_dataset, batch_size=s.dataset["batch_size"],  num_workers=cpu_count)

model = GPT(device).to(device)
model = torch.compile(model)
model_size(model)
count_parameters(model)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device)

assert s.dataset["total_batch_size"] % (
    s.dataset["batch_size"] * s.dataset["context_size"]) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = s.dataset["total_batch_size"] // (
    s.dataset["batch_size"] * s.dataset["context_size"])
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_dataloader_iter = iter(train_dataloader)

ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
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
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


def main():
    for epoch in range(s.max_epochs):
        print(f"Epoch: {epoch} \n")

        for step in range(s.max_steps):
            t0 = time.time()
            optimizer.zero_grad()
            step_loss = 0.0
            for micro_step in range(grad_accum_steps):
                x, y = next(train_dataloader_iter)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / grad_accum_steps
                step_loss += loss.detach()
                loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            torch.cuda.synchronize()  # wait for the GPU to finish work
            t1 = time.time()
            dt = t1 - t0  # time difference in seconds
            tokens_processed = s.dataset["batch_size"] * \
                s.dataset["context_size"] * grad_accum_steps
            tokens_per_sec = tokens_processed / dt

            print(
                f"step {step:4d} | loss: {step_loss:.4f} | norm: {norm:.4f} | lr:{lr:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()
