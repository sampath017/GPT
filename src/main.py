import settings as s
from torch.nn import functional as F
from models import GPT
from utils import accuracy, count_parameters, model_size
from dataset import DataLoaderLite
import torch
from pathlib import Path
import sys
import os
import time
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


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


ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    # ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_rank}'
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
    print(f"using device: {device}")


# create model
model = GPT(device, master_process)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_rank])
raw_model = model.module if ddp else model

if master_process:
    model_size(raw_model)
    count_parameters(raw_model)

train_loader = DataLoaderLite(s.data_path, process_rank=ddp_rank,
                              num_processes=ddp_world_size, split="train", master_process=master_process)

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device)

assert s.dataset["total_batch_size"] % (s.dataset["batch_size"] * s.dataset["context_size"]
                                        * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = s.dataset["total_batch_size"] // (
    s.dataset["batch_size"] * s.dataset["context_size"] * ddp_world_size)
if master_process:
    print(f"total desired batch size: {s.dataset['total_batch_size']}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


def main():
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        step_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
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
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1)
                loss.backward()

        if ddp:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)

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
            s.dataset["context_size"] * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:4d} | loss: {step_loss:.4f} | norm: {norm:.4f} | lr:{lr:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    if ddp:
        destroy_process_group()

# poetry run torchrun --standalone --nproc_per_node=2 src/main.py


if __name__ == "__main__":
    main()

